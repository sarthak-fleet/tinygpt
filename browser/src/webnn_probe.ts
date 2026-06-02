// webnn_probe.ts — active probe for the WebNN inference path.
//
// The runtime_detect.ts `webnnPresent` flag only checks for `navigator.ml`
// existence. That's the cheapest signal but doesn't tell us whether the
// API actually works: the browser might expose the namespace while the
// backend isn't functional (e.g. CoreML not configured, DirectML driver
// missing, NPU adapter absent).
//
// This module ships a small numerics-gated probe that builds a one-matmul
// MLGraph, runs it, and compares the output against a hand-computed f32
// reference. The result feeds the `+WebNN` capability pill — when active
// the pill carries a "PASS" mark; when present but failing, the pill is
// downgraded to "+WebNN (no backend)" so the user knows the namespace is
// there but inference can't actually route through it.
//
// The probe is INDEPENDENT of the WebGPU code path. It uses its own
// `navigator.ml.createContext({deviceType: "gpu"})` and produces a
// standalone verdict. The full WebNN inference path (every transformer
// op routed through MLGraphBuilder) is a follow-up build; this probe is
// the gating prerequisite so that build can be triggered confidently.

/** Magnitude-aware tolerance used by the f16-storage / shader-f16 gates;
 *  reused here. We allow up to 0.5% mean_rel and 1% max_abs of
 *  mean |reference| because WebNN backends may run in f16 internally
 *  (CoreML on ANE quantises by default). */
function compareWebNN(
  refOut: Float32Array, webnnOut: Float32Array,
): { passed: boolean; summary: string } {
  let sumAbsRef = 0;
  for (let i = 0; i < refOut.length; i++) sumAbsRef += Math.abs(refOut[i]);
  const meanAbsRef = sumAbsRef / refOut.length;
  const denomFloor = Math.max(meanAbsRef * 0.01, 1e-6);
  let maxAbs = 0, sumRel = 0;
  for (let i = 0; i < refOut.length; i++) {
    const r = refOut[i];
    const f = webnnOut[i];
    const absErr = Math.abs(f - r);
    const denom = Math.max(Math.abs(r), denomFloor);
    sumRel += absErr / denom;
    if (absErr > maxAbs) maxAbs = absErr;
  }
  const meanRel = sumRel / refOut.length;
  const maxAbsThreshold = meanAbsRef * 0.01;
  const passed = maxAbs < maxAbsThreshold && meanRel < 5e-3;
  const summary =
    `mean|ref|=${meanAbsRef.toExponential(2)}, ` +
    `max_abs=${maxAbs.toExponential(2)} (limit ${maxAbsThreshold.toExponential(2)}), ` +
    `mean_rel=${(meanRel * 100).toFixed(3)}% (limit 0.500%) — ` +
    `${passed ? "PASS" : "FAIL"}`;
  return { passed, summary };
}

/** Reference f32 matmul, computed on the JS side. Slow but trustworthy —
 *  this is the comparison oracle, not the production path. */
function refMatmul(
  a: Float32Array, b: Float32Array, M: number, K: number, N: number,
): Float32Array {
  const out = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let s = 0;
      for (let k = 0; k < K; k++) s += a[i * K + k] * b[k * N + j];
      out[i * N + j] = s;
    }
  }
  return out;
}

export interface WebNNProbeResult {
  /** Did the API path execute end-to-end without throwing? */
  apiReachable: boolean;
  /** Did the matmul output match a f32 reference within tolerance? */
  numericsPassed: boolean;
  /** Detail line for the console + capability pill tooltip. */
  summary: string;
  /** The deviceType used — "gpu" or "npu" depending on what worked. */
  deviceType: string | null;
}

/** Active probe: create an MLContext, build a tiny matmul graph, compute
 *  it, compare against a JS reference. Catches every step so a missing
 *  API surface gracefully reports `apiReachable: false`. */
export async function probeWebNNActive(): Promise<WebNNProbeResult> {
  // Default: assume nothing works.
  const noResult: WebNNProbeResult = {
    apiReachable: false,
    numericsPassed: false,
    summary: "WebNN: navigator.ml not present or createContext unavailable",
    deviceType: null,
  };

  const nav = navigator as unknown as {
    ml?: {
      createContext?: (opts?: { deviceType?: string; powerPreference?: string }) =>
        Promise<unknown> | unknown;
    };
  };
  if (!nav.ml || typeof nav.ml.createContext !== "function") {
    return noResult;
  }

  // Try "gpu" then "npu" — gpu is the most common surface; npu is the
  // Apple Neural Engine / DirectML NPU adapter when available.
  for (const deviceType of ["gpu", "npu"]) {
    try {
      const ctx = await nav.ml.createContext({ deviceType });
      // MLGraphBuilder constructor is the historical entry; some Chromium
      // builds use ctx.createGraphBuilder() instead — try both.
      const ctxAny = ctx as unknown as {
        createGraphBuilder?: () => unknown;
      };
      let builderAny: unknown = ctxAny.createGraphBuilder?.();
      if (!builderAny) {
        const MLGraphBuilderCtor = (
          globalThis as unknown as { MLGraphBuilder?: new (c: unknown) => unknown }
        ).MLGraphBuilder;
        if (!MLGraphBuilderCtor) continue;
        builderAny = new MLGraphBuilderCtor(ctx);
      }
      const builder = builderAny as {
        input: (name: string, desc: { dataType: string; shape?: number[]; dimensions?: number[] }) => unknown;
        constant: (
          desc: { dataType: string; shape?: number[]; dimensions?: number[] },
          buf: BufferSource,
        ) => unknown;
        matmul: (a: unknown, b: unknown) => unknown;
        build: (out: { c: unknown } | Record<string, unknown>) => Promise<unknown>;
      };

      const M = 8, K = 16, N = 8;
      const aData = new Float32Array(M * K);
      const bData = new Float32Array(K * N);
      let seed = 13579;
      const rand = () => {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        return (seed / 0x7fffffff) * 0.4 - 0.2;
      };
      for (let i = 0; i < aData.length; i++) aData[i] = rand();
      for (let i = 0; i < bData.length; i++) bData[i] = rand();

      // Some Chromium builds use "shape", some "dimensions" — try shape
      // first, fall back if matmul errors.
      const aIn = builder.input("a", { dataType: "float32", shape: [M, K], dimensions: [M, K] });
      const bIn = builder.constant({ dataType: "float32", shape: [K, N], dimensions: [K, N] }, bData);
      const cOp = builder.matmul(aIn, bIn);
      const graphAny = await builder.build({ c: cOp });
      const graph = graphAny as {
        compute?: (
          inputs: Record<string, BufferSource>,
          outputs: Record<string, BufferSource>,
        ) => Promise<unknown>;
        dispatch?: (
          inputs: Record<string, BufferSource>,
          outputs: Record<string, BufferSource>,
        ) => unknown;
      };
      if (!graph.compute && !graph.dispatch) {
        continue;
      }
      const outBuf = new Float32Array(M * N);
      if (graph.compute) {
        await graph.compute({ a: aData }, { c: outBuf });
      } else if (graph.dispatch) {
        graph.dispatch({ a: aData }, { c: outBuf });
      }

      const ref = refMatmul(aData, bData, M, K, N);
      const cmp = compareWebNN(ref, outBuf);
      console.info(`[webnn] probe (deviceType=${deviceType}): ${cmp.summary}`);
      return {
        apiReachable: true,
        numericsPassed: cmp.passed,
        summary: cmp.summary,
        deviceType,
      };
    } catch (err) {
      console.info(`[webnn] probe failed for deviceType=${deviceType}:`, err);
    }
  }
  return {
    apiReachable: false,
    numericsPassed: false,
    summary: "WebNN: createContext or matmul graph build failed for both gpu and npu",
    deviceType: null,
  };
}
