/**
 * backend.ts — typed wrapper around the WASM TinyGPT module (Phase 4).
 *
 * The kernels + model are compiled by wasm/build_wasm.sh into the ES module
 * browser/public/tinygpt.js. This file loads that module and exposes its C-ABI
 * (model.h) as a small typed class, handling the JS<->WASM memory marshalling
 * so the Worker can stay readable.
 *
 * Guide: docs/browser_notes.md ("WASM backend")
 */

/**
 * The subset of the Emscripten module surface we rely on.
 *
 * Memory64 ABI note: when the module is built with -sMEMORY64=1 + -sWASM_BIGINT,
 * any C function that takes or returns a pointer (`void*`, `T*`, …) uses i64
 * for that arg, which JS sees as `BigInt`. Plain `int` / `float` args stay
 * Number. The wrappers below either bridge those types or operate on the raw
 * exports directly. `_malloc` happens to return Number even on MEM64 (the
 * compiler narrows it because the heap fits in safe-integer range).
 */
type Ptr = number | bigint;
interface RawExports {
  _malloc(bytes: number | bigint): number;
  _free(ptr: Ptr): void;
  _tg_model_create(vocab: number, ctx: number, layers: number, heads: number, dModel: number, dMlp: number, seed: number): Ptr;
  _tg_model_free(handle: Ptr): void;
  _tg_model_num_params(handle: Ptr): number;
  _tg_model_step(handle: Ptr): number;
  _tg_set_data(handle: Ptr, data: Ptr, len: number, trainFrac: number): void;
  _tg_train_step(handle: Ptr, batch: number, lr: number, gradClip: number): number;
  _tg_eval(handle: Ptr, split: number, batch: number, nBatches: number): number;
  _tg_generate(handle: Ptr, prompt: Ptr, plen: number, out: Ptr, maxNew: number, temp: number, topK: number, seed: number): number;
  _matmul_forward(a: Ptr, b: Ptr, c: Ptr, M: number, K: number, N: number): void;
  _tg_state_bytes(handle: Ptr): number;
  _tg_export_state(handle: Ptr, buf: Ptr): void;
  _tg_import_state(handle: Ptr, buf: Ptr): void;
}
interface WasmModule extends RawExports {
  HEAPU8: Uint8Array;
  HEAPF32: Float32Array;
}

interface ModuleOpts {
  locateFile?: (path: string) => string;
  // Emscripten pthread builds spawn workers via `new Worker(new URL(import.meta.url))`.
  // When the module is loaded via blob URL, this would point at the blob (one-shot),
  // breaking thread spawn. Setting `mainScriptUrlOrBlob` overrides it to the real path.
  mainScriptUrlOrBlob?: string;
}
type Factory = (opts?: ModuleOpts) => Promise<WasmModule>;

/**
 * Where the compiled module is served from (see wasm/build_wasm.sh /
 * build_wasm64.sh output). We ship two builds:
 *
 *   tinygpt64.{js,wasm}  — Memory64 build, 64-bit pointers, heap > 4 GB.
 *                         Chromium 133+, Firefox 134+. Used by default since
 *                         we're targeting Chromium-only.
 *   tinygpt.{js,wasm}    — 32-bit fallback, capped at 4 GB heap (~250M params
 *                         fp32). Kept as a static-safety net.
 *
 * Memory64 detection: WebAssembly.Memory accepts `address: "i64"` only on
 * runtimes that support it. We probe once and cache.
 *
 * Files in /public can't be imported from source code under Vite 5+, so we
 * fetch the module text and import it via a blob URL. `locateFile` tells
 * Emscripten where to find the .wasm regardless of import.meta.url.
 */
const JS_URL_64 = "/tinygpt64.js";
const WASM_URL_64 = "/tinygpt64.wasm";
const JS_URL_32 = "/tinygpt.js";
const WASM_URL_32 = "/tinygpt.wasm";

function supportsMemory64(): boolean {
  // The Memory64 descriptor field changed names during the proposal. Newer
  // spec uses `address`; earlier Chromium implementations (and what Playwright
  // bundles as of late 2026) still ship `index`. Try both.
  for (const key of ["address", "index"] as const) {
    try {
      new WebAssembly.Memory({ initial: 1, maximum: 1, shared: false, [key]: "i64" } as unknown as WebAssembly.MemoryDescriptor);
      return true;
    } catch {
      // Try the next spelling.
    }
  }
  return false;
}

const MEM64 = supportsMemory64();
const JS_URL = MEM64 ? JS_URL_64 : JS_URL_32;
const WASM_URL = MEM64 ? WASM_URL_64 : WASM_URL_32;
export const usingMemory64 = MEM64;

let factoryPromise: Promise<Factory> | undefined;
async function loadFactory(): Promise<Factory> {
  if (!factoryPromise) {
    factoryPromise = (async () => {
      const res = await fetch(JS_URL);
      if (!res.ok) throw new Error(`failed to load ${JS_URL}: ${res.status}`);
      const code = await res.text();
      const blob = new Blob([code], { type: "application/javascript" });
      const url = URL.createObjectURL(blob);
      try {
        const mod = (await import(/* @vite-ignore */ url)) as { default: Factory };
        return mod.default;
      } finally {
        URL.revokeObjectURL(url);
      }
    })();
  }
  return factoryPromise;
}

/** Pointer argument wrapper. On the 32-bit module pointers are Number; on
 * Memory64 they're BigInt. _malloc returns Number on both ABIs, so we coerce
 * before passing to other C functions. */
const toPtr: (x: number) => Ptr = MEM64 ? (x) => BigInt(x) : (x) => x;

export class TinyGptBackend {
  private constructor(private readonly m: WasmModule) {}

  /** Load and instantiate the compiled WASM module. */
  static async load(): Promise<TinyGptBackend> {
    const factory = await loadFactory();
    const m = await factory({
      locateFile: (p) => (p.endsWith(".wasm") ? WASM_URL : p),
      mainScriptUrlOrBlob: JS_URL,
    });
    return new TinyGptBackend(m as WasmModule);
  }

  /** Raw C = A @ B via the WASM matmul kernel — the WebGPU parity reference. */
  matmul(a: Float32Array, b: Float32Array, M: number, K: number, N: number): Float32Array {
    const aPtr = this.m._malloc(M * K * 4);
    const bPtr = this.m._malloc(K * N * 4);
    const cPtr = this.m._malloc(M * N * 4);
    try {
      this.m.HEAPF32.set(a, aPtr >> 2);
      this.m.HEAPF32.set(b, bPtr >> 2);
      this.m._matmul_forward(toPtr(aPtr), toPtr(bPtr), toPtr(cPtr), M, K, N);
      return this.m.HEAPF32.slice(cPtr >> 2, (cPtr >> 2) + M * N);
    } finally {
      this.m._free(toPtr(aPtr));
      this.m._free(toPtr(bPtr));
      this.m._free(toPtr(cPtr));
    }
  }

  createModel(cfg: {
    ctx: number;
    layers: number;
    heads: number;
    dModel: number;
    dMlp: number;
    seed: number;
  }): TinyGptModel {
    const handle = this.m._tg_model_create(
      256, cfg.ctx, cfg.layers, cfg.heads, cfg.dModel, cfg.dMlp, cfg.seed,
    );
    // handle is Number on 32-bit, BigInt on MEM64. Compare loosely (0 ⇔ 0n).
    if (handle === 0 || handle === 0n) {
      throw new Error("tg_model_create failed (d_model % heads != 0?)");
    }
    return new TinyGptModel(this.m, handle);
  }
}

/** A live model handle. One per training run. Pointer-shaped values
 * (handle, ptr returns) are typed as `Ptr` because they live in the runtime's
 * native pointer type — Number on the 32-bit build, BigInt on Memory64.
 * HEAP slice/index math always operates in Number space via toNum(). */
export class TinyGptModel {
  constructor(
    private readonly m: WasmModule,
    private readonly handle: Ptr,
  ) {}

  numParams(): number {
    return this.m._tg_model_num_params(this.handle);
  }

  step(): number {
    return this.m._tg_model_step(this.handle);
  }

  /** Attach a byte-token corpus, split train/val by `trainFrac`. */
  setData(tokens: Uint8Array, trainFrac: number): void {
    const ptr = this.m._malloc(Math.max(1, tokens.length));
    this.m.HEAPU8.set(tokens, ptr);
    try {
      this.m._tg_set_data(this.handle, toPtr(ptr), tokens.length, trainFrac);
    } finally {
      this.m._free(toPtr(ptr));
    }
  }

  /** One AdamW step on a random batch; returns the batch loss. */
  trainStep(batchSize: number, lr: number, gradClip: number): number {
    return this.m._tg_train_step(this.handle, batchSize, lr, gradClip);
  }

  /** Average loss over `nBatches`. split: 0 = train, 1 = val. */
  evalLoss(split: 0 | 1, batchSize: number, nBatches: number): number {
    return this.m._tg_eval(this.handle, split, batchSize, nBatches);
  }

  /** Autoregressive generation. temperature <= 0 is greedy. */
  generate(
    prompt: Uint8Array,
    maxNew: number,
    temperature: number,
    topK: number,
    seed: number,
  ): Uint8Array {
    const promptPtr = this.m._malloc(Math.max(1, prompt.length));
    this.m.HEAPU8.set(prompt, promptPtr);
    const outPtr = this.m._malloc(Math.max(1, maxNew));
    try {
      const n = this.m._tg_generate(
        this.handle, toPtr(promptPtr), prompt.length, toPtr(outPtr),
        maxNew, temperature, topK, seed,
      );
      return this.m.HEAPU8.slice(outPtr, outPtr + n);
    } finally {
      this.m._free(toPtr(promptPtr));
      this.m._free(toPtr(outPtr));
    }
  }

  /** Serialise weights + AdamW moments + step for checkpointing. */
  exportState(): Uint8Array {
    const bytes = this.m._tg_state_bytes(this.handle);
    const ptr = this.m._malloc(bytes);
    try {
      this.m._tg_export_state(this.handle, toPtr(ptr));
      return this.m.HEAPU8.slice(ptr, ptr + bytes);
    } finally {
      this.m._free(toPtr(ptr));
    }
  }

  /** Load a state blob from exportState() — model config must match. */
  importState(state: Uint8Array): void {
    const ptr = this.m._malloc(Math.max(1, state.length));
    this.m.HEAPU8.set(state, ptr);
    try {
      this.m._tg_import_state(this.handle, toPtr(ptr));
    } finally {
      this.m._free(toPtr(ptr));
    }
  }

  free(): void {
    this.m._tg_model_free(this.handle);
  }
}
