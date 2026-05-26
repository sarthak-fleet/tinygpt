// smoke_wasm64_node.mjs — verify the Memory64 build trains end-to-end.
//
// Tests two things the Phase 4 32-bit smoke test can't:
//   1. The Memory64 ABI flows correctly: i64 pointers are BigInt, i32 scalars
//      are Number, _malloc returns Number. We call the raw exports directly
//      because cwrap with "number" doesn't auto-convert between them.
//   2. The Behemoth-shaped config (24L · d=1280 · ctx=512 → ~473M params)
//      both allocates AND advances the loss — the 32-bit module OOMs at this
//      size, proving the Memory64 ceiling-break is real.
//
// Run:  node tests/smoke_wasm64_node.mjs

import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const modPath = path.join(here, "..", "browser", "public", "tinygpt64.js");

const { default: createTinyGPT } = await import(modPath);
const M = await createTinyGPT();

// Raw exports (Memory64-aware ABI: pointers as BigInt, scalars as Number).
const create = M._tg_model_create;
const numParams = M._tg_model_num_params;
const setData = M._tg_set_data;
const trainStep = M._tg_train_step;
const evalLoss = M._tg_eval;
const freeModel = M._tg_model_free;

let failed = 0;
const check = (name, ok, detail) => {
  console.log(`${ok ? "ok  " : "FAIL"} ${name.padEnd(40)} (${detail})`);
  if (!ok) failed++;
};

// ========================================================================
// Round 1: small model — proves the 64-bit module's basic plumbing works.
// ========================================================================
{
  console.log("\n--- round 1: small (~360k params) ---");
  // C signature: (vocab, ctx, layers, heads, dModel, dMlp, seed)
  const model = create(256, 64, 3, 3, 96, 384, 42);
  check("tg_model_create returns nonzero", model !== 0n, model.toString());
  const params = numParams(model);
  check("tg_model_num_params is small-shaped", params > 300_000 && params < 500_000, params.toLocaleString("en-US"));

  const text = "the quick brown fox jumps over the lazy dog. ".repeat(70);
  const bytes = new TextEncoder().encode(text);
  const dataPtr = M._malloc(bytes.length); // Number under MEM64.
  M.HEAPU8.set(bytes, dataPtr);
  // setData wants (TgModel handle [i64], const char* data [i64], int len, float frac).
  setData(model, BigInt(dataPtr), bytes.length, 0.9);
  M._free(dataPtr);

  const initLoss = evalLoss(model, 0, 8, 4);
  check("initial loss near ln(256)", Math.abs(initLoss - 5.545) < 0.7, initLoss.toFixed(4));

  let loss = initLoss;
  for (let step = 1; step <= 200; step++) {
    loss = trainStep(model, 8, 1e-3, 1.0);
  }
  check("loss fell below initial", loss < initLoss * 0.4, loss.toFixed(4));
  freeModel(model);
}

// ========================================================================
// Round 2: Behemoth — the ceiling-break demo.
//   24 layers · d=1280 · ctx=512 · 20 heads → ~473M params
//   fp32 weights + AdamW state ≈ 5.6 GB heap. The 32-bit module OOMs here.
// ========================================================================
{
  console.log("\n--- round 2: behemoth (~473M params, needs Memory64) ---");
  const t0 = Date.now();
  // C signature: (vocab, ctx, layers, heads, dModel, dMlp, seed)
  const model = create(256, 512, 24, 20, 1280, 5120, 42);
  const allocMs = Date.now() - t0;
  check("behemoth alloc succeeded", model !== 0n, `handle=${model.toString()} in ${allocMs} ms`);
  if (model === 0n) {
    console.log("    bailing — alloc failed");
    process.exit(1);
  }
  const params = numParams(model);
  check("behemoth params ≈ 470M", params > 4.5e8 && params < 5e8, params.toLocaleString("en-US"));

  // Tiny corpus — we want one train step to land, not convergence.
  const text = "the quick brown fox jumps over the lazy dog. ".repeat(60);
  const bytes = new TextEncoder().encode(text);
  const dataPtr = M._malloc(bytes.length);
  M.HEAPU8.set(bytes, dataPtr);
  setData(model, BigInt(dataPtr), bytes.length, 0.9);
  M._free(dataPtr);

  console.log("    running 1 forward+backward+step on the 473M-param model...");
  const tStep = Date.now();
  // batch 1, ctx 512 — the smallest possible footprint at this shape.
  const loss = trainStep(model, 1, 1e-4, 1.0);
  const stepMs = Date.now() - tStep;
  check("one training step completed", Number.isFinite(loss), `${loss.toFixed(4)} in ${(stepMs / 1000).toFixed(1)} s`);
  check("loss is finite and roughly sane", loss > 0 && loss < 10, loss.toFixed(4));

  freeModel(model);
  console.log("    freed cleanly");
}

console.log(failed === 0 ? "\nMemory64 smoke test passed" : "\nMemory64 SMOKE TEST FAILED");
process.exit(failed === 0 ? 0 : 1);
