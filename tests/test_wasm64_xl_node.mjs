// test_wasm64_xl_node.mjs — does the Memory64 build actually run XL in Node?
//
// The existing tests/bench_wasm.mjs loads the 32-bit module, so the 64-bit
// pthread + Memory64 path has been completely untested in Node. If this fails
// the same way the browser does, the OOB is a kernel/build bug — not a
// browser-only pthread + SharedArrayBuffer issue.

import path from "node:path";
import { fileURLToPath } from "node:url";
const here = path.dirname(fileURLToPath(import.meta.url));

const { default: createTinyGPT64 } = await import(
  path.join(here, "..", "browser", "public", "tinygpt64.js")
);
console.log("loading tinygpt64.wasm in Node…");
const M = await createTinyGPT64({
  locateFile: (f) => path.join(here, "..", "browser", "public", f),
});
console.log("module loaded.");

// In the 64-bit build, pointer-sized args are i64 (BigInt) per Emscripten's
// -sWASM_BIGINT. cwrap with "bigint" passes BigInts through.
const P = "bigint"; // i64 for MEMORY64 build
const N = "number";
const create    = M.cwrap("tg_model_create",    P, [N, N, N, N, N, N, N]);
const numParams = M.cwrap("tg_model_num_params", N, [P]);
const setData   = M.cwrap("tg_set_data",        null, [P, P, P, N]);
const trainStep = M.cwrap("tg_train_step",      N,    [P, N, N, N]);
const freeModel = M.cwrap("tg_model_free",      null, [P]);

const text = "the quick brown fox jumps over the lazy dog. ".repeat(400);
const bytes = new TextEncoder().encode(text);

console.log(`malloc(${bytes.length}) …`);
const dataPtr = M._malloc(BigInt(bytes.length));
console.log(`  dataPtr=${dataPtr}`);
M.HEAPU8.set(bytes, Number(dataPtr));

// XL: vocab=256, ctx=128, layers=8, heads=8, d=256, dMlp=1024, batch=8, seed=42
console.log("create XL model: ctx=128 layers=8 d=256 batch=8 …");
const t0 = Date.now();
const model = create(256, 128, 8, 8, 256, 1024, 42);
console.log(`  model=${model}  (alloc ${Date.now() - t0}ms)`);
if (!model) {
  console.log("FAIL: create returned 0");
  process.exit(2);
}
const params = Number(numParams(model));
console.log(`  params: ${(params / 1e6).toFixed(2)}M`);

setData(model, dataPtr, BigInt(bytes.length), 0.9);
console.log("setData OK");

console.log("running 1 training step (lr=3e-4) …");
try {
  const t1 = Date.now();
  trainStep(model, 8, 3e-4, 1.0);
  console.log(`  step took ${Date.now() - t1}ms`);
  console.log("PASS — tinygpt64.wasm trains XL cleanly in Node.");
} catch (e) {
  console.log("FAIL during tg_train_step:", e?.message ?? e);
  console.log("→ The 64-bit module itself OOBs at XL shape in Node too.");
  console.log("→ The bug is in the 64-bit kernels / build, not browser-only pthread.");
  process.exit(1);
}

freeModel(model);
M._free(dataPtr);
console.log("done.");
