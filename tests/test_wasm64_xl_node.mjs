// test_wasm64_xl_node.mjs — does the Memory64 build run XL in Node?
//
// Calls the module the same way browser/src/backend.ts does: direct exports,
// pointer args wrapped in BigInt (because -sMEMORY64=1 -sWASM_BIGINT makes
// pointer-typed C args i64 → BigInt in JS). Mirrors the production browser
// path so a reproducer here is a real reproducer.

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

// On MEMORY64+WASM_BIGINT every pointer-typed arg is i64 → BigInt. Plain int
// args stay number. _malloc happens to return Number (the heap is below 2^53,
// so the compiler narrows it).
const toPtr = (x) => BigInt(x);

// XL: 8 layers, d=256, ctx=128, batch=8 — same shape the browser OOBs on.
const cfg = { vocab: 256, ctx: 128, layers: 8, heads: 8, dModel: 256, dMlp: 1024, batch: 8, seed: 42 };
console.log("creating model:", cfg);

const t0 = Date.now();
const handle = M._tg_model_create(
  cfg.vocab, cfg.ctx, cfg.layers, cfg.heads, cfg.dModel, cfg.dMlp, cfg.seed,
);
console.log(`  handle=${handle}  (typeof=${typeof handle})  alloc ${Date.now() - t0}ms`);
if (!handle) { console.log("FAIL: handle=0"); process.exit(2); }

const params = M._tg_model_num_params(handle);
console.log(`  params: ${(params / 1e6).toFixed(2)}M`);

const text = "the quick brown fox jumps over the lazy dog. ".repeat(400);
const bytes = new TextEncoder().encode(text);
const dataPtr = M._malloc(bytes.length);
console.log(`  dataPtr=${dataPtr}  (typeof=${typeof dataPtr})`);
M.HEAPU8.set(bytes, Number(dataPtr));
M._tg_set_data(handle, toPtr(dataPtr), bytes.length, 0.9);
console.log("setData OK");

console.log("\nrunning 5 training steps (batch=8, lr=3e-4) …");
for (let i = 0; i < 5; i++) {
  try {
    const t1 = Date.now();
    const loss = M._tg_train_step(handle, cfg.batch, 3e-4, 1.0);
    console.log(`  step ${i + 1}: loss=${loss.toFixed(4)}  ${Date.now() - t1}ms`);
  } catch (e) {
    console.log(`  step ${i + 1} FAIL:`, e?.message ?? e);
    console.log("→ The 64-bit module crashes in Node too.");
    console.log("→ Bug is in the 64-bit build itself, not browser-only pthread.");
    process.exit(1);
  }
}

console.log("\nPASS — tinygpt64.wasm trains XL in Node.");
console.log("The browser's OOB at d_model ≥ 256 is therefore browser-only");
console.log("(pthread + SharedArrayBuffer + Memory64 interaction).");
M._free(toPtr(dataPtr));
M._tg_model_free(handle);
