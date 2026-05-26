// train_demo.mjs — train a demo model and save it as browser/public/demo.tinygpt.
//
// The playground's first-time visitors will be able to click "Try a trained
// model" and immediately see coherent (well, shakespeare-shaped) output
// instead of garbled letters. This script runs end-to-end via Node + the
// compiled WASM module — no browser needed.
//
//   bash wasm/build_wasm.sh
//   node tests/train_demo.mjs
//
// Outputs: browser/public/demo.tinygpt (v2 file format, same as the browser
// download button produces).

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(here, "..");
const TINYGPT_JS = path.join(ROOT, "browser", "public", "tinygpt.js");
const OUT = path.join(ROOT, "browser", "public", "demo.tinygpt");

// --- corpus: Tiny Shakespeare from Karpathy's char-rnn ---------------------
// CORS-friendly raw URL. ~1 MB of plays.
const CORPUS_URL =
  "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

async function fetchCorpus() {
  console.log(`fetching ${CORPUS_URL}…`);
  const resp = await fetch(CORPUS_URL);
  if (!resp.ok) throw new Error(`corpus fetch failed: HTTP ${resp.status}`);
  const text = await resp.text();
  console.log(`  ${(text.length / 1024).toFixed(0)} KB`);
  return text;
}

// --- training config — Small preset, threaded WASM -------------------------
const CFG = {
  ctx: 96,
  layers: 4,
  heads: 4,
  dModel: 128,
  dMlp: 512,
  batchSize: 16,
  learningRate: 0.003,
  gradClip: 1.0,
  maxSteps: 3000,
  evalEvery: 100,
  backend: "wasm",
  seed: 42,
};

// --- file format: v2 .tinygpt header --------------------------------------
const MAGIC = "TGPT";
const VERSION = 2;

function buildManifest(cfg) {
  const { layers: L, dModel: C, ctx } = cfg;
  const dMlp = cfg.dMlp ?? C * 4;
  const V = 256;
  const ents = [];
  let off = 0;
  const push = (name, shape) => {
    ents.push({ name, shape, floatOffset: off });
    off += shape.reduce((a, b) => a * b, 1);
  };
  push("token_embedding.weight", [V, C]);
  push("position_embedding.weight", [ctx, C]);
  push("ln_final.weight", [C]);
  push("ln_final.bias", [C]);
  for (let i = 0; i < L; i++) {
    push(`blocks.${i}.ln1.weight`, [C]);
    push(`blocks.${i}.ln1.bias`, [C]);
    push(`blocks.${i}.attn.q_proj.weight`, [C, C]);
    push(`blocks.${i}.attn.q_proj.bias`, [C]);
    push(`blocks.${i}.attn.k_proj.weight`, [C, C]);
    push(`blocks.${i}.attn.k_proj.bias`, [C]);
    push(`blocks.${i}.attn.v_proj.weight`, [C, C]);
    push(`blocks.${i}.attn.v_proj.bias`, [C]);
    push(`blocks.${i}.attn.o_proj.weight`, [C, C]);
    push(`blocks.${i}.attn.o_proj.bias`, [C]);
    push(`blocks.${i}.ln2.weight`, [C]);
    push(`blocks.${i}.ln2.bias`, [C]);
    push(`blocks.${i}.mlp.fc_in.weight`, [dMlp, C]);
    push(`blocks.${i}.mlp.fc_in.bias`, [dMlp]);
    push(`blocks.${i}.mlp.fc_out.weight`, [C, dMlp]);
    push(`blocks.${i}.mlp.fc_out.bias`, [C]);
  }
  return ents;
}

function encodeFile(cfg, stateBuffer, history, sampleText) {
  const final = history[history.length - 1];
  const bestVal = history.reduce(
    (best, h) =>
      h.val != null && (!best || h.val < best.loss)
        ? { loss: h.val, step: h.step }
        : best,
    null,
  );
  const header = {
    version: VERSION,
    savedAt: new Date().toISOString(),
    config: cfg,
    manifest: buildManifest(cfg),
    includesOptimizerState: true,
    stateByteLength: stateBuffer.byteLength,
    lossHistory: history.slice(-512).map((p) => ({
      step: p.step,
      train: +p.train.toFixed(4),
      val: p.val != null ? +p.val.toFixed(4) : null,
    })),
    finalLoss: final
      ? { step: final.step, train: final.train, val: final.val ?? null }
      : null,
    sample: sampleText.slice(0, 320),
    bestVal,
    project: "https://github.com/sarthakagrawal927/tinygpt",
  };
  const headerJson = Buffer.from(JSON.stringify(header), "utf8");
  const prefix = Buffer.alloc(12);
  prefix.write(MAGIC, 0, 4, "ascii");
  prefix.writeUInt32LE(VERSION, 4);
  prefix.writeUInt32LE(headerJson.byteLength, 8);
  return Buffer.concat([prefix, headerJson, Buffer.from(stateBuffer)]);
}

// --- main ------------------------------------------------------------------
async function main() {
  const text = await fetchCorpus();
  const bytes = new TextEncoder().encode(text);

  console.log(`loading WASM module from ${TINYGPT_JS}…`);
  const { default: createTinyGPT } = await import(TINYGPT_JS);
  const M = await createTinyGPT();

  const N = "number";
  const create = M.cwrap("tg_model_create", N, [N, N, N, N, N, N, N]);
  const setData = M.cwrap("tg_set_data", null, [N, N, N, N]);
  const trainStep = M.cwrap("tg_train_step", N, [N, N, N, N]);
  const evalLoss = M.cwrap("tg_eval", N, [N, N, N, N]);
  const generate = M.cwrap("tg_generate", N, [N, N, N, N, N, N, N, N]);
  const numParams = M.cwrap("tg_model_num_params", N, [N]);
  const stateBytes = M.cwrap("tg_state_bytes", N, [N]);
  const exportState = M.cwrap("tg_export_state", null, [N, N]);

  // Allocate the corpus in WASM heap.
  const dataPtr = M._malloc(bytes.length);
  M.HEAPU8.set(bytes, dataPtr);

  // Build the model.
  const model = create(
    256, CFG.ctx, CFG.layers, CFG.heads, CFG.dModel, CFG.dMlp, CFG.seed,
  );
  setData(model, dataPtr, bytes.length, 0.9);
  console.log(`model: ${(numParams(model) / 1e6).toFixed(2)}M params`);
  console.log(`training ${CFG.maxSteps} steps on ${(bytes.length / 1024).toFixed(0)} KB…`);

  const history = [];
  const t0 = performance.now();
  let chunk = 8;
  for (let step = 0; step < CFG.maxSteps; ) {
    let trainLoss = 0;
    const cap = Math.min(step + chunk, CFG.maxSteps);
    while (step < cap) {
      trainLoss = trainStep(model, CFG.batchSize, CFG.learningRate, CFG.gradClip);
      step++;
    }
    if (step % CFG.evalEvery === 0 || step >= CFG.maxSteps) {
      const valLoss = evalLoss(model, 1, CFG.batchSize, 5);
      const elapsed = (performance.now() - t0) / 1000;
      history.push({ step, train: trainLoss, val: valLoss });
      console.log(
        `  step ${step.toString().padStart(4)}  train ${trainLoss.toFixed(3)}  val ${valLoss.toFixed(3)}  (${elapsed.toFixed(0)}s elapsed)`,
      );
    }
  }
  const totalSec = (performance.now() - t0) / 1000;
  console.log(`\ndone in ${(totalSec / 60).toFixed(1)} min.`);

  // Generate a sample to embed in the header.
  const prompt = "ROMEO:";
  const promptBytes = new TextEncoder().encode(prompt);
  const promptPtr = M._malloc(promptBytes.length);
  M.HEAPU8.set(promptBytes, promptPtr);
  const outLen = 200;
  const outPtr = M._malloc(outLen);
  const seed = (Date.now() & 0xffff) >>> 0;
  const produced = generate(
    model, promptPtr, promptBytes.length, outPtr, outLen, 0.8, 40, seed,
  );
  const sampleBytes = new Uint8Array(M.HEAPU8.buffer, outPtr, produced).slice();
  const sample = prompt + new TextDecoder().decode(sampleBytes);
  console.log(`\nsample (200 tokens, temp 0.8):`);
  console.log("  " + sample.replace(/\n/g, "\n  "));

  // Export model state.
  const sb = stateBytes(model);
  const statePtr = M._malloc(sb);
  exportState(model, statePtr);
  const stateBuffer = M.HEAPU8.slice(statePtr, statePtr + sb).buffer;

  // Build the .tinygpt file and write.
  const file = encodeFile(CFG, stateBuffer, history, sample);
  await fs.writeFile(OUT, file);
  console.log(`\nwrote ${OUT}  (${(file.length / 1024).toFixed(0)} KB)`);
}

main().catch((err) => {
  console.error("error:", err);
  process.exit(1);
});
