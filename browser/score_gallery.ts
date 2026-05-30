// score_gallery.ts — DEPRECATED IN FAVOR OF MAC-SIDE SCORING.
//
// History: this script used to load the WASM module, parse each
// `.tinygpt` in `data/gallery/`, recreate a vocab=256 byte-level model
// via `tg_model_create`, import the saved state, and run `tg_eval` to
// produce a TinyStories perplexity score. That worked fine for the five
// browser-trained gallery cards (all byte-level, ~9.6M params), but
// silently broke the moment we wanted to score a BPE-trained
// checkpoint: the script hardcoded `vocab=256` while the BPE model's
// embedding table had vocab=49152, so `tg_import_state` would either
// crash or produce garbage outputs and meaningless perplexity numbers.
//
// Replacement: `tinygpt score-bench` (Mac-side). The native binary
// loads the model through `ModelLoader` (which already auto-detects
// BPE-vs-byte from the header), runs the benchmark via the same
// `model.loss` path used by `tinygpt eval`, and writes the result back
// into `browser/public/gallery/manifest.json` — exactly the same shape
// the leaderboard reads. See docs/bpe_browser_scoring.md.
//
// Why no in-browser tokenizer dependency: HuggingFace tokenizers are
// big (~6 MB compressed for a SmolLM-class BPE), need a wasm runtime
// of their own, and would tie the leaderboard build to a tokenizer-fs
// asset pipeline. The browser playground already had to make this
// tradeoff for the BPE-loaded path; the leaderboard scorer skips it
// entirely by pre-scoring on the Mac.
//
// This file is kept as a thin shim that re-runs the legacy byte-level
// path for the original five gallery models so existing CI invocations
// (`node browser/score_gallery.ts`) still produce identical scores for
// those entries. Any NEW model goes through the Mac path.
//
// Usage:
//   tinygpt score-bench <model.tinygpt> --benchmarks bench/benchmarks.json
//   node browser/score_gallery.ts   # legacy byte-only refresh
//
// The legacy refresh below ignores any `.tinygpt` files that don't
// declare `weightDtype: "fp32"` and `vocab` ≤ 256 — anything BPE just
// gets a warning, not a corrupted score.

import { promises as fs } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import type { GalleryManifest, GalleryModel } from "./src/gallery-schema.ts";

type WasmHandle = number;

interface TinyGPTModule {
  cwrap: (name: string, ret: string | null, args: (string | null)[]) => (...a: number[]) => number;
  _malloc: (n: number) => number;
  _free: (ptr: number) => void;
  HEAPU8: Uint8Array;
}

const here = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(here, "..");
const WASM_JS = resolve(ROOT, "browser/public/tinygpt.js");
const GALLERY_DIR = resolve(ROOT, "data/gallery");
const MANIFEST_PATH = resolve(ROOT, "browser/public/gallery/manifest.json");
const HOLDOUT_PATH = resolve(ROOT, "browser/public/benchmarks/tinystories-eval.json");

console.log("[score] DEPRECATED — see docs/bpe_browser_scoring.md.");
console.log("[score] running legacy byte-only refresh for the original gallery cards…");

// We only try to load the WASM module if there's actually a byte-level
// model directory to score — the BPE path doesn't need it at all.
let candidates: string[] = [];
try {
  candidates = (await fs.readdir(GALLERY_DIR)).filter((f) => f.endsWith(".tinygpt"));
} catch (e) {
  console.log(`[score] no gallery dir (${GALLERY_DIR}); nothing to refresh.`);
  process.exit(0);
}
if (candidates.length === 0) {
  console.log("[score] no .tinygpt files in gallery; nothing to refresh.");
  process.exit(0);
}

console.log("[score] loading WASM module…");
const { default: createTinyGPT } = await import(WASM_JS);
const M: TinyGPTModule = await createTinyGPT();

const N = "number";
const tgModelCreate = M.cwrap("tg_model_create", N, [N, N, N, N, N, N, N]);
const tgModelNumParams = M.cwrap("tg_model_num_params", N, [N]);
const tgSetData = M.cwrap("tg_set_data", null, [N, N, N, N]);
const tgEval = M.cwrap("tg_eval", N, [N, N, N, N]);
const tgModelFree = M.cwrap("tg_model_free", null, [N]);
const tgImportState = M.cwrap("tg_import_state", null, [N, N]);

interface HoldoutFile {
  source: string;
  count: number;
  totalBytes: number;
  stories: string[];
}
console.log("[score] reading TinyStories holdout…");
const holdout: HoldoutFile = JSON.parse(await fs.readFile(HOLDOUT_PATH, "utf8"));
const holdoutBytes = new TextEncoder().encode(holdout.stories.join("\n\n"));
console.log(`[score] holdout: ${holdout.stories.length} stories, ${holdoutBytes.length} bytes`);

const manifest: GalleryManifest = JSON.parse(await fs.readFile(MANIFEST_PATH, "utf8"));

const MAGIC = "TGPT";

interface TinygptHeaderConfig {
  layers?: number;
  dModel?: number;
  ctx?: number;
  heads?: number;
  dMlp?: number;
  vocabSize?: number;       // present in BPE headers; absent ⇒ byte-level (256)
  tokenizerSource?: string; // present in BPE headers
}
interface TinygptHeader {
  config: TinygptHeaderConfig;
  weightDtype?: string;
}
interface ParsedTinygpt {
  config: TinygptHeaderConfig;
  stateBytes: Uint8Array;
  isBpe: boolean;
}

function parseTinygpt(buf: ArrayBuffer): ParsedTinygpt {
  if (buf.byteLength < 12) throw new Error("file too small");
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 4));
  if (magic !== MAGIC) throw new Error(`bad magic: ${magic}`);
  const dv = new DataView(buf);
  const version = dv.getUint32(4, true);
  if (version !== 1 && version !== 2) {
    throw new Error(`unsupported version ${version}`);
  }
  const headerLen = dv.getUint32(8, true);
  if (12 + headerLen > buf.byteLength) throw new Error("malformed header");
  const headerJson = new TextDecoder().decode(new Uint8Array(buf, 12, headerLen));
  const header: TinygptHeader = JSON.parse(headerJson);
  const stateBytes = new Uint8Array(buf.slice(12 + headerLen));
  if (header.weightDtype && header.weightDtype !== "fp32") {
    throw new Error(
      `score_gallery legacy path only handles fp32 .tinygpt files; ` +
      `this one has weightDtype=${header.weightDtype}.`,
    );
  }
  const isBpe = (header.config.vocabSize !== undefined && header.config.vocabSize > 256)
             || (header.config.tokenizerSource !== undefined);
  return { config: header.config, stateBytes, isBpe };
}

function evalPerplexity(handle: WasmHandle, batchSize: number, nBatches: number) {
  const dataPtr = M._malloc(holdoutBytes.length);
  M.HEAPU8.set(holdoutBytes, dataPtr);
  tgSetData(handle, dataPtr, holdoutBytes.length, 0.0);
  M._free(dataPtr);
  const meanLoss = tgEval(handle, 1, batchSize, nBatches);
  return { loss: meanLoss, perplexity: Math.exp(meanLoss) };
}

const benchId = "tinystories-ppl";
interface ScoreUpdate {
  id: string;
  score: number | null;
  details: { error?: string; skipped?: string; loss?: number; vocab?: number; batches?: number; tokens?: number };
}
const updated: ScoreUpdate[] = [];

for (const filename of candidates.sort()) {
  const id = filename.replace(/\.tinygpt$/, "");
  const path = resolve(GALLERY_DIR, filename);
  console.log(`\n[score] === ${id} ===`);
  try {
    const buf = (await fs.readFile(path)).buffer as ArrayBuffer;
    const parsed = parseTinygpt(buf);
    if (parsed.isBpe) {
      console.log(`[score] skipping ${id}: BPE model (vocab=${parsed.config.vocabSize}). ` +
                  `Run \`tinygpt score-bench ${path} --benchmarks bench/benchmarks.json\` ` +
                  `from the worktree root to score it natively.`);
      updated.push({
        id, score: null,
        details: { skipped: "BPE model — use Mac-side `tinygpt score-bench`" },
      });
      continue;
    }
    const { config, stateBytes } = parsed;
    console.log(`        config: ${config.layers}L · d=${config.dModel} · ctx=${config.ctx} · ${stateBytes.length} state bytes`);
    const handle: WasmHandle = tgModelCreate(
      256,
      config.ctx ?? 256,
      config.layers ?? 12,
      config.heads ?? 8,
      config.dModel ?? 256,
      config.dMlp ?? (config.dModel ? config.dModel * 4 : 1024),
      42,
    );
    if (handle === 0) throw new Error("tg_model_create returned 0");
    const nParams = tgModelNumParams(handle);
    console.log(`        params: ${nParams.toLocaleString()}`);
    const statePtr = M._malloc(stateBytes.length);
    M.HEAPU8.set(stateBytes, statePtr);
    tgImportState(handle, statePtr);
    M._free(statePtr);
    const t0 = Date.now();
    const { loss, perplexity } = evalPerplexity(handle, 8, 32);
    const dt = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(`        eval: loss=${loss.toFixed(3)}  perplexity=${perplexity.toFixed(2)}  (${dt}s)`);
    tgModelFree(handle);
    updated.push({
      id, score: perplexity,
      details: { loss, vocab: 256, batches: 32, tokens: 32 * 8 * (config.ctx ?? 256) },
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error(`[score] FAIL ${id}: ${msg}`);
    updated.push({ id, score: null, details: { error: msg } });
  }
}

// Merge byte-only scores into the manifest. We do NOT touch BPE entries
// here (those come through `tinygpt score-bench`); if `id` already has
// a score from the Mac path we leave it alone unless we have a new
// number to write.
console.log("\n[score] merging legacy byte-only scores into manifest…");
const byId = new Map<string, GalleryModel>((manifest.models || []).map((m) => [m.id, m]));
for (const { id, score } of updated) {
  if (score == null) continue;          // skipped or failed → no write
  let entry = byId.get(id);
  if (!entry) {
    entry = {
      id, name: id, file: `${id}.bin`,
      benchmarks: {},
      submission: { author: "TinyGPT", submittedAt: new Date().toISOString(), featured: false },
    };
    manifest.models = manifest.models || [];
    manifest.models.push(entry);
  }
  entry.benchmarks = entry.benchmarks || {};
  entry.benchmarks[benchId] = score;
}

await fs.writeFile(MANIFEST_PATH, JSON.stringify(manifest, null, 2));
console.log(`[score] wrote ${MANIFEST_PATH}\n`);

console.log("[score] summary:");
console.log("   id                   score (ppl)   loss");
console.log("   ───────────────────  ───────────   ─────");
for (const { id, score, details } of updated) {
  if (score == null) {
    const reason = details.skipped ?? details.error ?? "—";
    console.log(`   ${id.padEnd(20)} —             ${reason}`);
  } else {
    console.log(`   ${id.padEnd(20)} ${score.toFixed(2).padStart(10)}    ${details.loss?.toFixed(3) ?? "—"}`);
  }
}
