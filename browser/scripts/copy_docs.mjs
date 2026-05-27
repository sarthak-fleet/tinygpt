// copy_docs.mjs — pre-build step.
//
// The doc pages under browser/src/pages/docs/*.astro render markdown that
// lives in the repo root at docs/*.md (so GitHub renders the same source).
// Astro's build needs that markdown inside src/ to be portable across build
// environments — Cloudflare Pages' Vite + Node may not honor
// `vite.fs.allow: [".."]` the same way local dev does. This script
// copies the docs into browser/src/content/docs/ before `astro build` runs,
// so the import path is fully inside src/. The copy is gitignored.

import { promises as fs } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(here, "..", "..");
const SRC_DIR = resolve(REPO_ROOT, "docs");
const DST_DIR = resolve(here, "..", "src", "content", "docs");

// Exact set of docs the playground renders. Adding new web-rendered docs
// means: add the file here AND a matching wrapper at
// browser/src/pages/docs/<slug>.astro.
const DOCS = [
  "lessons.md",
  "session_retrospective.md",
  "qa_log.md",
  "decision_log.md",
  "study_guide.md",
  "annotated_transcript.md",
];

await fs.mkdir(DST_DIR, { recursive: true });
let copied = 0;
for (const name of DOCS) {
  const src = resolve(SRC_DIR, name);
  const dst = resolve(DST_DIR, name);
  try {
    await fs.copyFile(src, dst);
    copied++;
  } catch (err) {
    console.error(`copy_docs.mjs: failed to copy ${name}: ${err.message}`);
    process.exit(1);
  }
}
console.log(`copy_docs.mjs: copied ${copied}/${DOCS.length} docs → src/content/docs/`);
