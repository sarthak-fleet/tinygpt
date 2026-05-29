// copy_docs.mjs — pre-build step.
//
// The dynamic doc route at browser/src/pages/docs/[...slug].astro
// renders every markdown file under docs/. Astro's build needs that
// markdown inside src/ to be portable across build environments —
// Cloudflare Pages' Vite + Node may not honor `vite.fs.allow: [".."]`
// the same way local dev does. This script mirrors docs/**/*.md into
// browser/src/content/docs/ before `astro build` runs, so the import
// path is fully inside src/. The copy is gitignored.

import { promises as fs } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(here, "..", "..");
const SRC_DIR = resolve(REPO_ROOT, "docs");
const DST_DIR = resolve(here, "..", "src", "content", "docs");

/**
 * Recursively walk dir and yield absolute paths to *.md files.
 */
async function* walkMarkdown(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = resolve(dir, entry.name);
    if (entry.isDirectory()) {
      yield* walkMarkdown(full);
    } else if (entry.isFile() && entry.name.endsWith(".md")) {
      yield full;
    }
  }
}

// Wipe destination — we want copy_docs to be the single source of truth.
// Otherwise stale files (e.g. renamed/removed docs) linger between builds.
try {
  await fs.rm(DST_DIR, { recursive: true, force: true });
} catch (err) {
  // Ignore — directory may not exist on first run.
}
await fs.mkdir(DST_DIR, { recursive: true });

let copied = 0;
for await (const src of walkMarkdown(SRC_DIR)) {
  const rel = relative(SRC_DIR, src);
  const dst = resolve(DST_DIR, rel);
  await fs.mkdir(dirname(dst), { recursive: true });
  try {
    await fs.copyFile(src, dst);
    copied++;
  } catch (err) {
    console.error(`copy_docs.mjs: failed to copy ${rel}: ${err.message}`);
    process.exit(1);
  }
}
console.log(`copy_docs.mjs: copied ${copied} markdown files → src/content/docs/`);
