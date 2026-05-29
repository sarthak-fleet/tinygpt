// smoke_autosave.mjs — verify the `?autoSave=NAME` URL param fires the
// model download automatically when training completes. Tiny preset +
// 30 steps so the whole run finishes in <30 seconds. Compares the
// downloaded `.tinygpt` size against a sanity bound.
//
// Run: node browser/smoke_autosave.mjs
// Expects the dev server already running on :5173.

import { chromium } from "playwright";
import { promises as fs } from "node:fs";

const APP_URL = "http://localhost:5173/?autoSave=smoke-autosave";
const OUT = "/tmp/smoke-autosave.tinygpt";

const browser = await chromium.launch({
  headless: false,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan"],
});
const ctx = await browser.newContext({
  viewport: { width: 1200, height: 800 },
  acceptDownloads: true,
});
const page = await ctx.newPage();
page.on("dialog", (d) => d.accept().catch(() => {}));
page.on("pageerror", (e) => console.log("[pageerror]", e.message));

await page.goto(APP_URL, { waitUntil: "networkidle" });
await page.locator("#welcomeSkip").click({ timeout: 1500 }).catch(() => {});

// Tiny preset, 30 steps. Default corpus is fine (the existing textarea
// has the bundled sample loaded), but ensure something's there.
await page.evaluate(() => {
  const setVal = (id, v) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.value = String(v);
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  };
  document.getElementById("sizePreset").value = "small";
  document.getElementById("sizePreset").dispatchEvent(new Event("change", { bubbles: true }));
  setVal("maxSteps", 30);
  const back = document.getElementById("backend");
  back.value = "webgpu";
  back.dataset.userPicked = "1";
  back.dispatchEvent(new Event("change", { bubbles: true }));
});

// Register the download listener BEFORE clicking start (same pattern as
// train_gallery_one.mjs). 2-minute timeout — generous for a 30-step run.
const downloadPromise = page.waitForEvent("download", { timeout: 120_000 });
downloadPromise.catch(() => {});

const t0 = Date.now();
await page.locator("#start").click({ force: true });
console.log("[smoke] training started");

const download = await downloadPromise;
const tmp = await download.path();
await fs.copyFile(tmp, OUT);
const stat = await fs.stat(OUT);
const wallSec = ((Date.now() - t0) / 1000).toFixed(1);

await browser.close();

// Sanity: a Small-preset .tinygpt should be a few MB (state buffer is
// 3 × params × 4 bytes — w/m/v fp32 triplets plus a 4-byte step counter).
const bytes = stat.size;
const sizeMB = (bytes / 1024 / 1024).toFixed(2);
if (bytes < 1_000_000) {
  console.error(`[smoke] FAIL: ${OUT} is suspiciously small (${bytes} bytes)`);
  process.exit(1);
}
console.log(`[smoke] PASS: ${OUT} = ${sizeMB} MB in ${wallSec}s`);
console.log(`[smoke] auto-save URL param works end-to-end ✓`);
