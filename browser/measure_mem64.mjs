// measure_mem64.mjs — actually click Start on the Small preset with the
// Memory64 module loaded, and capture real measured numbers from the
// in-browser training loop. Conservative: Small preset only (~360k params),
// 50 steps, then Stop. Should complete in well under a minute on M-series.

import { chromium } from "playwright";

const browser = await chromium.launch();
const ctx = await browser.newContext({
  viewport: { width: 1400, height: 900 },
  deviceScaleFactor: 2,
});
const page = await ctx.newPage();

const consoleErrors = [];
const consoleLines = [];
page.on("pageerror", (e) => consoleErrors.push(`pageerror: ${e.message}`));
page.on("console", (msg) => {
  consoleLines.push(`[${msg.type()}] ${msg.text()}`);
  if (msg.type() === "error") consoleErrors.push(`console.error: ${msg.text()}`);
});

console.log("loading homepage...");
await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
await page.locator("#welcomeSkip").click({ timeout: 1500 }).catch(() => {});
await page.waitForTimeout(300);

// Confirm Memory64 is on. If not, abort — the whole point is to measure it.
const mem64On = await page.evaluate(() => {
  const el = document.querySelector('[data-explain="memory64"]');
  return el && el.classList.contains("on");
});
console.log(`Memory64 pill on: ${mem64On}`);
if (!mem64On) {
  console.log("Memory64 not available in this browser — aborting.");
  await browser.close();
  process.exit(1);
}

// Set the preset to Small.
await page.locator("#sizePreset").selectOption({ value: "small" });
await page.waitForTimeout(150);

// Set max steps to 50 — keep the run short.
await page.evaluate(() => {
  const maxSteps = document.getElementById("maxSteps");
  maxSteps.value = "50";
  maxSteps.dispatchEvent(new Event("input", { bubbles: true }));
});

// Confirm backend is wasm (don't kick the GPU). #backend lives inside the
// Advanced collapsible — set the value via evaluate() so we don't have to
// crack the panel open.
await page.evaluate(() => {
  const el = document.getElementById("backend");
  el.value = "wasm";
  el.dataset.userPicked = "1";
  el.dispatchEvent(new Event("change", { bubbles: true }));
});

// Suppress any preflight confirm dialog.
page.on("dialog", (d) => d.accept());

const config = await page.evaluate(() => {
  const get = (id) => document.getElementById(id)?.value;
  return {
    layers: get("layers"), dModel: get("dModel"), ctx: get("ctx"),
    batch: get("batch"), maxSteps: get("maxSteps"), backend: get("backend"),
  };
});
console.log(`config: ${JSON.stringify(config)}`);

const t0 = Date.now();
console.log(`[t=0] clicking Start training...`);
await page.locator("#start").click();

// Watch for the run to finish — status becomes "training complete" or similar,
// or the loss-curve final step reaches maxSteps.
const finalStep = await page.waitForFunction(() => {
  // Look at the stat panel
  const stepEl = document.getElementById("stStep");
  if (!stepEl) return null;
  const m = (stepEl.textContent || "").match(/(\d+)\s*\/\s*(\d+)/);
  if (!m) return null;
  if (parseInt(m[1]) >= parseInt(m[2])) return m[1];
  return null;
}, null, { timeout: 120_000, polling: 500 }).catch((e) => {
  console.log(`waitForFunction error: ${e.message}`);
  return null;
});

const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
console.log(`[t=${elapsed}s] training reached step ${await finalStep?.jsonValue() ?? "?"}`);

await page.waitForTimeout(800);

// Grab the final stats.
const stats = await page.evaluate(() => {
  const get = (id) => document.getElementById(id)?.textContent?.trim();
  return {
    step: get("stStep"),
    loss: get("stLoss"),
    tracker: get("stTracker"),
    elapsed: get("stElapsed"),
    backend: get("stBackend"),
  };
});
console.log("final stats:", JSON.stringify(stats));

// Switch to Watch + take screenshot.
await page.locator('.screen-tab[data-screen="watch"]').click().catch(() => {});
await page.waitForTimeout(400);
await page.screenshot({ path: "/tmp/measured-watch.png", fullPage: false });
console.log("-> /tmp/measured-watch.png");

if (consoleErrors.length) {
  console.log(`\n${consoleErrors.length} console errors:`);
  consoleErrors.forEach((e) => console.log("  " + e));
} else {
  console.log("\nzero console errors");
}

await browser.close();
