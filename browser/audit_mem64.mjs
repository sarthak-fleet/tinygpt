// audit_mem64.mjs — verify the Memory64 + Behemoth preset wiring in the
// browser without ever clicking Start (per "don't wreck my GPU/CPU").
//
// Checks:
//   1. Page loads with no console errors
//   2. The "Memory64" capability pill is rendered
//   3. The size-preset dropdown contains "Behemoth"
//   4. Selecting Behemoth populates the expected hyperparams
//   5. Screenshots taken at each step for visual inspection.

import { chromium } from "playwright";

const browser = await chromium.launch();
const ctx = await browser.newContext({
  viewport: { width: 1400, height: 900 },
  deviceScaleFactor: 2,
});
const page = await ctx.newPage();

const consoleErrors = [];
page.on("pageerror", (e) => consoleErrors.push(`pageerror: ${e.message}`));
page.on("console", (msg) => {
  if (msg.type() === "error") consoleErrors.push(`console.error: ${msg.text()}`);
});

console.log("loading homepage...");
await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });

// Skip welcome modal if present.
await page.locator("#welcomeSkip").click({ timeout: 1500 }).catch(() => {});
await page.waitForTimeout(400);

await page.screenshot({ path: "/tmp/mem64-1-loaded.png", fullPage: false });
console.log("  -> /tmp/mem64-1-loaded.png");

// --- 2. Memory64 pill rendered? -----------------------------------------
const mem64Pill = page.locator('[data-explain="memory64"]');
const mem64Count = await mem64Pill.count();
console.log(`Memory64 pill count: ${mem64Count}`);
if (mem64Count > 0) {
  const text = (await mem64Pill.innerText()).trim();
  const onClass = await mem64Pill.evaluate((el) => el.classList.contains("on"));
  console.log(`  text="${text}" on=${onClass}`);
}

// Capability bar screenshot.
const caps = page.locator("#caps");
if (await caps.count()) {
  await caps.screenshot({ path: "/tmp/mem64-2-caps.png" });
  console.log("  -> /tmp/mem64-2-caps.png");
}

// --- 3. Behemoth in the preset dropdown? --------------------------------
const presetSelect = page.locator("#sizePreset");
const presetOptions = await presetSelect.locator("option").allTextContents();
console.log(`preset options (${presetOptions.length}):`);
presetOptions.forEach((o) => console.log(`  - ${o}`));
const hasBehemoth = presetOptions.some((o) => o.toLowerCase().includes("behemoth"));
console.log(`hasBehemoth: ${hasBehemoth}`);

// --- 4. Select Behemoth and read hyperparams ---------------------------
if (hasBehemoth) {
  await presetSelect.selectOption({ value: "behemoth" });
  await page.waitForTimeout(300);
  // Expand the hyperparam details if collapsed.
  await page.locator(".hyperparam-details").click().catch(() => {});
  await page.waitForTimeout(150);
  const config = await page.evaluate(() => {
    const get = (id) => /** @type {HTMLInputElement} */ (document.getElementById(id))?.value;
    return {
      layers: get("layers"),
      dModel: get("dModel"),
      ctx: get("ctx"),
      batch: get("batch"),
      maxSteps: get("maxSteps"),
      heads: get("heads"),
      dMlp: get("dMlp"),
    };
  });
  console.log("Behemoth config in form:", JSON.stringify(config));
  await page.screenshot({ path: "/tmp/mem64-3-behemoth-selected.png", fullPage: false });
  console.log("  -> /tmp/mem64-3-behemoth-selected.png");
}

// --- 5. Load the pre-trained demo to exercise the WASM module in the
// worker (TinyGptBackend.load() is lazy, so without this the audit only
// verifies the probe, not that the actual .wasm bytes load + the model
// rehydrates under the Memory64 ABI).
console.log("\nloading pre-trained demo to exercise worker WASM load...");
const demoBtn = page.locator("#loadDemoBtn");
if (await demoBtn.count()) {
  await demoBtn.click();
  // Wait for the worker to report "restored" — main.ts enables the Watch
  // tab + populates sample on that event.
  await page.waitForFunction(() => {
    const el = document.getElementById("sample");
    return el && !el.disabled;
  }, { timeout: 30_000 }).catch((e) => console.log(`  (sample never enabled: ${e.message})`));
  const sampleEnabled = await page.evaluate(() => {
    const el = /** @type {HTMLButtonElement} */ (document.getElementById("sample"));
    return el ? !el.disabled : null;
  });
  console.log(`  sample button enabled after demo load: ${sampleEnabled}`);
  // Switch to Watch.
  await page.locator('[data-screen="watch"]').click().catch(() => {});
  await page.waitForTimeout(400);
  await page.screenshot({ path: "/tmp/mem64-4-watch-after-demo.png", fullPage: false });
  console.log("  -> /tmp/mem64-4-watch-after-demo.png");
}

if (consoleErrors.length) {
  console.log(`\n${consoleErrors.length} console errors:`);
  consoleErrors.forEach((e) => console.log("  " + e));
} else {
  console.log("\nno console errors");
}

await browser.close();
