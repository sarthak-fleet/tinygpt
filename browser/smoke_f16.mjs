// smoke_f16.mjs — runtime smoke test for the f16-storage matmul path.
//
// Loads the gallery's Shakespeare model and verifies that:
//   1. The new train_f16.wgsl shader compiles (no WGSL errors in console).
//   2. The numerics gate runs and logs its verdict.
//   3. If the gate passes, the +f16 storage pill appears in the UI.
//   4. Generation runs to completion without WebGPU errors.
//
// Run with the dev server (preview) up on :5173:
//   node browser/smoke_f16.mjs

import { chromium } from "playwright";

const URL = process.env.SMOKE_URL ?? "http://localhost:5173/";

const browser = await chromium.launch({
  headless: false,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan"],
});
const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 } });
const page = await ctx.newPage();

const consoleLines = [];
page.on("console", (m) => {
  const t = m.type();
  const text = m.text();
  consoleLines.push({ t, text });
  if (t === "error") console.log("[browser err]", text);
  else if (text.includes("[ops]") || text.includes("f16") || text.includes("gate")) {
    console.log(`[browser ${t}]`, text);
  }
});
page.on("pageerror", (e) => console.log("[pageerror]", e.message));

console.log(`opening ${URL} …`);
await page.goto(URL, { waitUntil: "networkidle" });
await page.locator("#welcomeSkip").click({ timeout: 1500 }).catch(() => {});

// Wait for the demo banner to be visible — that's the signal that the
// gallery manifest fetched successfully and the page is ready.
await page.waitForSelector("#demoBanner:not([hidden])", { timeout: 10_000 });
console.log("page loaded; gallery banner visible");

// Open the gallery dialog and click the Shakespeare card.
await page.locator("#openGalleryBtn").click();
await page.waitForSelector("#galleryDialog[open]", { timeout: 5_000 });
console.log("gallery dialog opened");
await page.locator('.gallery-card[data-id="shakespeare"]').click();
console.log("clicked Shakespeare card — model loading…");

// Wait for the worker to post "ready to generate." status. This includes
// importState + prepareForInference + warmupGenerate. The status message
// contains the "f16-storage matmul active" suffix iff the gate passed.
const readyTimeoutMs = 60_000;
const t0 = Date.now();
let status = "";
while (Date.now() - t0 < readyTimeoutMs) {
  status = await page.evaluate(() => document.getElementById("status")?.textContent ?? "");
  if (/ready to generate/i.test(status)) break;
  await page.waitForTimeout(200);
}
console.log(`final status: "${status}"`);

const wasF16Active = /f16-storage matmul active/.test(status);
console.log(`f16-storage active per status: ${wasF16Active}`);

// Check for the +f16 storage pill.
const hasF16Pill = await page.evaluate(() => {
  return !!document.querySelector('#caps [data-explain="f16Storage"]');
});
console.log(`+f16 storage pill present: ${hasF16Pill}`);

// Look for the gate log line specifically.
const gateLine = consoleLines.find((l) => l.text.includes("[ops] f16-storage gate"));
console.log(`gate log: ${gateLine ? gateLine.text : "NOT FOUND"}`);

// Surface any WGSL errors that might have happened during shader compile.
const wgslErrors = consoleLines.filter((l) =>
  l.t === "error" && /wgsl|shader|pipeline/i.test(l.text),
);
console.log(`wgsl/shader errors: ${wgslErrors.length}`);
for (const e of wgslErrors) console.log("  -", e.text);

// Optional: drive a tiny Generate to confirm the f16 path can produce tokens.
console.log("\nclicking Generate to validate end-to-end …");
await page.locator("#prompt").fill("The model ");
await page.locator("#genTokens").fill("32");
await page.locator("#sample").click();
await page.waitForFunction(
  () => {
    const out = document.getElementById("output");
    if (!out || out.classList.contains("empty")) return false;
    return (out.textContent ?? "").length > 30;
  },
  null,
  { timeout: 30_000 },
);
const generated = await page.evaluate(() => document.getElementById("output")?.textContent ?? "");
console.log(`generated ${generated.length} chars: ${generated.slice(0, 120)}…`);

// Summary.
console.log("\n=== smoke summary ===");
console.log(`f16 gate ran:           ${gateLine ? "yes" : "NO"}`);
console.log(`f16 gate passed:        ${wasF16Active ? "yes" : "no"}`);
console.log(`+f16 storage pill:      ${hasF16Pill ? "yes" : "no"}`);
console.log(`wgsl errors:            ${wgslErrors.length}`);
console.log(`generation produced:    ${generated.length > 30 ? "ok" : "FAIL"}`);

const pass = gateLine && generated.length > 30 && wgslErrors.length === 0;
console.log(`\n${pass ? "✅ SMOKE PASS" : "❌ SMOKE FAIL"}`);

await browser.close();
process.exit(pass ? 0 : 1);
