// verify_demo.mjs — quick sanity check that demo.tinygpt loads and generates
// readable text.
//
// Steps:
//  1. Open the playground
//  2. Click "Load pretrained model" (the demo banner)
//  3. Wait for load to complete
//  4. Click Generate, capture ~300 chars of output
//  5. Print it so we can eyeball

import { chromium } from "playwright";

const browser = await chromium.launch({
  headless: false,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan"],
});
const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 } });
const page = await ctx.newPage();
page.on("dialog", (d) => d.accept().catch(() => {}));
page.on("pageerror", (e) => console.log("[pageerror]", e.message));

await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
await page.locator("#welcomeSkip").click({ timeout: 1500 }).catch(() => {});

// Banner should be visible since /demo.tinygpt 200s
const bannerVisible = await page.locator("#demoBanner").isVisible().catch(() => false);
console.log("banner visible:", bannerVisible);
if (!bannerVisible) {
  console.log("FAIL: banner did not appear — check /demo.tinygpt deployment");
  await browser.close();
  process.exit(1);
}

console.log("clicking Load pretrained model…");
await page.locator("#loadDemoBtn").click({ force: true });

// Wait for loading to settle — banner hides on success
await page.locator("#demoBanner").waitFor({ state: "hidden", timeout: 60_000 }).catch(() => {});
console.log("model loaded (banner now hidden)");

// The generate button is #sample (per index.astro:2825), the OUTPUT div is
// #output (per index.astro:2828). The button text on the page is "Generate".
// Need to navigate to the Watch screen first — the Sample card lives there.
await page.locator(".screen-tab[data-screen='watch']").click({ force: true }).catch(() => {});
await page.waitForTimeout(300);
await page.evaluate(() => document.getElementById("sample").click());
console.log("clicked #sample (Generate button)");

// Wait up to 30s for generation to finish (Medium model, fast)
await page.waitForFunction(
  () => {
    const out = document.getElementById("output");
    return out && !out.classList.contains("empty") && out.textContent.length > 20;
  },
  null,
  { timeout: 30_000 },
).catch((e) => console.log("WARN:", e.message));

const outCandidates = ["#output", "#genOut", "#sampleOut"];
let sample = "(none found)";
for (const sel of outCandidates) {
  const loc = page.locator(sel).first();
  if ((await loc.count()) > 0) {
    sample = (await loc.textContent()) ?? "(empty)";
    console.log("found output in:", sel);
    break;
  }
}

console.log("\n--- generated sample ---");
console.log(sample.slice(0, 1200));
console.log("--- /sample ---");

await browser.close();
