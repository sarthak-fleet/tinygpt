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

// Find a Generate button. The playground has a sample/generate control.
const genCandidates = ["#generate", "#sampleBtn", "#genBtn", "button:has-text('Generate')", "button:has-text('Sample')"];
let clicked = false;
for (const sel of genCandidates) {
  const loc = page.locator(sel).first();
  if ((await loc.count()) > 0 && (await loc.isVisible().catch(() => false))) {
    await loc.click({ force: true }).catch(() => {});
    clicked = true;
    console.log("clicked", sel);
    break;
  }
}
if (!clicked) console.log("WARNING: no Generate button found — listing all visible buttons:");
if (!clicked) {
  const all = await page.locator("button").elementHandles();
  for (const h of all) {
    const t = (await h.textContent())?.trim().slice(0, 40);
    const vis = await h.isVisible();
    if (vis && t) console.log("   button:", t);
  }
}

// Wait ~10s for the generation to finish
await new Promise((r) => setTimeout(r, 10_000));

// Try multiple known output element ids
const outCandidates = ["#genOut", "#sample", "#sampleOut", "#generationText", "[data-role='generation']"];
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
