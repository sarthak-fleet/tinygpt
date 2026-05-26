// Focused screenshots of small UI elements for nitpick-level review.
import { chromium } from "playwright";

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: 2 });
const page = await ctx.newPage();
await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
await page.locator("#welcomeSkip").click().catch(() => {});
await page.waitForTimeout(300);

const targets = [
  ["screen-nav",       ".screen-nav"],
  ["demo-banner",      ".demo-banner"],
  ["train-title",      ".hero-config .card-title.hero-title"],
  ["corpus-head",      ".corpus-head"],
  ["fetch-size",       "#fetchSize"],
  ["tab-bar",          ".tab-bar"],
  ["hf-dataset",       "#hfDataset"],
  ["model-size-sel",   "#sizePreset"],
  ["hyperparam-summary", ".hyperparam-details > summary"],
  ["controls",         ".hero-config .controls"],
  ["model-menu-btn",   "#modelMenuBtn"],
  ["notify-toggle",    ".notify-toggle"],
  ["machine-card",     ".machine-card > summary"],
  ["info-button",      ".card-title .info"],
  ["section-divider",  ".section-divider"],
  ["fine-tune-summary", "details#sec-diagnostics > summary"],
];

for (const [name, sel] of targets) {
  const el = page.locator(sel).first();
  try {
    await el.scrollIntoViewIfNeeded({ timeout: 2000 });
    await page.waitForTimeout(80);
    const out = `/tmp/nit-${name}.png`;
    await el.screenshot({ path: out, scale: "device" });
    const box = await el.boundingBox();
    console.log(`${name.padEnd(20)} ${box ? `${Math.round(box.width)}x${Math.round(box.height)}` : "??"}  -> ${out}`);
  } catch (e) {
    console.log(`${name.padEnd(20)} (not found: ${e.message.slice(0, 40)})`);
  }
}
await browser.close();
