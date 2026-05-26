// chevron_audit.mjs — capture every chevron on the page to verify they
// now look like one family (single SVG, single color, consistent rotation).
import { chromium } from "playwright";

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: 2 });
const page = await ctx.newPage();
await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
await page.locator("#welcomeSkip").click().catch(() => {});
await page.waitForTimeout(300);

const targets = [
  ["select-model-size", "#sizePreset"],
  ["select-fetch-size", "#fetchSize"],
  ["hyperparam-summary", ".hyperparam-details > summary"],
  ["model-menu-btn", "#modelMenuBtn"],
  ["machine-card-summary", ".machine-card > summary"],
  ["stats-more-summary", ".stats-more > summary"],
  ["finetune-summary", "#sec-finetune > summary"],
  ["resources-summary", ".card.collapsible.resources > summary"],
  ["diagnostics-summary", "#sec-diagnostics > summary"],
  ["advanced-summary", "details.advanced > summary"],
];

for (const [name, sel] of targets) {
  const el = page.locator(sel).first();
  try {
    await el.scrollIntoViewIfNeeded({ timeout: 1500 });
    await page.waitForTimeout(80);
    await el.screenshot({ path: `/tmp/chev-${name}.png`, scale: "device" });
    const box = await el.boundingBox();
    console.log(`${name.padEnd(26)} ${box ? `${Math.round(box.width)}x${Math.round(box.height)}` : "??"}`);
  } catch (e) {
    console.log(`${name.padEnd(26)} (not found: ${e.message.slice(0, 40)})`);
  }
}

await browser.close();
