// Capture the live "what this loss means" badge + threshold lines in the chart.
// Loads the pre-trained demo (which has final loss ~1.585 — exactly in the
// "grammar emerging" band) so we see the colour + copy switching.
import { chromium } from "playwright";

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: 2 });
const page = await ctx.newPage();
await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
await page.locator("#welcomeSkip").click().catch(() => {});
await page.locator("#loadDemoBtn").click();
await page.waitForFunction(() => {
  const el = document.getElementById("sample");
  return el && !el.disabled;
}, { timeout: 30000 });
await page.waitForTimeout(700);

await page.locator('.screen-tab[data-screen="watch"]').click().catch(() => {});
await page.waitForTimeout(300);

// Screenshot 1: the live stats row including the new loss-meaning tag.
const stats = page.locator(".stats-strip, .stats-big, .stats").first();
await stats.scrollIntoViewIfNeeded().catch(() => {});
await page.waitForTimeout(150);
await page.screenshot({ path: "/tmp/loss-meaning-1-stats.png", fullPage: false, clip: { x: 0, y: 200, width: 1400, height: 700 } });
console.log("-> /tmp/loss-meaning-1-stats.png");

// Screenshot 2: the loss curve with the new threshold bands.
const chart = page.locator(".loss-card, .chart, #lossChart").first();
await chart.scrollIntoViewIfNeeded().catch(() => {});
await page.waitForTimeout(150);
await page.screenshot({ path: "/tmp/loss-meaning-2-chart.png", fullPage: false });
console.log("-> /tmp/loss-meaning-2-chart.png");

// Read the live text for verification.
const live = await page.evaluate(() => ({
  loss: document.getElementById("stTrain")?.textContent,
  meaning: document.getElementById("stLossMeaning")?.textContent,
  cls: document.getElementById("stLossMeaning")?.className,
}));
console.log("live text:", JSON.stringify(live));

await browser.close();
