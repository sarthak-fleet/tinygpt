// screenshot_evo.mjs — capture the speed-evolution chart on /roadmap.
import { chromium } from "playwright";

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1200, height: 900 }, deviceScaleFactor: 2 });
const page = await ctx.newPage();
await page.goto("http://localhost:5173/roadmap.html", { waitUntil: "networkidle" });
await page.waitForTimeout(300);

const card = page.locator(".speed-evo");
await card.scrollIntoViewIfNeeded();
await page.waitForTimeout(150);
await card.screenshot({ path: "/tmp/evo-chart.png" });
console.log("-> /tmp/evo-chart.png");

// Also a mobile-width version
const mobile = await browser.newContext({ viewport: { width: 380, height: 800 }, deviceScaleFactor: 2 });
const mp = await mobile.newPage();
await mp.goto("http://localhost:5173/roadmap.html", { waitUntil: "networkidle" });
await mp.waitForTimeout(300);
const mcard = mp.locator(".speed-evo");
await mcard.scrollIntoViewIfNeeded();
await mp.waitForTimeout(150);
await mcard.screenshot({ path: "/tmp/evo-chart-mobile.png" });
console.log("-> /tmp/evo-chart-mobile.png");

await browser.close();
