// audit_mobile.mjs — sanity-check the mobile layout for regressions
// after the Memory64 pill + Behemoth preset additions.
import { chromium } from "playwright";

const browser = await chromium.launch();
const ctx = await browser.newContext({
  viewport: { width: 390, height: 844 }, // iPhone 14 Pro
  deviceScaleFactor: 3,
});
const page = await ctx.newPage();
page.on("pageerror", (e) => console.log("[pageerror]", e.message));

await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
await page.locator("#welcomeSkip").click().catch(() => {});
await page.waitForTimeout(300);
await page.screenshot({ path: "/tmp/mobile-1-setup.png", fullPage: true });
console.log("-> /tmp/mobile-1-setup.png");

// Pick behemoth — verify the dropdown text wraps OK
await page.locator("#sizePreset").selectOption({ value: "behemoth" });
await page.waitForTimeout(200);
await page.screenshot({ path: "/tmp/mobile-2-behemoth.png", fullPage: true });
console.log("-> /tmp/mobile-2-behemoth.png");

await browser.close();
