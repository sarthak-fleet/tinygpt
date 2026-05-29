// diagnose_behemoth.mjs — figure out what actually happens when we kick off
// Behemoth WASM training in the browser. Two prior attempts timed out;
// instead of running blind we now JUST OBSERVE — no completion criterion,
// just print state every 15 s for 5 minutes.
//
// We collect:
//   - The page status banner (status / error messages)
//   - stStep, stTrain, stEta, stElapsed every 15 s
//   - All console errors / pageerrors as they happen
//   - All dialog events (window.alert / confirm) — these block training
//     until they're dismissed; if my handler misses one we'd see it hang
//
// This is read-only observation, single-shot — within AGENTS.md safety
// rules. No second phase, no backend switching, no retries.

import { chromium } from "playwright";

const browser = await chromium.launch({
  headless: false,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan"],
});
const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 } });
const page = await ctx.newPage();

const events = [];
page.on("pageerror", (e) => events.push(`[${new Date().toISOString()}] pageerror: ${e.message}`));
page.on("console", (m) => {
  if (m.type() === "error" || m.type() === "warning") {
    events.push(`[${new Date().toISOString()}] console.${m.type()}: ${m.text().slice(0, 200)}`);
  }
});
page.on("dialog", (d) => {
  events.push(`[${new Date().toISOString()}] DIALOG (${d.type()}): ${d.message().slice(0, 300)}`);
  d.accept().catch(() => {});
});

await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
await page.locator("#welcomeSkip").click({ timeout: 1500 }).catch(() => {});

console.log("Selecting Behemoth preset...");
await page.locator("#sizePreset").selectOption({ value: "behemoth" });
await page.waitForTimeout(300);

await page.evaluate(() => {
  const setVal = (id, v) => {
    const el = document.getElementById(id);
    el.value = String(v);
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  };
  setVal("maxSteps", 2);
  const back = document.getElementById("backend");
  back.value = "wasm";
  back.dataset.userPicked = "1";
  back.dispatchEvent(new Event("change", { bubbles: true }));
});

console.log("Clicking Start...");
const tStart = Date.now();
await page.locator("#start").click({ force: true });

for (let i = 0; i < 20; i++) {
  await page.waitForTimeout(15000);
  const state = await page.evaluate(() => ({
    step: document.getElementById("stStep")?.textContent,
    loss: document.getElementById("stTrain")?.textContent,
    eta: document.getElementById("stEta")?.textContent,
    elapsed: document.getElementById("stElapsed")?.textContent,
    status: document.getElementById("status")?.textContent?.slice(0, 80),
  }));
  const wall = ((Date.now() - tStart) / 1000).toFixed(0);
  console.log(`t+${wall.padStart(4)}s  step=${state.step}  loss=${state.loss}  eta=${state.eta}  elapsed=${state.elapsed}  status=${state.status}`);
  if (state.eta?.trim().toLowerCase() === "done") {
    console.log("DONE — completed normally.");
    break;
  }
}

console.log("\n--- Events captured ---");
events.forEach((e) => console.log(e));

await browser.close();
