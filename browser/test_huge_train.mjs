// test_huge_train.mjs — long-run feasibility check for "wait 15-30 minutes, get a usable model"
//
// Config: Huge preset (12L, d=256, ctx=256, batch=8) on WebGPU, default 1500 steps.
// Question: does loss reach val<1.5 (or train<1.5) before time/memory falls over?
// Polls every 15s, logs step / loss / wall ms-per-step. At the end, generates
// a short sample so we can eyeball "is this readable Shakespeare or gibberish?"

import { chromium } from "playwright";

const PRESET = "huge";
const POLL_MS = 15_000;
const MAX_WALL_MS = 45 * 60 * 1000;   // 45 min hard cap — we expect ~15-25 min
const GEN_TOKENS = 240;

const browser = await chromium.launch({
  headless: false,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan"],
});
const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 } });
const page = await ctx.newPage();
page.on("dialog", (d) => d.accept().catch(() => {}));
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
page.on("console", (m) => {
  const t = m.type();
  if (t === "error" || t === "warning") console.log(`[${t}]`, m.text());
});

await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
await page.locator("#welcomeSkip").click({ timeout: 1500 }).catch(() => {});

await page.evaluate((preset) => {
  const setVal = (id, v) => {
    const el = document.getElementById(id);
    el.value = String(v);
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  };
  document.getElementById("sizePreset").value = preset;
  document.getElementById("sizePreset").dispatchEvent(new Event("change", { bubbles: true }));
  const back = document.getElementById("backend");
  back.value = "webgpu";
  back.dataset.userPicked = "1";
  back.dispatchEvent(new Event("change", { bubbles: true }));
  const seed = document.getElementById("seed");
  if (seed) setVal("seed", 42);
}, PRESET);

const cfg = await page.evaluate(() => ({
  preset: document.getElementById("sizePreset").value,
  maxSteps: document.getElementById("maxSteps").value,
  backend: document.getElementById("backend").value,
  params: document.querySelector("[data-stat='params']")?.textContent ?? document.getElementById("status")?.textContent ?? "?",
}));
console.log("config:", JSON.stringify(cfg));

const tStart = Date.now();
await page.locator("#start").click({ force: true });
console.log("training started at", new Date().toISOString());

const trajectory = [];
let lastStep = 0;
let lastT = tStart;

while (true) {
  await new Promise((r) => setTimeout(r, POLL_MS));
  const wall = Date.now() - tStart;
  const s = await page.evaluate(() => ({
    step: document.getElementById("stStep")?.textContent ?? "",
    train: document.getElementById("stTrain")?.textContent ?? "",
    valEl: document.getElementById("stVal")?.textContent ?? "",
    eta: document.getElementById("stEta")?.textContent ?? "",
    status: document.getElementById("status")?.textContent ?? "",
    memHint: (performance.memory && performance.memory.usedJSHeapSize)
      ? Math.round(performance.memory.usedJSHeapSize / 1e6) + " MB JS heap"
      : null,
  }));
  const m = s.step.match(/^(\d+)\s*\/\s*(\d+)/);
  const curStep = m ? Number(m[1]) : 0;
  const maxStep = m ? Number(m[2]) : 0;
  const dStep = curStep - lastStep;
  const dT = Date.now() - lastT;
  const recentMsPerStep = dStep > 0 ? (dT / dStep).toFixed(0) : "—";
  lastStep = curStep; lastT = Date.now();
  trajectory.push({ wallMs: wall, step: curStep, train: s.train, val: s.valEl });
  console.log(
    `t+${(wall / 1000).toFixed(0)}s  step=${curStep}/${maxStep}  train=${s.train}  val=${s.valEl}  recent=${recentMsPerStep}ms/step  ${s.memHint ?? ""}  status="${s.status.slice(0, 80)}"`,
  );
  const done = curStep >= maxStep && maxStep > 0;
  const err = /error|failed/i.test(s.status);
  if (done) { console.log("=== training complete ==="); break; }
  if (err)  { console.log("=== ERROR ==="); break; }
  if (wall > MAX_WALL_MS) { console.log("=== HARD-CAP ==="); break; }
}

// Generate a sample
console.log("\n--- generating sample ---");
const genBtn = page.locator("#generate, button:has-text('Generate')").first();
if (await genBtn.count() > 0) {
  await genBtn.click({ force: true }).catch(() => {});
  await new Promise((r) => setTimeout(r, 8000));
}
const sample = await page.evaluate(() => {
  const el = document.getElementById("genOut") ?? document.getElementById("sample") ?? document.querySelector("[data-role='generation']");
  return el ? el.textContent : "(no #genOut element found)";
});
console.log("--- sample ---");
console.log(sample.slice(0, 2000));
console.log("--- /sample ---");

console.log("\n--- trajectory (every 5th sample) ---");
for (let i = 0; i < trajectory.length; i += 5) {
  const r = trajectory[i];
  console.log(`  t=${(r.wallMs / 1000).toFixed(0)}s step=${r.step} train=${r.train} val=${r.val}`);
}

await browser.close();
