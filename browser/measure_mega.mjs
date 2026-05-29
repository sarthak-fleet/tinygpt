// measure_mega.mjs — single-shot end-to-end measurement on the Mega preset
// (ctx=512, ~25M params) to back up a headline speedup number bigger than
// the conservative 9.7× the Small-preset parity test produces.
//
// Per AGENTS.md "Safety rules for heavy GPU/compile loops" — this is a
// single-shot heavy run, not a loop. No benchmark sweep, no install, no
// parallel compiles. After it finishes, kill the browser cleanly.
//
// Mega preset specifics:
//   layers=14, d_model=384, ctx=512, batch=2  →  ~25M params, attention
//   matrix per layer per batch = 2 · 8 · 512² · 4 B = 16 MB (the regime
//   FA2 was built for: dropping the writeback saves 16 MB · 14 layers =
//   ~224 MB of global memory traffic per step that now stays on-chip).
//
// We run 15 steps. WASM SIMD takes ~3-5 s/step on Mega; WebGPU+FA2 should
// be a fraction of that. Same seed both phases, drift-< 5% gate.

import { chromium } from "playwright";

const STEPS = 5;
const PRESET = "mega";

const browser = await chromium.launch({
  headless: false,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan"],
});
const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 } });
const page = await ctx.newPage();
page.on("dialog", (d) => d.accept());
page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));

await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
await page.locator("#welcomeSkip").click({ timeout: 2000 }).catch(() => {});
await page.waitForTimeout(400);

async function runPhase(backend) {
  // Reset to a known fresh state by reloading.
  await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
  await page.locator("#welcomeSkip").click({ timeout: 1500 }).catch(() => {});

  await page.locator("#sizePreset").selectOption({ value: PRESET });
  await page.waitForTimeout(150);
  await page.evaluate(({ s, b }) => {
    const setVal = (id, v) => {
      const el = document.getElementById(id);
      el.value = String(v);
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    };
    setVal("maxSteps", s);
    const back = document.getElementById("backend");
    back.value = b;
    back.dataset.userPicked = "1";
    back.dispatchEvent(new Event("change", { bubbles: true }));
    const seed = document.getElementById("seed");
    if (seed) setVal("seed", 42);
  }, { s: STEPS, b: backend });

  const tStart = Date.now();
  await page.locator("#start").click({ force: true });

  // Wait for completion (stEta == "done") or up to 10 min.
  await page.waitForFunction(
    () => {
      const eta = document.getElementById("stEta");
      return eta && eta.textContent.trim().toLowerCase() === "done";
    },
    null, { timeout: 600_000, polling: 500 },
  );
  const wallSec = (Date.now() - tStart) / 1000;
  const loss = await page.evaluate(() => parseFloat(document.getElementById("stTrain").textContent));
  const elapsed = await page.evaluate(() => document.getElementById("stElapsed").textContent.trim());
  return { wallSec, loss, elapsed };
}

console.log(`Mega preset · ${STEPS} steps · seed 42\n`);

console.log("Phase 1: WASM ...");
const wasm = await runPhase("wasm");
console.log(`  loss ${wasm.loss.toFixed(4)}   elapsed ${wasm.elapsed}   wall ${wasm.wallSec.toFixed(1)} s`);

console.log("Phase 2: WebGPU ...");
const gpu = await runPhase("webgpu");
console.log(`  loss ${gpu.loss.toFixed(4)}   elapsed ${gpu.elapsed}   wall ${gpu.wallSec.toFixed(1)} s`);

const speedup = wasm.wallSec / gpu.wallSec;
const drift = Math.abs(wasm.loss - gpu.loss) / Math.max(Math.abs(wasm.loss), 1e-6);
console.log(`\n=== Mega preset speedup: ${speedup.toFixed(1)}× end-to-end ===`);
console.log(`Loss drift: ${(drift * 100).toFixed(1)}%`);
console.log(drift < 0.05 ? "PASS" : "FAIL — out of tolerance");

await browser.close();
