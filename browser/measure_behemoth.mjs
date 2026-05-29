// measure_behemoth.mjs — the headline measurement. 473M-param Behemoth
// preset (Memory64) on WASM vs WebGPU+FA2, two training steps each, same
// seed. Behemoth's matmul shapes (M=batch·ctx=512, K=1280, N=5120 etc.)
// are exactly the regime every optimization in this project was built
// for: blocked4 matmul, FA2 forward + backward, Memory64. So the
// expected speedup is huge.
//
// Wall-time budget per AGENTS.md "Safety rules": single-shot heavy run,
// not a loop. WASM phase ≈ 2 × 82 s/step = ~3 min (measured 82 s/step
// earlier in the Memory64 verification). WebGPU phase should be a small
// fraction of that. Total ≈ 4 min.
//
// We don't care about loss drift on 2 steps of Behemoth-on-tiny-corpus —
// that's a deeply under-data run and would be hostile to either backend.
// The number we want is timing, not training quality.

import { chromium } from "playwright";

const STEPS = 2;
const PRESET = "behemoth";

const browser = await chromium.launch({
  headless: false,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan"],
});
const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 } });
const page = await ctx.newPage();
page.on("dialog", (d) => d.accept());
page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));

async function runPhase(backend) {
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

  await page.waitForFunction(
    () => {
      const eta = document.getElementById("stEta");
      return eta && eta.textContent.trim().toLowerCase() === "done";
    },
    null, { timeout: 900_000, polling: 1000 },
  );
  const wallSec = (Date.now() - tStart) / 1000;
  const loss = await page.evaluate(() => parseFloat(document.getElementById("stTrain").textContent));
  const elapsed = await page.evaluate(() => document.getElementById("stElapsed").textContent.trim());
  return { wallSec, loss, elapsed };
}

console.log(`Behemoth preset · 24L · d=1280 · ctx=512 · ~473M params · ${STEPS} steps · seed 42\n`);

console.log("Phase 1: WASM SIMD ...");
const wasm = await runPhase("wasm");
console.log(`  loss ${wasm.loss.toFixed(4)}   elapsed ${wasm.elapsed}   wall ${wasm.wallSec.toFixed(1)} s   (${(wasm.wallSec / STEPS).toFixed(1)} s/step)`);

console.log("Phase 2: WebGPU + FA2 + blocked4 ...");
const gpu = await runPhase("webgpu");
console.log(`  loss ${gpu.loss.toFixed(4)}   elapsed ${gpu.elapsed}   wall ${gpu.wallSec.toFixed(1)} s   (${(gpu.wallSec / STEPS).toFixed(1)} s/step)`);

const speedup = wasm.wallSec / gpu.wallSec;
console.log(`\n=== Behemoth end-to-end speedup: ${speedup.toFixed(1)}× ===`);
console.log(`WASM:   ${(wasm.wallSec / STEPS).toFixed(1)} s/step`);
console.log(`WebGPU: ${(gpu.wallSec / STEPS).toFixed(1)} s/step`);

await browser.close();
