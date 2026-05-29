// measure_massive.mjs — the headline measurement, second try. Behemoth
// crashed the WASM Memory64 build with "memory access out of bounds" after
// ~2 min (see diagnose_behemoth.mjs output). Mega timed out. Massive is
// the right cut: ~25M params, ctx=256, same model family as Mega/Behemoth
// but stays inside the 32-bit WASM heap and well within every kernel's
// bounds. Big enough that FA2 + blocked-matmul + Memory64 still dominate
// step time.
//
// Single-shot heavy run, per AGENTS.md "Safety rules". 5 training steps
// per phase. Same seed, same data, same model — the only difference is
// the backend.

import { chromium } from "playwright";

const STEPS = 5;
const PRESET = "xl";

const browser = await chromium.launch({
  headless: false,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan"],
});
const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 } });
const page = await ctx.newPage();
page.on("dialog", (d) => d.accept().catch(() => {}));
page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));

async function runPhase(backend) {
  await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
  await page.locator("#welcomeSkip").click({ timeout: 1500 }).catch(() => {});
  await page.locator("#sizePreset").selectOption({ value: PRESET });
  await page.waitForTimeout(200);
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

  // Watch for either completion OR an error status; bail on error.
  const result = await page.waitForFunction(
    () => {
      const eta = document.getElementById("stEta");
      const status = document.getElementById("status");
      const isDone = eta && eta.textContent.trim().toLowerCase() === "done";
      const hasError = status?.textContent?.toLowerCase().includes("error");
      if (isDone) return { ok: true };
      if (hasError) return { ok: false, status: status.textContent };
      return false;
    },
    null, { timeout: 900_000, polling: 1000 },
  );
  const r = await result.jsonValue();
  const wallSec = (Date.now() - tStart) / 1000;
  if (!r.ok) {
    return { backend, error: r.status, wallSec };
  }
  const loss = await page.evaluate(() => parseFloat(document.getElementById("stTrain").textContent));
  const elapsed = await page.evaluate(() => document.getElementById("stElapsed").textContent.trim());
  return { backend, wallSec, loss, elapsed };
}

console.log(`XL preset · 8L · d=256 · ctx=128 · ~6.4M params · ${STEPS} steps · seed 42\n`);

console.log("Phase 1: WASM SIMD (multi-thread) ...");
const wasm = await runPhase("wasm");
if (wasm.error) {
  console.log(`  FAILED: ${wasm.error}`);
} else {
  console.log(`  loss ${wasm.loss.toFixed(4)}   elapsed ${wasm.elapsed}   wall ${wasm.wallSec.toFixed(1)} s   (${(wasm.wallSec / STEPS).toFixed(2)} s/step)`);
}

console.log("\nPhase 2: WebGPU + FA2 + blocked4 ...");
const gpu = await runPhase("webgpu");
if (gpu.error) {
  console.log(`  FAILED: ${gpu.error}`);
} else {
  console.log(`  loss ${gpu.loss.toFixed(4)}   elapsed ${gpu.elapsed}   wall ${gpu.wallSec.toFixed(1)} s   (${(gpu.wallSec / STEPS).toFixed(2)} s/step)`);
}

if (!wasm.error && !gpu.error) {
  const speedup = wasm.wallSec / gpu.wallSec;
  const drift = Math.abs(wasm.loss - gpu.loss) / Math.max(Math.abs(wasm.loss), 1e-6);
  console.log(`\n=== Massive end-to-end speedup: ${speedup.toFixed(1)}× ===`);
  console.log(`Loss drift: ${(drift * 100).toFixed(1)}%`);
}

await browser.close();
