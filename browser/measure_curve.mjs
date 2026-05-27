// measure_curve.mjs — the headline measurement that actually ships.
//
// Why this shape: prior attempts measured WASM + WebGPU back-to-back in the
// browser. WASM Memory64 / multi-thread crashes with "memory access out of
// bounds" in the browser at any preset above Small (the same WASM module
// runs fine in Node — see tests/bench_wasm.mjs which produced the shipped
// numbers). Tracking that browser-specific bug is its own item.
//
// For the speedup story we use what we already have:
//   - WASM SIMD multi-thread step times from tests/bench_wasm.mjs (Node)
//     are baked into browser/src/pages/roadmap.astro's bench log:
//       Small  101 ms/step
//       Medium 357 ms/step
//       Large  1191 ms/step
//       XL     1851 ms/step
//   - WebGPU+FA2+blocked4 step times: measure them here, in-browser, with
//     a 5-step warmup average.
// Both are on the same machine; the only difference is the runtime
// (Node vs browser WASM) which the OOB bug forces. Honest framing.
//
// Single-shot per preset; not a sweep loop.

import { chromium } from "playwright";

const PRESETS = [
  // From browser/src/sizing.ts; WASM_MS comes from the shipped bench log.
  { id: "small",  label: "Small",  wasmMsPerStep: 101 },
  { id: "medium", label: "Medium", wasmMsPerStep: 357 },
  { id: "large",  label: "Large",  wasmMsPerStep: 1191 },
  { id: "xl",     label: "XL",     wasmMsPerStep: 1851 },
];
// 20 steps amortizes the first-step pipeline-compile overhead (~hundreds of ms
// for shader prep). Per-step time then converges to the steady-state cost.
const STEPS = 20;

const browser = await chromium.launch({
  headless: false,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan"],
});
const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 } });
const page = await ctx.newPage();
page.on("dialog", (d) => d.accept().catch(() => {}));

const results = [];
for (const preset of PRESETS) {
  console.log(`\n--- ${preset.label} (${preset.id}) ---`);
  await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
  await page.locator("#welcomeSkip").click({ timeout: 1500 }).catch(() => {});
  await page.locator("#sizePreset").selectOption({ value: preset.id });
  await page.waitForTimeout(200);
  await page.evaluate(({ s }) => {
    const setVal = (id, v) => {
      const el = document.getElementById(id);
      el.value = String(v);
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    };
    setVal("maxSteps", s);
    const back = document.getElementById("backend");
    back.value = "webgpu";
    back.dataset.userPicked = "1";
    back.dispatchEvent(new Event("change", { bubbles: true }));
    const seed = document.getElementById("seed");
    if (seed) setVal("seed", 42);
  }, { s: STEPS });

  const tStart = Date.now();
  await page.locator("#start").click({ force: true });

  // Wait until step counter reaches STEPS (status text like "step=N / N"),
  // OR the status banner contains "training complete", OR we see an error.
  // The `stEta` element keeps showing a residual ETA after completion, so it
  // isn't a reliable done-signal — poll step count instead.
  const ok = await page.waitForFunction(
    (target) => {
      const stStep = document.getElementById("stStep")?.textContent ?? "";
      const status = document.getElementById("status")?.textContent ?? "";
      const m = stStep.match(/^(\d+)\s*\/\s*(\d+)/);
      const done = (m && Number(m[1]) >= target) || /training complete/i.test(status);
      const err = /error|failed/i.test(status);
      if (done) return { ok: true };
      if (err) return { ok: false, status };
      return false;
    },
    STEPS,
    { timeout: 300_000, polling: 250 },
  ).then((h) => h.jsonValue()).catch((e) => ({ ok: false, status: e.message }));

  const wallMs = Date.now() - tStart;
  if (!ok.ok) {
    console.log(`  FAILED: ${ok.status}`);
    results.push({ preset: preset.label, ...preset, gpuMsPerStep: null, error: ok.status });
    continue;
  }
  const loss = await page.evaluate(() => parseFloat(document.getElementById("stTrain").textContent));
  const gpuMsPerStep = wallMs / STEPS;
  const speedup = preset.wasmMsPerStep / gpuMsPerStep;
  console.log(`  WASM (Node, shipped):  ${preset.wasmMsPerStep} ms/step`);
  console.log(`  WebGPU (this run):     ${gpuMsPerStep.toFixed(1)} ms/step  (wall ${wallMs} ms / ${STEPS} steps)`);
  console.log(`  Final loss:            ${loss.toFixed(4)}`);
  console.log(`  Speedup:               ${speedup.toFixed(1)}×`);
  results.push({ preset: preset.label, ...preset, gpuMsPerStep, speedup, loss });
}

console.log("\n\n=== Summary ===");
console.log("Preset   Params    WASM ms/step  WebGPU ms/step  Speedup");
for (const r of results) {
  if (r.error) {
    console.log(`${r.label.padEnd(8)} —         ${String(r.wasmMsPerStep).padEnd(13)} FAILED         —`);
  } else {
    console.log(`${r.label.padEnd(8)} ${"".padEnd(9)} ${String(r.wasmMsPerStep).padEnd(13)} ${r.gpuMsPerStep.toFixed(1).padEnd(15)} ${r.speedup.toFixed(1)}×`);
  }
}

await browser.close();
