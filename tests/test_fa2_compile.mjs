// test_fa2_compile.mjs — verify that the FA2 forward kernel actually
// compiles against a real WebGPU shader compiler. The Node-side parity
// test only checks the algorithm; it doesn't catch WGSL syntax errors
// or driver-specific validation issues.
//
// Approach: open a headed Chromium with WebGPU enabled, fetch the
// attention_fa2.wgsl source from the dev server, and try to create a
// compute pipeline from it using the same g0..g5 + p uniform bind
// layout that ops.ts uses. If pipeline creation succeeds, the shader
// is valid for production integration.
import { chromium } from "playwright";
import fs from "node:fs";
import path from "node:path";
import url from "node:url";

const here = path.dirname(url.fileURLToPath(import.meta.url));
const wgslPath = path.join(here, "..", "webgpu", "attention_fa2.wgsl");
const shaderSrc = fs.readFileSync(wgslPath, "utf8");

const browser = await chromium.launch({
  headless: false,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-vulkan"],
});
const ctx = await browser.newContext();
const page = await ctx.newPage();
await page.goto("http://localhost:5173/", { waitUntil: "domcontentloaded" });

const result = await page.evaluate(async (text) => {
  if (!navigator.gpu) return { ok: false, reason: "no navigator.gpu" };
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { ok: false, reason: "no adapter" };
  const device = await adapter.requestDevice();
  device.pushErrorScope?.("validation");

  const module = device.createShaderModule({ code: text });
  // wait for any compile errors
  const info = await module.getCompilationInfo?.();
  const errors = info?.messages?.filter((m) => m.type === "error") ?? [];
  if (errors.length) {
    return { ok: false, reason: "WGSL compile errors", errors: errors.map((e) => ({ line: e.lineNum, msg: e.message })).slice(0, 5) };
  }

  // Same bind layout as ops.ts
  const storage = (binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } });
  const layout = device.createBindGroupLayout({
    entries: [
      storage(0), storage(1), storage(2), storage(3), storage(4), storage(5),
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });

  try {
    const pipeline = await device.createComputePipelineAsync({
      layout: pipelineLayout,
      compute: { module, entryPoint: "fa2_forward" },
    });
    void pipeline;
    return { ok: true, msg: "fa2_forward pipeline created OK" };
  } catch (e) {
    const err = await device.popErrorScope?.();
    return { ok: false, reason: "pipeline creation failed", err: String(e), validation: err ? String(err) : null };
  }
}, shaderSrc);

console.log(JSON.stringify(result, null, 2));
await browser.close();
process.exit(result.ok ? 0 : 1);
