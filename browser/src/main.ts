/**
 * main.ts — UI / main-thread controller (Phase 4).
 *
 * The main thread owns ONLY the UI: capability panel, corpus input, controls,
 * and the loss chart. Training runs entirely in worker.ts, so the page stays
 * responsive — this file never does model math.
 *
 * Guide: docs/browser_notes.md ("Web Worker")
 */

import { LossChart } from "./charts";
import { detectCapabilities } from "./runtime_detect";
import { loadRun, requestDurableStorage, saveRun } from "./storage";
import { DEFAULT_CONFIG, type FromWorker, type RunConfig, type ToWorker } from "./types";

const byId = <T extends HTMLElement>(id: string): T =>
  document.getElementById(id) as T;

const els = {
  caps: byId<HTMLDivElement>("caps"),
  corpus: byId<HTMLTextAreaElement>("corpus"),
  start: byId<HTMLButtonElement>("start"),
  pause: byId<HTMLButtonElement>("pause"),
  stop: byId<HTMLButtonElement>("stop"),
  sample: byId<HTMLButtonElement>("sample"),
  status: byId<HTMLDivElement>("status"),
  output: byId<HTMLDivElement>("output"),
  stStep: byId<HTMLElement>("stStep"),
  stTrain: byId<HTMLElement>("stTrain"),
  stVal: byId<HTMLElement>("stVal"),
  stToks: byId<HTMLElement>("stToks"),
  stBackend: byId<HTMLElement>("stBackend"),
};

const canvas = byId<HTMLCanvasElement>("chart");
canvas.width = canvas.clientWidth || 560;
canvas.height = 220;
const chart = new LossChart(canvas);

const worker = new Worker(new URL("./worker.ts", import.meta.url), {
  type: "module",
});
const send = (msg: ToWorker) => worker.postMessage(msg);

let paused = false;
let history: { step: number; trainLoss: number; valLoss?: number }[] = [];

// --- config ---------------------------------------------------------------
function readConfig(): RunConfig {
  const intOf = (id: string) => parseInt(byId<HTMLInputElement>(id).value, 10);
  const dModel = parseInt(byId<HTMLSelectElement>("dModel").value, 10);
  return {
    ctx: intOf("ctx"),
    layers: intOf("layers"),
    heads: 3, // d_model options are all multiples of 3
    dModel,
    dMlp: dModel * 4,
    batchSize: intOf("batch"),
    learningRate: parseFloat(byId<HTMLInputElement>("lr").value),
    gradClip: DEFAULT_CONFIG.gradClip,
    maxSteps: intOf("maxSteps"),
    evalEvery: DEFAULT_CONFIG.evalEvery,
    seed: DEFAULT_CONFIG.seed,
  };
}

// --- button state ---------------------------------------------------------
function setRunning(on: boolean): void {
  els.start.disabled = on;
  els.pause.disabled = !on;
  els.stop.disabled = !on;
}

// --- controls -------------------------------------------------------------
els.start.addEventListener("click", () => {
  const text = els.corpus.value;
  if (text.length < 80) {
    els.status.textContent = "corpus is very short — add more text";
    return;
  }
  history = [];
  chart.reset();
  els.output.textContent = "Training… generate once a few steps have run.";
  paused = false;
  els.pause.textContent = "Pause";
  setRunning(true);
  els.sample.disabled = false;
  send({ type: "train", text, config: readConfig() });
});

els.pause.addEventListener("click", () => {
  paused = !paused;
  els.pause.textContent = paused ? "Resume" : "Pause";
  send({ type: paused ? "pause" : "resume" });
});

els.stop.addEventListener("click", () => send({ type: "stop" }));

els.sample.addEventListener("click", () => {
  send({
    type: "sample",
    prompt: byId<HTMLInputElement>("prompt").value,
    tokens: parseInt(byId<HTMLInputElement>("genTokens").value, 10),
    temperature: parseFloat(byId<HTMLInputElement>("temp").value),
  });
  els.output.textContent = "generating…";
});

// --- worker messages ------------------------------------------------------
worker.onmessage = (e: MessageEvent<FromWorker>) => {
  const msg = e.data;
  switch (msg.type) {
    case "status":
      els.status.textContent = msg.message;
      break;
    case "progress": {
      const p = msg.progress;
      history.push({ step: p.step, trainLoss: p.trainLoss, valLoss: p.valLoss });
      chart.addPoint({ step: p.step, trainLoss: p.trainLoss, valLoss: p.valLoss });
      els.stStep.textContent = `${p.step} / ${p.maxSteps}`;
      els.stTrain.textContent = p.trainLoss.toFixed(4);
      els.stVal.textContent = p.valLoss?.toFixed(4) ?? "–";
      els.stToks.textContent = Math.round(p.tokensPerSecond).toLocaleString();
      els.stBackend.textContent = p.backend;
      break;
    }
    case "sample":
      els.output.textContent = msg.text;
      break;
    case "done":
      setRunning(false);
      els.status.textContent =
        msg.reason === "finished" ? "training complete" : "training stopped";
      void saveRun({
        savedAt: new Date().toISOString(),
        config: readConfig(),
        lossHistory: history,
      });
      break;
    case "error":
      setRunning(false);
      els.status.textContent = `error: ${msg.message}`;
      break;
  }
};

worker.onerror = (e) => {
  setRunning(false);
  els.status.textContent = `worker error: ${e.message}`;
};

// --- startup --------------------------------------------------------------
async function init(): Promise<void> {
  const caps = await detectCapabilities();
  const storage = await requestDurableStorage();
  const pill = (label: string, on: boolean) =>
    `<span class="pill ${on ? "on" : "off"}">${label} ${on ? "✓" : "—"}</span>`;
  els.caps.innerHTML =
    pill("WebGPU", caps.webgpu) +
    pill("WASM SIMD", caps.wasmSimd) +
    pill("cross-origin isolated", caps.crossOriginIsolated) +
    `<span class="pill on">training backend: ${caps.active}</span>` +
    `<span class="pill off">OPFS quota ~${storage.quotaMB} MB</span>`;

  // Restore the previous run's chart, if one was persisted.
  const prev = await loadRun();
  if (prev && prev.lossHistory.length > 0) {
    history = prev.lossHistory;
    for (const pt of history) chart.addPoint(pt);
    els.status.textContent = `restored chart from a previous run (${history.length} points)`;
  }
}

void init();
