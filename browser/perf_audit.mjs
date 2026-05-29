// perf_audit.mjs — manual Core Web Vitals + long-task + load-timing audit
// of the live site, using Playwright's CDP + Performance APIs. No external
// services. Captures TTFB, FCP, LCP, every long task >50ms, every resource
// load with its timing, and what's blocking the main thread.
//
// Run:  node browser/perf_audit.mjs
//       AUDIT_URL=http://localhost:5173 node browser/perf_audit.mjs

import { chromium } from "playwright";

const SITE = process.env.AUDIT_URL || "https://tinygpt.sarthakagrawal.dev";

const browser = await chromium.launch({ headless: true });
const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 } });
const page = await ctx.newPage();

// Set up observers BEFORE navigation so we capture the cold load.
await page.addInitScript(() => {
  window.__perf = { lcp: 0, longTasks: [], resources: [], paints: {} };
  // Largest Contentful Paint
  try {
    new PerformanceObserver((list) => {
      const last = list.getEntries().pop();
      if (last) window.__perf.lcp = Math.max(window.__perf.lcp, last.startTime);
    }).observe({ type: "largest-contentful-paint", buffered: true });
  } catch {}
  // Long tasks (>50ms blocking the main thread)
  try {
    new PerformanceObserver((list) => {
      for (const e of list.getEntries()) {
        window.__perf.longTasks.push({ start: e.startTime, duration: e.duration, name: e.name });
      }
    }).observe({ type: "longtask", buffered: true });
  } catch {}
  // Paint timing
  try {
    new PerformanceObserver((list) => {
      for (const e of list.getEntries()) {
        window.__perf.paints[e.name] = e.startTime;
      }
    }).observe({ type: "paint", buffered: true });
  } catch {}
});

console.log(`\n>> Auditing ${SITE}\n${"=".repeat(60)}\n`);

const t0 = Date.now();
const resp = await page.goto(SITE, { waitUntil: "networkidle", timeout: 30_000 });
const navWall = Date.now() - t0;

// Give the page another 4s to settle so any deferred work shows up in long tasks
await page.waitForTimeout(4000);

const perf = await page.evaluate(() => {
  const nav = performance.getEntriesByType("navigation")[0] || {};
  const resources = performance.getEntriesByType("resource").map((r) => ({
    name: r.name,
    duration: r.duration,
    transferSize: r.transferSize || 0,
    encodedBodySize: r.encodedBodySize || 0,
    decodedBodySize: r.decodedBodySize || 0,
    initiatorType: r.initiatorType,
    renderBlocking: r.renderBlockingStatus || null,
  }));
  return {
    nav: {
      ttfb: nav.responseStart - nav.requestStart,
      domInteractive: nav.domInteractive,
      domContentLoadedEventEnd: nav.domContentLoadedEventEnd,
      loadEventEnd: nav.loadEventEnd,
    },
    paints: window.__perf?.paints || {},
    lcp: window.__perf?.lcp || 0,
    longTasks: window.__perf?.longTasks || [],
    resources,
    heapUsedMB: performance.memory ? performance.memory.usedJSHeapSize / 1024 / 1024 : null,
  };
});

console.log(`Navigation wall time: ${navWall}ms`);
console.log(`HTTP status: ${resp?.status()}`);
console.log("");
console.log("CORE METRICS");
console.log(`  TTFB:                  ${perf.nav.ttfb.toFixed(0)}ms  (good < 800ms)`);
console.log(`  First Paint:           ${(perf.paints["first-paint"] || 0).toFixed(0)}ms`);
console.log(`  First Contentful Paint:${(perf.paints["first-contentful-paint"] || 0).toFixed(0)}ms  (good < 1800ms)`);
console.log(`  Largest Contentful Paint:${perf.lcp.toFixed(0)}ms  (good < 2500ms)`);
console.log(`  DOM Interactive:       ${perf.nav.domInteractive.toFixed(0)}ms`);
console.log(`  DOMContentLoaded:      ${perf.nav.domContentLoadedEventEnd.toFixed(0)}ms`);
console.log(`  Load event:            ${perf.nav.loadEventEnd.toFixed(0)}ms`);
if (perf.heapUsedMB) console.log(`  JS heap used:          ${perf.heapUsedMB.toFixed(1)} MB`);

console.log("");
console.log(`LONG TASKS (>50ms blocking main thread): ${perf.longTasks.length}`);
const longTasksByDur = [...perf.longTasks].sort((a, b) => b.duration - a.duration);
const totalBlock = perf.longTasks.reduce((acc, t) => acc + Math.max(0, t.duration - 50), 0);
console.log(`  Total Blocking Time (TBT, excess over 50ms): ${totalBlock.toFixed(0)}ms  (good < 200ms)`);
for (const t of longTasksByDur.slice(0, 8)) {
  console.log(`    ${t.duration.toFixed(0)}ms  at t=${t.start.toFixed(0)}ms  (${t.name})`);
}

console.log("");
console.log("HEAVY RESOURCES (>20KB transfer or >200ms duration)");
const heavy = perf.resources
  .filter((r) => r.transferSize > 20 * 1024 || r.duration > 200)
  .sort((a, b) => b.duration - a.duration);
for (const r of heavy.slice(0, 12)) {
  const short = r.name.replace(SITE, "").replace(/^https?:\/\/[^/]+/, "<ext>");
  const kb = (r.transferSize / 1024).toFixed(0);
  const decoded = (r.decodedBodySize / 1024).toFixed(0);
  const block = r.renderBlocking === "blocking" ? "  [render-blocking]" : "";
  console.log(`  ${r.duration.toFixed(0)}ms  ${kb}KB→${decoded}KB  ${r.initiatorType.padEnd(8)} ${short.slice(0, 60)}${block}`);
}

console.log("");
console.log("RENDER-BLOCKING RESOURCES");
const blocking = perf.resources.filter((r) => r.renderBlocking === "blocking");
if (blocking.length === 0) console.log("  none flagged");
for (const r of blocking) {
  console.log(`  ${r.duration.toFixed(0)}ms  ${r.name.replace(SITE, "").slice(0, 80)}`);
}

console.log("\n=== Top 3 actionable items ===");
const items = [];
if (perf.nav.ttfb > 800) items.push(`TTFB ${perf.nav.ttfb.toFixed(0)}ms — server / edge-network latency; not much you can fix in code`);
if (totalBlock > 200) items.push(`Total Blocking Time ${totalBlock.toFixed(0)}ms — main thread chewing for ${(totalBlock / 1000).toFixed(1)}s after FCP. Defer heavy init.`);
const heaviestResource = heavy[0];
if (heaviestResource && heaviestResource.duration > 500) {
  items.push(`${heaviestResource.name.replace(SITE, "").slice(0, 50)} took ${heaviestResource.duration.toFixed(0)}ms — consider lazy/defer`);
}
const heaviestLongTask = longTasksByDur[0];
if (heaviestLongTask && heaviestLongTask.duration > 200) {
  items.push(`Longest single task: ${heaviestLongTask.duration.toFixed(0)}ms at t=${heaviestLongTask.start.toFixed(0)}ms — break it up or defer`);
}
const jsRes = perf.resources.filter((r) => r.initiatorType === "script" && r.transferSize > 100 * 1024);
if (jsRes.length > 0) {
  items.push(`${jsRes.length} JS bundle(s) > 100 KB — biggest is ${(jsRes[0].transferSize / 1024).toFixed(0)} KB`);
}

for (const i of items.slice(0, 5)) console.log(`  • ${i}`);
if (items.length === 0) console.log("  No major issues detected.");

await browser.close();
