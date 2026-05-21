/**
 * storage.ts — browser-local persistence via OPFS (Phase 4).
 *
 * Saves a run's lightweight state (config + loss history) to the Origin-Private
 * File System so the chart survives a page refresh. OPFS is subject to storage
 * quota; clearing site data deletes it — so durability is requested up front.
 *
 * Full model-weight checkpointing (weights + AdamW moments) is milestone 7;
 * it needs weight export/import added to the WASM C ABI. This module provides
 * the OPFS plumbing that milestone-7 work will build on.
 *
 * Guide: docs/browser_notes.md ("Checkpointing")
 */

const RUN_FILE = "last_run.json";

export interface RunSnapshot {
  savedAt: string;
  config: unknown;
  lossHistory: { step: number; trainLoss: number; valLoss?: number }[];
}

/** Ask the browser to keep our storage across eviction, and report the quota. */
export async function requestDurableStorage(): Promise<{
  persisted: boolean;
  quotaMB: number;
}> {
  let persisted = false;
  let quotaMB = 0;
  try {
    if (navigator.storage?.persist) persisted = await navigator.storage.persist();
    if (navigator.storage?.estimate) {
      const est = await navigator.storage.estimate();
      quotaMB = Math.round((est.quota ?? 0) / (1024 * 1024));
    }
  } catch {
    /* storage API unavailable — best-effort only */
  }
  return { persisted, quotaMB };
}

function opfsAvailable(): boolean {
  return typeof navigator !== "undefined" && !!navigator.storage?.getDirectory;
}

/** Persist a run snapshot to OPFS. Returns false if OPFS is unavailable. */
export async function saveRun(snapshot: RunSnapshot): Promise<boolean> {
  if (!opfsAvailable()) return false;
  try {
    const root = await navigator.storage.getDirectory();
    const handle = await root.getFileHandle(RUN_FILE, { create: true });
    const writable = await handle.createWritable();
    await writable.write(JSON.stringify(snapshot));
    await writable.close();
    return true;
  } catch {
    return false;
  }
}

/** Load the last persisted run snapshot, or null if there is none. */
export async function loadRun(): Promise<RunSnapshot | null> {
  if (!opfsAvailable()) return null;
  try {
    const root = await navigator.storage.getDirectory();
    const handle = await root.getFileHandle(RUN_FILE);
    const text = await (await handle.getFile()).text();
    return JSON.parse(text) as RunSnapshot;
  } catch {
    return null; // no prior run, or OPFS read failed
  }
}

/** Delete the persisted run, if any. */
export async function clearRun(): Promise<void> {
  if (!opfsAvailable()) return;
  try {
    const root = await navigator.storage.getDirectory();
    await root.removeEntry(RUN_FILE);
  } catch {
    /* nothing to clear */
  }
}
