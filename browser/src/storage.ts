/**
 * storage.ts — checkpoint persistence in the browser (Phase 4).
 *
 * STATUS: documented stub. No implementation yet.
 *
 * Persist checkpoints to OPFS (origin-private file system) or IndexedDB so a
 * training run survives a page refresh.
 *
 * Checkpoint contents (see docs/browser_notes.md "Checkpointing"):
 *   model_config.json, training_config.json, dataset_manifest.json,
 *   trainer_state.json, weights.f32, adam_m.f32, adam_v.f32, loss_history.json
 *
 * OPFS is subject to browser storage quotas; clearing site storage deletes it.
 * Request durability up front:
 *   await navigator.storage.persist();
 *   const estimate = await navigator.storage.estimate();
 *
 * Guide: docs/browser_notes.md ("Checkpointing")
 *
 * TODO(phase-4): saveCheckpoint() / loadCheckpoint() / listCheckpoints().
 * TODO(phase-4): request persistent storage; surface quota to the UI.
 */
export {};
