# Quality Bundle: tests, lint, crash recovery

Three loose pieces of the build/test infrastructure shipped together so
the next agent inherits a CI floor that catches the obvious regressions
before they reach `main`.

## 1. XCTest coverage extension

The existing tests covered file-format round-trip and a sanity check on
`ModelConfig.huge`. The bundle extends both test targets to lock in the
schemas and architectural invariants that have been changing fastest.

### `TinyGPTIOTests` (now 19 tests, was 12)

- `test_configFields_roundTripBPEMetadata` — `vocabSize`,
  `tokenizerSource` round-trip cleanly.
- `test_configFields_roundTripMoEMetadata` — `nExperts`, `moeTopK`,
  `loadBalanceWeight` round-trip.
- `test_configFields_roundTripArchitectureFlags` — `slidingWindow`,
  `useMoD`, `useDifferentialAttention`, `useYOCO`, `useGradCheckpoint`
  round-trip.
- `test_configFields_omitNilFieldsFromJSON` — pins the
  "nil → no key in JSON" behaviour. Critical because pre-YOCO and
  pre-MoE readers don't know those keys; the encoder must keep them
  absent rather than emitting `"useYOCO": null`.
- `test_headerRoundTrips_withFullSchema` — end-to-end encode/decode
  with every current optional populated.
- `test_v1File_withoutManifestReportsMissingManifest` — v1 files
  predate the manifest field; the reader must surface the dedicated
  `missingManifest` error so the CLI can print the upgrade hint
  instead of a generic decode error.
- `test_v1File_versionIsAcceptedInTheSupportedSet` — defensive: if
  someone drops v1 from `supportedVersions`, the migration story
  gets reviewed first.

### `TinyGPTModelTests` (now 14 tests, was 2)

- `test_modelConfigDefaults_areTinyByteLevel` — every default field
  pinned so a "bump the default" change is deliberate.
- `test_modelConfig_nKvHeadsDefaultsToNHeads` /
  `test_modelConfig_gqaWithFewerKVHeads` — grouped-query attention
  invariants.
- `test_forwardIsDeterministicUnderFixedSeed` — two models seeded
  identically must produce bit-identical fp32 logits for the same
  input. Catches any "introduces non-determinism through default
  RNG capture" regression.
- `test_forwardShapesMatchVocabAndContext` — surface-level shape
  smoke test.
- `test_kvCacheMatchesUncachedForwardOnPrefill` — cached prefill
  produces the same logits (within fp16 SDPA tolerance) as the
  full forward. This is the load-bearing invariant for streaming
  sampling; if the cache and full path diverge, the model "lies"
  during decode.
- `test_kvCacheGrowsAcrossSteps` — the cache appends correctly
  across multi-step decode.
- `test_loraInjection_atInitDoesNotChangeForward` — LoraLinear at
  `B = 0` is a pure no-op; injecting LoRA on a trained model
  doesn't change its outputs at init.
- `test_loraInjection_trainableParamCount` — pins the param-count
  arithmetic against shape drift.
- `test_modelLoader_recognisesTinygptFile` — auto-detection picks
  the from-scratch path for a `.tinygpt` file.
- `test_modelLoader_directoryWithoutConfigJSONFails` — directory
  without `config.json` produces a clean error.

### CI wiring

`.github/workflows/ci.yml` already runs `xcodebuild test -scheme
TinyGPT-Package` on every push / PR via the `mac` job. The new tests
piggy-back on that runner — no new GitHub Actions cost. The job runs
in ~10-15 minutes on `macos-15`.

## 2. SwiftFormat config + CI lint

### Rationale: opt-in via `--rules`, not opt-out via `--disable`

The codebase has its own style — column-aligned trailing comments,
8-space continuation indent in MLX argument lists, math operators
written compact (`a*b + c*d`). Running swiftformat with its defaults
flagged **775 issues across 75 of 76 files**. That's too much churn
to land as a "lint cleanup" pass alongside three other things.

So `.swiftformat` enables ONLY the 25 rules the codebase already
conforms to — duplicate-import detection, semicolon stripping,
`Void` consistency, `isEmpty` over `count == 0`, `headerFileName`,
the various `redundantX` rules — and leaves the layout rules
(`indent`, `wrapArguments`, `spaceAroundOperators`,
`hoistPatternLet`, etc.) disabled. The CI lint will catch a
**new** file that uses `count == 0` or introduces a duplicate
`import`, but won't churn-fight the existing files. Once the
codebase grows out of its current style, the rule set can be
expanded incrementally.

### CI job

`.github/workflows/ci.yml` gains a `swift-quality` job:

```yaml
swift-quality:
  runs-on: macos-15
  steps:
    - uses: actions/checkout@v4
    - run: brew install swiftformat
    - run: swiftformat --lint native-mac/Sources native-mac/Tests
```

Runs in parallel with the `mac` job; ~1 minute end-to-end. Fails on
the first violation (`--lint` exits 1 on drift). A developer hitting
this can run `swiftformat native-mac/Sources native-mac/Tests` (no
`--lint`) locally to auto-fix.

### Pre-existing warnings fixed

- `native-mac/Sources/TinyGPT/Finetune.swift:52` — `var url` →
  `let url` (variable never mutated).
- `native-mac/Sources/TinyGPT/ES.swift:207/216/247` — three
  `try? model.update(...)` calls had unused optional results;
  changed to `_ = try? ...` to make the discard explicit.

After these fixes, `xcodebuild build` is warning-free for the Mac
targets.

## 3. Crash-recovery tests

Three scenarios that all hit the `--resume` / atomic-save code paths
the long-running mac training depends on.

### Test 1 — in-process: 50 steps vs 25 + save/reload + 25

`test_crashRecovery_inProcess_50vs25Plus25` in
`TinyGPTModelTests.swift`.

Trains a tiny `ModelConfig` (2 layers, d=16, vocab=16) on a
deterministic 256-byte cyclic corpus. Path A runs 50 contiguous
steps. Path B runs 25 steps, writes a `.tinygpt` checkpoint,
constructs a fresh `TinyGPTModel`, loads the checkpoint via
`TinyGPTWeightLoader.load`, then runs 25 more steps. Both paths
share a **deterministic batch generator** (`makeDeterministicBatches`)
so the random-window draws are identical.

The test compares every parameter element-wise and asserts a per-
element RMS difference under `0.05`. AdamW state restarts on resume
(MLX-Swift doesn't expose the optimizer's internal `m`, `v`), so
the second halves diverge slightly — the tolerance is calibrated to
let that drift through while catching the bugs that matter (missing
manifest entry, swapped param order, dropped architecture flag).

Runs in ~4 seconds.

### Test 2 — subprocess: kill mid-train, resume, compare losses

`test_subprocessCrashRecovery_resumeMatchesContiguousFinalLoss` in
`CrashRecoverySubprocessTests.swift`.

Spawns the real `tinygpt` CLI via `Foundation.Process`:

1. **Contiguous** run: `tinygpt train --preset tiny --steps 100`,
   capture the final loss from `step N/T loss F.fff` log line.
2. **Interrupted** run: `tinygpt train --preset tiny --steps 100
   --save-every 25`. Poll for the checkpoint file. When it
   appears (≥ step 25), SIGKILL the process.
3. **Resumed** run: `tinygpt train --resume <ckpt> --steps 100`.
4. Assert `|contig_loss − resume_loss| < 0.5` (absolute units;
   the corpus is short so both runs converge fast).

Binary discovery: `TINYGPT_BIN` env var first, then walk up from
the test bundle's directory looking for `Build/Products/Debug/tinygpt`.
Skips gracefully (`XCTSkip`) when neither resolves.

Runs in ~6 seconds.

### Test 3 — atomic write: SIGTERM race against save-every

`test_atomicWrite_leavesOnlyCompleteOrPreviousCheckpointOnDisk` in
`CrashRecoverySubprocessTests.swift`.

Spawns `tinygpt train --save-every 5`, polls for the first
checkpoint to land, then `terminate()`s the process. Once the
process is dead, the test asserts: the file at the target path is
either **absent** or **decodes cleanly via `TinyGPTFileReader.read`**.
The atomic-save path writes to `<path>.tmp` and renames; if SIGTERM
hits between write-and-rename, the target path is unchanged. A stray
`.tmp` sidecar is allowed (next run's atomicSave overwrites it).

Runs in ~4 seconds.

## Test inventory totals

|                                  | before | after |
| -------------------------------- | -----: | ----: |
| `TinyGPTIOTests`                 |     12 |    19 |
| `TinyGPTModelTests`              |      2 |    14 |
| `CrashRecoverySubprocessTests`   |      0 |     2 |
| `TinyGPTServeTests` (unchanged)  |      5 |     5 |
| **Total**                        |     19 |    40 |

All 40 pass on `macos-15` with `xcodebuild test -scheme
TinyGPT-Package -destination "platform=macOS"`. Total runtime: ~30s.
