---
name: B31 unified model gallery + project-level model pins
status: scaffolding-shipped-2026-06-13 (browser gallery extension + CLI pull pending)
owner: unassigned
created: 2026-06-13
parent_plan: docs/PLAN.md §3 Tier B (B31)
related_prds: A1-first-specialist-tool-caller.md (the first specialist to publish), B6-mac-app-demo.md (Factory tab consumes the gallery), B25-scaledown-specialist.md, B8-multilingual-specialist.md (siblings that need to ship via the gallery)
related_existing: browser/src/gallery-schema.ts (the browser-side schema we extend), docs/gallery_v2_plan.md (v1.5 + v2 roadmap to fold into this)
---

# PRD — Unify the gallery (browser + Mac) + project-level model pins

## Goal

Two surfaces:

1. **Unified gallery manifest** — extend the existing
   `browser/public/gallery/manifest.json` so it lists not just the
   browser-loadable from-scratch `.bin` models but also the
   *Mac-side* specialists (LoRA + DoRA adapters, GGUF base bundles,
   .tinygpt full models, R2-hosted artifacts). One published JSON,
   two clients (browser playground + `tinygpt` CLI), no parallel
   registries.

2. **Project-level model pins** — a `tinygpt.project.json` per-
   project manifest declaring which models the project depends on
   (analogous to `package.json` / `requirements.txt`). Running
   `tinygpt pull` from a project dir fetches everything; `tinygpt
   validate` checks every pin resolves; the Mac app's Factory tab
   (B6) reads pins to populate its model picker.

Together these make the "ship a specialist, ship a project" story
real. Without them, A1/B1/B25/B8 each ship as bespoke download
URLs in the docs — doesn't scale past 3 specialists.

## Why now

- A1 lands a Mac-runnable specialist that users will want to
  pull. Without a gallery entry, the shipping story is "URL in
  the PR description."
- Browser gallery (`gallery-schema.ts`) + Mac CloudPush/Pull
  already exist and overlap in concept but not in schema. Now
  is the cheap moment to unify.
- B6 Mac app (Factory tab) needs a populated model picker.
  The gallery manifest is its data source.

### The trace-loop dividend (added 2026-06-13)

The thing Castform has that we don't is *extensive production
traces* — they observe their customers' agent traffic at the SaaS
boundary, which feeds their dataset-synthesis loop (filtered into
training data → better next iteration). We don't have a SaaS;
we've been on the wrong end of that asymmetry.

Project-level pins flip it: when a project pins models from this
gallery AND runs them through our serve, **we (or the project
owner) own the trace stream by construction**. B22 already
ships per-rollout `.atraj` files at the serve layer; B29 turns
those into training data. The missing piece was *which model the
project is running* — and the project file gives us exactly that
in machine-readable form.

That makes B31 the architectural pivot: it's the file that ties
together gallery (substrate distribution) + traces (B22 substrate
capture) + trace-to-data (B29 substrate refinement) + composite
reward (B28 substrate scoring). The trace dividend isn't a
hypothetical future "if users adopt"; it's free the first time a
project file ships with a non-trivial `models` list.

## Scope — in

### Unified manifest (extends the browser schema)

- Browser-side `browser/src/gallery-schema.ts` gains an optional
  `kind` discriminator: `"browser-bin" | "mac-tinygpt" |
  "mac-adapter" | "mac-gguf" | "mac-safetensors-hf"`. Existing
  rows default to `"browser-bin"` so nothing breaks.
- New optional fields on `GalleryModel`:
  - `kind: GalleryModelKind`
  - `parent`: for adapters, the gallery id of the base model
    (e.g. `a1-tool-caller` has `parent: "qwen3-4b-instruct-2507"`)
  - `r2_path`: relative path under the R2 bucket for Mac
    artifacts that don't live in `browser/public/gallery/`
  - `tags`: `["tool-call", "english", "lora-adapter", ...]`
  - `benchmarks_extended`: composite-reward block per B28 +
    per-suite breakdowns (already partially covered by
    `benchmarks: Record<string, number>` — we extend with named
    dimensions when present)
- Swift mirror: `native-mac/Sources/TinyGPTModel/GalleryManifest.swift` —
  the same Codable shape, single source of truth for parsing.
  Generated from `gallery-schema.ts` by hand on schema edits
  (no codegen tooling for V1).

### Project pins

- New file format: `tinygpt.project.json` at repo root of a
  consumer project. JSON (not TOML — TS-first codebase, no
  TOML readers shipped). Schema:
  ```json
  {
    "name": "my-pace",
    "tinygpt_version": ">=2026.06",
    "models": [
      {"id": "qwen3-4b-instruct-2507", "role": "base"},
      {"id": "a1-tool-caller", "role": "adapter",
       "applies_to": "qwen3-4b-instruct-2507"}
    ],
    "datasets": [
      {"id": "hermes-function-calling-v1", "optional": false}
    ]
  }
  ```
- New CLI: `tinygpt pull` resolves the project file + the gallery
  manifest + the R2 bucket, fetches everything to
  `~/.cache/tinygpt/`, prints what's missing.
- New CLI: `tinygpt validate` checks every pin in
  `tinygpt.project.json` exists in the published gallery; flags
  unknown ids before you ship.

### What ships in *this* PR (the scaffolding)

| File | Content |
|---|---|
| `native-mac/Sources/TinyGPTModel/GalleryManifest.swift` | new — Codable types matching gallery-schema.ts; reader for a manifest JSON file. |
| `native-mac/Sources/TinyGPTModel/ProjectManifest.swift` | new — Codable types for `tinygpt.project.json`. |
| `native-mac/Tests/TinyGPTModelTests/GalleryManifestTests.swift` | new — parse the shipped browser gallery manifest; round-trip; reject malformed shapes. |
| `native-mac/Tests/TinyGPTModelTests/ProjectManifestTests.swift` | new — parse a fixture project file; reject schema violations. |
| `examples/tinygpt.project.json` | new — fixture example a consumer project can copy. |
| `docs/prds/B31-gallery-and-project-pins.md` | this file. |

### What's deferred (PRD captures, not shipped this PR)

- Browser-side gallery UI changes (filter by `kind`, show
  adapters alongside bases). Browser team consumes the schema
  extension when ready.
- `tinygpt pull` CLI extension — extends the existing
  `Sources/TinyGPT/CloudPull.swift`; needs the published gallery
  on R2 to test against. Spec is in this PRD.
- `tinygpt validate` CLI — same.
- Publishing `a1-tool-caller` as the first new gallery row — gated
  on A1 actually shipping.

## Scope — out

- **HF Hub integration** as the source of truth. R2 is faster (no
  egress) and we already own the pipeline.
- **Multi-version pinning** with semver resolution. V1 is
  exact-id pinning. Versions come later if churn becomes a problem.
- **Lockfile** (`tinygpt.project.lock.json`). Defer; until we
  have versioning, lockfile content is the same as the pins.
- **OCI registry-style content addressing.** Big design surface;
  defer.

## Acceptance criteria

### Scaffolding (this PR)

- [x] `GalleryManifest.swift` parses
  `browser/public/gallery/manifest.json` without error (existing
  rows; new `kind` field defaults to `browser-bin`).
- [x] `ProjectManifest.swift` parses `examples/tinygpt.project.json`
  and rejects a malformed fixture.
- [x] Both Swift modules ship with passing unit tests.

### Full B31 ship (remaining)

- [ ] One Mac-side specialist (A1 once shipped) added to the
  gallery manifest with `kind: "mac-adapter"`, `parent:`, `r2_path:`.
- [ ] `tinygpt pull` reads `./tinygpt.project.json`, fetches every
  pin from R2, validates checksums (the `fileBytes` field from
  the manifest), reports skipped + failed.
- [ ] `tinygpt validate` flags unknown ids.
- [ ] Browser playground filters out `kind != browser-bin` rows
  from its dropdown (so it stays focused on the in-browser
  models).
- [ ] B6 Mac app Factory tab reads the gallery + project files
  for its model picker.

## Reference patterns

- `browser/src/gallery-schema.ts` — the existing schema; extend,
  don't fork.
- `native-mac/Sources/TinyGPT/CloudPush.swift` /
  `CloudPull.swift` / `CloudList.swift` — the R2 plumbing; `pull`
  extends Pull, `validate` is a new sibling.
- `docs/gallery_v2_plan.md` — the v1.5 + v2 roadmap; this PRD
  supersedes its "v2 hosting" section since R2 is already shipped
  and the new question is "what about Mac models?"

## Open questions

- Whether the project file lives in `./tinygpt.project.json` (at
  project root) or `./.tinygpt/project.json` (hidden, package-
  manager convention). **Recommendation:** visible — users should
  be able to read it without `ls -la`. Convention follows
  `package.json`.
- Whether to mirror the schema in YAML for friendlier diffs.
  **Recommendation:** no — keep one format; JSON is what
  `gallery-schema.ts` already speaks.

## Sibling note: mascot

Castform's brand-ambassador cartoon prompted a related question:
should TinyGPT have one? **Yes** — pace-the-companion is the
natural Pace caricature; "tinygpt-the-trainer" is the distinct
character for the platform/toolkit side. Filed in
`docs/decision_log.md` as a marketing/identity item (not a feature
PRD).
