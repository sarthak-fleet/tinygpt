---
name: VLM M4 architecture decision — Qwen3-VL port vs LLaVA fallback
status: decided-2026-06-08-option-A-full-Qwen3VL-port
owner: VLM elf (next session)
created: 2026-06-08
decided: 2026-06-08 — Option A (full Qwen3-VL port, base = UI-Venus-1.5-2B)
blocks: factory-vision-specialist.md M4 onwards
priority: P0 — DECIDED. VLM elf may proceed.
---

## DECISION (2026-06-08)

**Option A** — Port Qwen3-VL fully (mRoPE + image-token replacement + deepstack). Base: **UI-Venus-1.5-2B**.

**Why this over Option B (LLaVA fallback):**

1. Pace already daily-drives UI-Venus-1.5 successfully in LM Studio. We KNOW the ceiling — any LLaVA fallback that uses a non-UI-pretrained base would be a measurable regression from the model Pace currently uses. Falling back trades 2 weeks of arch work for visibly worse output. That trade fails.

2. Per `[[feedback_leverage_first]]`: leverage existing screen-reading pretraining (UI-Venus's whole reason to exist) instead of re-teaching it from raw VLM pretraining.

3. The "2 weeks" effort for mRoPE + image-token-replacement + deepstack is well-bounded engineering — three named features with clear references (HF's Qwen3-VL implementation is the parity reference). Not open-ended risk.

4. Owner's stated philosophy: "When you are going forward with the most ambitious goals in the world, then you have to take as much leverage as possible." Option A is the leverage-maximal path.

**Why not Option C (B-then-A):** Option B's wasted work (LLaVA-style v1 LoRA trained on a regressing base) is real. Once Pace tries v1 and reports quality drop, we'd throw the LoRA + SFT data away and do A's work anyway. C is only justified when v1's quality is unknown; here it's knowable (UI-Venus baseline already daily-driven). Skip the speculative ship.

**Implementation notes for the elf:**
- HF reference: <https://huggingface.co/inclusionAI/UI-Venus-Ground-2B> (architecture name `Qwen3VLForConditionalGeneration`)
- Parity gates: each of the three architectural features (mRoPE / token-replacement / deepstack) gets a per-feature parity test against HF PyTorch reference before composing into TinyGPTModelVLM
- Schedule: ~2 weeks added to original VLM PRD timeline (M4 expands; M5-M10 unchanged conceptually)
- New files expected: `MultimodalRoPE.swift`, `Qwen3VLForward.swift`, modifications to `TinyGPTModelVLM.swift` for token-replacement embedding

---

# Decision doc — VLM M4 architecture

## The situation

The VLM elf shipped M1+M2+M3 cleanly. At M3 (and after the
leverage-first gate added to the PRD mid-flight), the elf identified
that **UI-Venus-1.5-2B** is the best open-weights student base — not
the earlier Qwen3-VL-2B or UI-TARS-1.5B (which doesn't exist on HF).

UI-Venus-1.5-2B is `Qwen3VLForConditionalGeneration`, which has three
architectural features the original LLaVA-style PRD didn't account for:

1. **mRoPE** (multimodal RoPE) — RoPE positions split into 3 sections
   `[24, 20, 20]` for `(time, height, width)` axes. Image tokens get
   2D spatial positions; text tokens get scalar positions. Standard
   RoPE doesn't handle this.

2. **Image-token replacement at embed stage** — Qwen3-VL doesn't
   prepend image tokens before text (LLaVA convention). Instead, the
   text tokenizer emits placeholder `<image>` tokens, and the embedder
   substitutes vision features in-place. This means the VLM forward
   pass needs to know WHERE in the token sequence each image's
   features go.

3. **deepstack_visual_indexes=[5, 11, 17]** — visual features are
   re-injected at multiple LLM depths, not just front-loaded into the
   first layer. The elf would need to expose hooks at layers 5, 11,
   17 and route vision features into them.

All three are addressable but represent real engineering scope.

## Three options

### Option A — Port Qwen3-VL fully

Implement mRoPE, image-token replacement, and deepstack injection in
TinyGPTModelVLM.

**Effort**: ~2 weeks of focused work (M4 expands significantly).

**Pros**:
- Use UI-Venus-1.5-2B's pretrained weights — best leverage
- UI-Venus is already screen-reading-trained; least amount of our SFT
  needed to specialize
- Future-proofs us for Qwen3-VL family generally (the architecture is
  growing in popularity)

**Cons**:
- Significant new MLX-Swift code (three non-trivial features)
- Each feature has its own parity-gate vs HF PyTorch reference
- Pushes M5-M10 out by 2 weeks

### Option B — Pick a LLaVA-style base instead

Swap to a base that uses LLaVA's prepend-image-tokens convention
with standard RoPE. Candidates:
- Qwen2-VL-2B (older but compatible with our current PRD design)
- LLaVA-OneVision-0.5B / -7B (small, well-documented)
- MobileVLM-3B

**Effort**: ~3 days (PRD's current M4-M5 plan works as written).

**Pros**:
- Hit M4-M10 ship in original 4-week timeline
- No new architectural primitives — uses M1-M3 work as-is
- Simpler debugging surface

**Cons**:
- Lose UI-Venus's screen-reading pretraining
- Need MORE SFT data to compensate (M6 Stage A becomes larger, more
  reliance on Pace-specific AX data)
- Standard VLM might generalize less well to UI-specific patterns
  (icons, layouts, dense element grids)

### Option C — Hybrid: ship Option B first, port Qwen3-VL later

Ship LLaVA-style VLM with simpler base now (M4-M10 in original
timeline). When v1 is in Pace's hands and we've measured the quality
ceiling, port Qwen3-VL features incrementally if quality demands it.

**Effort**: ~3 days now, ~2 weeks later if needed.

**Pros**:
- Pace gets a VLM specialist in original timeline
- Defer Qwen3-VL eng cost until we know it's actually needed
- Architecture stays simple until complexity is justified

**Cons**:
- If Option B's quality ceiling is too low for daily-drive, we end up
  doing Option A's work anyway, having shipped a model that gets
  immediately deprecated
- Two LoRA artifacts to support (the LLaVA-style v1 + Qwen3-VL v2)

## Trade-off summary

| Axis | A (Qwen3-VL port) | B (LLaVA fallback) | C (B then A) |
|---|---|---|---|
| Pace ships VLM by | ~6-8 weeks | ~3-4 weeks | ~3-4 weeks (v1) + ~6-8 (v2) |
| Use UI-Venus pretrain | Yes (best leverage) | No | No (v1), maybe (v2) |
| Architectural complexity | High | Low | Low → high |
| Risk of quality miss | Lowest | Medium-high | Medium-high → low |
| Total eng cost if v1 is enough | 2 weeks | 3 days | 3 days |
| Total eng cost if v1 isn't enough | 2 weeks | 3 days + thrown work | 3 days + 2 weeks |

## My recommendation

**Option C, with a clear quality bar before deciding on v2 work.**
Ship the LLaVA-style v1 in 3-4 weeks with Qwen2-VL-2B (or similar)
as base. Define a clear quality bar: "VLM v1 passes ≥85% of held-out
Pace screen-reading scenarios." If v1 beats that, stop. If it
doesn't, do Option A's port work having confirmed it's worth the
weeks.

**Why this is the right Bayesian move**: the prior is unclear whether
LLaVA-style on Qwen2-VL is enough. Don't pay 2 weeks of arch work
speculatively. Pay 3 days to find out, then pay more only if needed.

**The risk to flag explicitly**: if Pace already daily-drives
UI-Venus-1.5-2B via LM Studio and it works well, then we KNOW the
ceiling is at UI-Venus quality. Falling back to Qwen2-VL gives up
that ceiling. So if owner reports "UI-Venus quality is great, anything
less would be a regression," go straight to Option A.

## What the next elf needs from the owner

A one-line decision: **A**, **B**, or **C**. Plus, if **C**: the
quality bar for "is v1 enough?"

## Files affected

- `docs/prds/factory-vision-specialist.md` — update M4 onwards to
  reflect chosen architecture
- New: `native-mac/Sources/TinyGPTModel/MultimodalRoPE.swift` (if A)
- New: `native-mac/Sources/TinyGPTModel/Qwen3VLForward.swift` (if A)
- M5/M7 SFT plan changes depending on chosen base
