---
name: Domain-adapt SFT mode — `--domain-adapt` flag
status: shipped-2026-06-07
owner: unassigned
created: 2026-06-07
priority: P3 — small but completes the recipe taxonomy
---

# PRD — `tinygpt train --domain-adapt` mode

## 2026-06-07 ship note

`tinygpt train` now supports:

- `--domain-adapt`
- `--base <checkpoint.tinygpt>` as an alias for `--resume`
- conservative domain-adapt defaults when the user has not explicitly set them:
  `--lr-schedule wsd`, `--warmup 100`, `--max-lr 1e-4`, `--min-lr 1e-5`,
  `--decay-steps 5% of --steps`, and `--lr-layer-decay 0.85`

The CLI help and no-base validation path were smoke-tested. A real 100-step
training smoke was not run from this agent context because project rules require
approval before starting training workloads.

## Goal

Add a `--domain-adapt` flag to `tinygpt train` that does continued
pretraining on a domain corpus with appropriate defaults. Today it's
possible via manual `--resume` + custom flags but the recipe isn't
first-class.

## Why ship

Honest recipe parity. The factory has explicit modes for:
- Pretrain (from scratch)
- SFT (instruct tune)
- DPO (preference)
- Distill (teacher→student)

But "continued pretrain on domain text" — taking a base, training it on
a focused corpus to sharpen domain knowledge before SFT — is a real
intermediate step that deserves its own surface.

## Scope — in

### CLI

```
tinygpt train \
    --base qwen3-0.6b.tinygpt \
    --corpus medical-text.txt \
    --domain-adapt \                            # NEW: opinionated defaults
    --steps 5000 \                              # smaller than pretrain
    --max-lr 1e-4 \                             # smaller than fresh pretrain (3e-4)
    --warmup 100 \                              # tiny
    --out qwen3-medical.tinygpt
```

The `--domain-adapt` flag sets defaults that differ from from-scratch
pretrain:
- LR cap lower (catastrophic forgetting risk)
- Shorter warmup
- WSD decay window shorter
- Forces `--resume`-equivalent (must have a base)

### Behavior

Same training loop. Just a different default profile. Documents which
hyperparameters are conservative for domain-adapt and why.

## Scope — out

- LoRA-only domain adapt (use `tinygpt sft --lora` for that)
- Adapter fusion across multiple domains

## Acceptance

1. `tinygpt train --domain-adapt --base X.tinygpt --corpus Y.txt --steps 100 --out Z.tinygpt` works
2. `tinygpt train --help` documents the flag + the default deltas
3. Smoke test: 100-step run shows loss going down (sanity check)

## File paths

| Action | Path |
|---|---|
| **modify** | `native-mac/Sources/TinyGPT/Train.swift` — flag + default overrides |

## Estimated effort

**~1 day.** Lightest of the bunch.
