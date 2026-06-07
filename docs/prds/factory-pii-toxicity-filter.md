---
name: PII + toxicity filter — data-cleaning gap
status: shipped-2026-06-07
owner: unassigned
created: 2026-06-07
priority: P2
---

# PRD — `tinygpt filter` for PII + toxicity

## 2026-06-07 ship note

`tinygpt filter` now ships a dependency-free v1:

- JSONL/text input via `--in` and cleaned output via `--out`
- regex PII redaction for email, phone, SSN, IP, and credit cards
- optional `--drop-pii`
- optional `--sidecar` audit JSONL with redacted text
- heuristic toxicity scoring via `--toxicity`

Detoxify is **not bundled** in v1; `--toxicity-model` accepts `heuristic` or
`off` and rejects unavailable external model names.

## Goal

Add a PII + toxicity filter to the data-prep pipeline. Currently we have
dedupe (`tinygpt dedupe`) + quality classifier (`tinygpt quality-filter`)
but no PII or toxicity removal.

## Scope — in

### CLI

```
tinygpt filter \
    --in raw.jsonl \
    --out cleaned.jsonl \
    --pii email,phone,ssn,ip,credit-card \         # built-in PII types
    --toxicity 0.8 \                                # drop above threshold
    --toxicity-model detoxify-en                    # local model
```

### PII detection

Regex-based + optional model-based (Microsoft Presidio patterns).
Built-in types:
- Email
- Phone (US + international)
- SSN
- IPv4 / IPv6
- Credit cards
- Names (heuristic only; flag for review, don't auto-drop)

### Toxicity

Bundle a small classifier (Detoxify English, ~150 MB). Threshold-based
drop or replace.

### Output

For each filtered row, optionally write to a sidecar `filtered.jsonl`
with the reason — for audit + recovery.

## Acceptance

1. Smoke: feed a small JSONL with seeded PII; verify all redacted
2. Smoke: feed known toxic text; verify dropped above threshold
3. Pass-through verifies clean rows are unchanged

## File paths

| Action | Path |
|---|---|
| **create** | `native-mac/Sources/TinyGPT/Filter.swift` |
| **create** | regex patterns for PII (built-in) |

## Estimated effort

**~1-2 days.** Regex PII is trivial; toxicity model bundling adds ~half
a day.

## Source

- Microsoft Presidio: https://github.com/microsoft/presidio
- Detoxify: https://github.com/unitaryai/detoxify
