#!/usr/bin/env python3
"""build-v11-corpus.py — merge v10 happy-path + v11 seeds + v11 amplified
into the final v11 training JSONL.

Inputs (any missing optional input is skipped with a warning):
  ~/.cache/tinygpt/datasets/pace-v10-multiplied.jsonl   (404 rows, required)
  ~/.cache/tinygpt/datasets/pace-v11-seed.jsonl         (93 rows, required)
  ~/.cache/tinygpt/datasets/pace-v11-amplified.jsonl    (optional — output of v11-amplify.py)

Output:
  ~/.cache/tinygpt/datasets/pace-v11-train.jsonl

Steps:
  1. Load all sources; strip _meta keys (training shape is {instruction, response} only)
  2. Dedup on normalized instruction text
  3. Validate every response parses as JSON with a known v11 intent
  4. Deterministic shuffle (seed=11) so class examples interleave
  5. Report class distribution
"""
import json
import random
from pathlib import Path

DS = Path.home() / ".cache/tinygpt/datasets"
V10 = DS / "pace-v10-multiplied.jsonl"
SEED = DS / "pace-v11-seed.jsonl"
AMP = DS / "pace-v11-amplified.jsonl"
OUT = DS / "pace-v11-train.jsonl"

KNOWN_INTENTS = {"action", "answer", "dictate", "edit",
                 "out_of_scope", "clarify", "confirm_destructive"}


def load(p: Path, required: bool) -> list[dict]:
    if not p.exists():
        if required:
            raise SystemExit(f"ERROR: required input missing: {p}")
        print(f"  (optional input missing, skipping: {p.name})")
        return []
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    print(f"  loaded {len(rows):4d} rows from {p.name}")
    return rows


def main():
    print("=== build-v11-corpus ===")
    rows = load(V10, required=True) + load(SEED, required=True) + load(AMP, required=False)

    # strip _meta, validate, dedup
    seen: set[str] = set()
    out_rows: list[dict] = []
    bad_json = 0
    bad_intent = 0
    dups = 0
    by_intent: dict[str, int] = {}

    for r in rows:
        inst = r.get("instruction", "").strip()
        resp = r.get("response", "")
        if not inst or not resp:
            bad_json += 1
            continue
        key = " ".join(inst.lower().split())
        if key in seen:
            dups += 1
            continue
        try:
            inner = json.loads(resp) if isinstance(resp, str) else resp
        except json.JSONDecodeError:
            bad_json += 1
            continue
        intent = inner.get("intent")
        if intent not in KNOWN_INTENTS:
            bad_intent += 1
            continue
        seen.add(key)
        out_rows.append({"instruction": inst,
                         "response": json.dumps(inner, ensure_ascii=False)})
        by_intent[intent] = by_intent.get(intent, 0) + 1

    random.Random(11).shuffle(out_rows)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n")

    print(f"\n  kept {len(out_rows)} rows → {OUT}")
    print(f"  dropped: {dups} dups, {bad_json} bad-json, {bad_intent} unknown-intent")
    print("  class distribution:")
    for c, n in sorted(by_intent.items(), key=lambda kv: -kv[1]):
        print(f"    {c:24s} {n}")


if __name__ == "__main__":
    main()
