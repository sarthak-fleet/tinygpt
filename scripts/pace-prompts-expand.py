#!/usr/bin/env python3
"""Expand the 44-prompt Pace seed into ~500 variants via teacher prompt mutation.

Hits the teacher endpoint and asks for N paraphrases per fixture, plus
adjacent-intent variants. Output is more raw prompts to feed back into
`tinygpt synthesize` for labeling.
"""
import json
import os
import requests
import sys
from pathlib import Path

SEED = Path.home() / ".cache" / "tinygpt" / "datasets" / "pace-prompts.jsonl"
OUT  = Path.home() / ".cache" / "tinygpt" / "datasets" / "pace-prompts-v2.jsonl"
TEACHER_URL = "http://127.0.0.1:1234/v1/chat/completions"
TEACHER_MODEL = "qwen/qwen3-30b-a3b"

EXPANSION_INSTRUCTION = """generate 12 paraphrases of the following user request, keeping the same intent and category but varying phrasing, formality, and word choice. output one paraphrase per line, no numbering, no extra text."""


def expand_one(prompt: str, category: str, n: int = 12) -> list[str]:
    payload = {
        "model": TEACHER_MODEL,
        "messages": [
            {"role": "system", "content": EXPANSION_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.9,
        "max_tokens": 600,
    }
    try:
        r = requests.post(TEACHER_URL, json=payload, timeout=120)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        lines = [ln.strip() for ln in text.split("\n") if ln.strip() and not ln.strip().startswith("#")]
        # Strip leading "1. " / "- " / "* " numbering if present
        cleaned = []
        for ln in lines:
            ln = ln.lstrip("0123456789. -*•").strip()
            if ln and len(ln) > 5:
                cleaned.append(ln)
        return cleaned[:n]
    except Exception as e:
        print(f"  ! expand failed for {prompt[:40]}: {e}", file=sys.stderr)
        return []


def main():
    if not SEED.exists():
        print(f"error: seed not found: {SEED}", file=sys.stderr)
        sys.exit(1)
    seeds = [json.loads(l) for l in SEED.open()]
    print(f"seed: {len(seeds)} prompts")

    out_rows: list[dict] = []
    # Keep originals
    for s in seeds:
        out_rows.append(s)
    # Expand each seed
    for i, s in enumerate(seeds):
        print(f"[{i+1}/{len(seeds)}] expanding [{s['category']}] {s['prompt'][:60]}...", flush=True)
        variants = expand_one(s["prompt"], s["category"], n=12)
        for v in variants:
            out_rows.append({"prompt": v, "category": s["category"], "source": "v2_expanded"})

    with OUT.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    by_cat: dict[str, int] = {}
    for r in out_rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print(f"\nwrote {len(out_rows)} prompts → {OUT}")
    print("by category:")
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat:30s} {count}")


if __name__ == "__main__":
    main()
