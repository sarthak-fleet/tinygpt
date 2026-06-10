#!/usr/bin/env python3
"""Convert PACE-dataset-capture samples into fm-vlm eval fixtures.

Reads manifest.jsonl from ~/Desktop/PACE-dataset-capture (see its
docs/label-schema.md) and emits one fixture .txt per sample in the
fm-vlm-fixtures format consumed by eval_pace_vlm_ab.py and
fake_pace_vlm.py — including SCREENSHOT_PATH so the A/B runner attaches
the image.

Mapping:
  app            → APP_FRONTMOST + EXPECT_APP
  ui_elements    → AX_TREE rows (id = index, pos = bbox center)
  user_intent    → USER
  next_action    → click/read expectations:
                     type=click → SPOKEN_MUST_CONTAIN: <target>
                     type=read  → SPOKEN_MUST_CONTAIN: <text> (if given)
  screenshot     → SCREENSHOT_PATH (absolute)

Privacy: rows with redaction_needed=true are skipped unless a redacted
image exists at redacted/<id>.png (then that image is used).

Usage:
  python3 scripts/vlm-captures-to-fixtures.py \
    --capture-dir ~/Desktop/PACE-dataset-capture \
    --out ../pace/evals/fm-vlm-fixtures-shots-v1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def slugify(s: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:max_len] or "sample"


def element_rows(ui_elements: list[dict]) -> list[str]:
    rows = []
    for i, el in enumerate(ui_elements):
        bbox = el.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
            pos = f"{cx},{cy}"
        else:
            pos = "0,0"
        role = el.get("type", "element")
        label = el.get("label", "")
        rows.append(f"  [{i}] {role}|{pos}|{label}|")
    return rows


def convert(sample: dict, capture_dir: Path) -> tuple[str, str] | None:
    sid = sample.get("id", "")
    privacy = sample.get("privacy", {})
    shot = capture_dir / sample.get("screenshot", f"screenshots/{sid}.png")
    if privacy.get("redaction_needed"):
        redacted = capture_dir / "redacted" / f"{sid}.png"
        if not redacted.exists():
            return None  # skip — unredacted sensitive screenshot
        shot = redacted
    if not shot.exists():
        return None

    user = sample.get("user_intent") or sample.get("task") or ""
    if not user:
        return None

    lines = [f"USER: {user}",
             f"APP_FRONTMOST: {sample.get('app', '')}",
             f"SCREENSHOT_PATH: {shot.resolve()}"]
    elements = sample.get("ui_elements") or []
    if elements:
        lines.append("AX_TREE:")
        lines.extend(element_rows(elements))
    else:
        lines.append("AX_BLIND: true")

    expects = []
    if sample.get("app"):
        expects.append(f"EXPECT_APP: {sample['app']}")
    action = sample.get("next_action") or {}
    a_type, target = action.get("type"), (action.get("target") or "").strip()
    if a_type == "click" and target:
        expects.append(f"SPOKEN_MUST_CONTAIN: {target}")
    elif a_type == "read" and action.get("text"):
        expects.append(f"SPOKEN_MUST_CONTAIN: {action['text']}")
    expects.append("SPOKEN_MUST_NOT_CONTAIN: ID")
    lines.append("")
    lines.extend(expects)

    name = f"shot-{slugify(sample.get('task') or user)}-{sid[-6:]}"
    return name, "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--capture-dir", type=Path,
                   default=Path.home() / "Desktop/PACE-dataset-capture")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    manifest = args.capture_dir / "manifest.jsonl"
    if not manifest.exists():
        print(f"no manifest at {manifest} — capture some samples first "
              f"(python3 scripts/capture.py capture)", file=sys.stderr)
        return 2

    written = skipped = 0
    args.out.mkdir(parents=True, exist_ok=True)
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        result = convert(sample, args.capture_dir)
        if result is None:
            skipped += 1
            continue
        name, text = result
        if args.dry_run:
            print(f"would write {name}.txt")
        else:
            (args.out / f"{name}.txt").write_text(text)
        written += 1

    print(f"fixtures written: {written}, skipped (privacy/missing): {skipped}")
    print(f"out: {args.out}")
    if skipped:
        print("note: redaction_needed rows are only included once a "
              "redacted/<id>.png exists")
    return 0


if __name__ == "__main__":
    sys.exit(main())
