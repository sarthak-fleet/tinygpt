#!/usr/bin/env python3
"""Build Planner v7 held-out tool-call eval rows.

Rows use tool names that can be withheld from training but shown in the prompt
at eval time. Correct behavior is to use the schema in the prompt, not memorize
the verb/tool name from SFT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROWS = [
    {
        "intent": "drag the file into the upload box",
        "tools": [{"type": "function", "function": {"name": "drag", "description": "Drag one visible UI target to another.", "parameters": {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]}}}],
        "gold": {"verb": "drag", "args": {"source": "file", "destination": "upload box"}, "spoken_text": "dragging"}
    },
    {
        "intent": "take a screenshot of this window",
        "tools": [{"type": "function", "function": {"name": "capture_screen", "description": "Capture a screenshot.", "parameters": {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]}}}],
        "gold": {"verb": "capture_screen", "args": {"target": "current window"}, "spoken_text": "capturing"}
    },
    {
        "intent": "mute this tab",
        "tools": [{"type": "function", "function": {"name": "toggle_tab_audio", "description": "Mute or unmute a browser tab.", "parameters": {"type": "object", "properties": {"target": {"type": "string"}, "muted": {"type": "boolean"}}, "required": ["target", "muted"]}}}],
        "gold": {"verb": "toggle_tab_audio", "args": {"target": "current tab", "muted": True}, "spoken_text": "muting"}
    },
    {
        "intent": "summarize the selected text into three bullets",
        "tools": [{"type": "function", "function": {"name": "transform_selection", "description": "Transform selected text.", "parameters": {"type": "object", "properties": {"operation": {"type": "string"}, "format": {"type": "string"}}, "required": ["operation"]}}}],
        "gold": {"verb": "transform_selection", "args": {"operation": "summarize", "format": "three bullets"}, "spoken_text": "summarizing"}
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("scripts/v7-eval/heldout-tools.jsonl"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for row in ROWS:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(ROWS)} held-out rows to {args.out}")


if __name__ == "__main__":
    main()
