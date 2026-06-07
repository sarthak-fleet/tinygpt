#!/usr/bin/env python3
"""Run Planner v7 held-out tool eval against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any


def chat(url: str, model: str, tools: list[dict[str, Any]], intent: str) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You have these tools available. Choose exactly one tool call.\n"
                    f"Tools:\n{json.dumps({'tools': tools}, ensure_ascii=False)}"
                ),
            },
            {"role": "user", "content": intent},
        ],
        "temperature": 0,
        "max_tokens": 180,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    data = json.loads(urllib.request.urlopen(req, timeout=180).read())
    content = data["choices"][0]["message"]["content"]
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object in response: {content!r}")
    return json.loads(content[start:end + 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765/v1/chat/completions")
    parser.add_argument("--model", default="tinygpt")
    parser.add_argument("--data", type=Path, default=Path("scripts/v7-eval/heldout-tools.jsonl"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    passed = 0
    for row in rows:
        try:
            got = chat(args.url, args.model, row["tools"], row["intent"])
            want = row["gold"]
            ok = got.get("verb") == want.get("verb")
            print(("[PASS]" if ok else "[FAIL]"), row["intent"])
            if not ok:
                print("  got :", got)
                print("  want:", want)
            passed += int(ok)
        except Exception as exc:
            print("[FAIL]", row["intent"])
            print("  error:", exc)
    print(f"\n=== {passed}/{len(rows)} held-out tool rows passed ===")


if __name__ == "__main__":
    main()
