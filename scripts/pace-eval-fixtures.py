#!/usr/bin/env python3
"""pace-eval-fixtures.py — run clickyLocal/evals/fixtures/*.json against
an OpenAI-compat endpoint (e.g. our tinygpt serve --lora) and report
pass/fail per fixture.

Pass = response matches all `must_contain_patterns` AND none of the
`must_not_contain_patterns`. Same logic Pace's own pipeline applies.

Usage:
  python pace-eval-fixtures.py [base-url] [model-id]
  defaults: http://127.0.0.1:8765/v1/chat/completions, pace-planner-v2
"""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

FIXTURES_DIR = Path("/Users/sarthak/Desktop/fleet/clickyLocal/evals/fixtures")


def request(url: str, body: dict) -> str:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def evaluate_one(fx_path: Path, url: str, model_id: str) -> dict:
    fx = json.loads(fx_path.read_text())
    body = dict(fx["request"])
    body["model"] = model_id
    body["stream"] = False
    expectations = fx.get("expectations", {})
    must = expectations.get("must_contain_patterns", [])
    must_not = expectations.get("must_not_contain_patterns", [])

    try:
        content = request(url, body)
    except Exception as e:
        return {"name": fx_path.stem, "pass": False, "failures": [f"request error: {e}"],
                "content": ""}

    failures = []
    for pat in must:
        if not re.search(pat, content):
            failures.append(f"missing required pattern: {pat}")
    for pat in must_not:
        if re.search(pat, content):
            failures.append(f"matched forbidden pattern: {pat}")

    return {
        "name": fx_path.stem,
        "pass": len(failures) == 0,
        "failures": failures,
        "content": content[:400],
    }


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/v1/chat/completions"
    model_id = sys.argv[2] if len(sys.argv) > 2 else "tinygpt"

    print(f"=== pace eval ({url}, model={model_id}) ===\n")
    fixtures = sorted(FIXTURES_DIR.glob("*.json"))
    results = []
    passed = 0
    for fx_path in fixtures:
        r = evaluate_one(fx_path, url, model_id)
        results.append(r)
        status = "PASS" if r["pass"] else "FAIL"
        print(f"[{status}] {r['name']}")
        if not r["pass"]:
            for f in r["failures"]:
                print(f"    - {f}")
            print(f"    content: {r['content'][:160]}…")
        else:
            passed += 1

    print(f"\n=== {passed}/{len(results)} fixtures passed ===")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
