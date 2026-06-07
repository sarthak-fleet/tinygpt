#!/usr/bin/env python3
"""pace-eval-cli.py — eval pace LoRA via `tinygpt hf-load --sample` (CLI,
no HTTP). Workaround for the serve-side crash on long generations
(separate bug).

Usage:
  python pace-eval-cli.py <hf-base-dir> <lora-path>
"""
import json
import re
import subprocess
import sys
from pathlib import Path

FIXTURES_DIR = Path("/Users/sarthak/Desktop/fleet/clickyLocal/evals/fixtures")
TINYGPT = "/Users/sarthak/Desktop/fleet/tinygpt/native-mac/.build/arm64-apple-macosx/release/tinygpt"


def evaluate_one(fx_path: Path, hf_dir: str, lora_path: str) -> dict:
    fx = json.loads(fx_path.read_text())
    # Build single-string prompt = system + "\n\nuser: " + user
    msgs = fx["request"]["messages"]
    parts = []
    for m in msgs:
        parts.append(f"{m['role']}: {m['content']}")
    prompt = "\n\n".join(parts) + "\n\nassistant:"
    max_tokens = min(int(fx["request"].get("max_tokens", 200)), 200)  # cap for safety

    expectations = fx.get("expectations", {})
    must = expectations.get("must_contain_patterns", [])
    must_not = expectations.get("must_not_contain_patterns", [])

    try:
        result = subprocess.run(
            [TINYGPT, "hf-load", hf_dir, "--lora", lora_path, "--sample",
             "--prompt", prompt, "--tokens", str(max_tokens), "--temperature", "0.0"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return {"name": fx_path.stem, "pass": False,
                    "failures": [f"CLI exit {result.returncode}"],
                    "content": result.stderr[:200]}
        # Output starts with banner; the generated text begins after the
        # last empty line before the tok/s footer. Easier: take everything
        # AFTER the prompt as printed in the output.
        full = result.stdout
        if prompt in full:
            content = full.split(prompt, 1)[1]
        else:
            content = full
        # Strip the trailing "(N tokens in ...)" footer
        content = re.split(r"\n\(\d+ tokens? in", content)[0]
    except subprocess.TimeoutExpired:
        return {"name": fx_path.stem, "pass": False, "failures": ["timeout"], "content": ""}
    except Exception as e:
        return {"name": fx_path.stem, "pass": False, "failures": [f"error: {e}"], "content": ""}

    failures = []
    for pat in must:
        if not re.search(pat, content):
            failures.append(f"missing: {pat}")
    for pat in must_not:
        if re.search(pat, content):
            failures.append(f"forbidden matched: {pat}")
    return {"name": fx_path.stem, "pass": len(failures) == 0,
            "failures": failures, "content": content[:300]}


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: pace-eval-cli.py <hf-base-dir> <lora-path>", file=sys.stderr)
        return 2
    hf_dir, lora = sys.argv[1], sys.argv[2]

    print(f"=== pace eval (CLI, lora={Path(lora).name}) ===\n")
    fixtures = sorted(FIXTURES_DIR.glob("*.json"))
    passed = 0
    for fx_path in fixtures:
        r = evaluate_one(fx_path, hf_dir, lora)
        status = "PASS" if r["pass"] else "FAIL"
        print(f"[{status}] {r['name']}")
        if not r["pass"]:
            for f in r["failures"]:
                print(f"    - {f}")
            print(f"    content: {r['content'][:200]}")
        else:
            passed += 1
        print()
    print(f"=== {passed}/{len(fixtures)} fixtures passed ===")
    return 0 if passed == len(fixtures) else 1


if __name__ == "__main__":
    sys.exit(main())
