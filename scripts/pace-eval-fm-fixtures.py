#!/usr/bin/env python3
"""pace-eval-fm-fixtures.py — production-format eval against clickyLocal's
fm-fixtures using CLI hf-load --sample (serve crash workaround).

Each fm-fixture has:
- USER: user's voice command
- ELEMENT: [N] role|x,y|label|text   (zero or more)
- FREE_TEXT_MODE: true|false
- EXPECT_POINT_ID: <int>    or EXPECT_POINT_ID_ONE_OF: a,b,...
- EXPECT_CLICK_ID: <int>    or EXPECT_CLICK_ID_ONE_OF: a,b,...
- SPOKEN_MUST_CONTAIN: <substring>
- SPOKEN_MUST_NOT_CONTAIN: <substring>
- SPOKEN_MUST_MATCH_REGEX: <regex>
- SPOKEN_MAX_WORDS: <int>

Usage:
  python pace-eval-fm-fixtures.py <hf-base-dir> <lora-path> [system-prompt-path]
"""
import json
import re
import subprocess
import sys
from pathlib import Path

FM_FIX_DIR = Path("/Users/sarthak/Desktop/fleet/clickyLocal/evals/fm-fixtures")
DEFAULT_PROMPT = Path("/Users/sarthak/Desktop/fleet/tinygpt/grammars/pace-system-prompt-v3.txt")
TINYGPT = "/Users/sarthak/Desktop/fleet/tinygpt/native-mac/.build/arm64-apple-macosx/release/tinygpt"


def parse_fixture(text: str) -> dict:
    out = {"user": "", "elements": [], "expects": {}, "free_text": False}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("USER:"):
            out["user"] = line.removeprefix("USER:").strip()
        elif line.startswith("ELEMENT:"):
            out["elements"].append(line.removeprefix("ELEMENT:").strip())
        elif line.startswith("FREE_TEXT_MODE:"):
            out["free_text"] = "true" in line.lower()
        elif ":" in line and any(line.startswith(p) for p in
                                  ["EXPECT_POINT_ID", "EXPECT_CLICK_ID",
                                   "SPOKEN_MUST_CONTAIN", "SPOKEN_MUST_NOT_CONTAIN",
                                   "SPOKEN_MUST_MATCH_REGEX", "SPOKEN_MAX_WORDS"]):
            key, _, val = line.partition(":")
            out["expects"][key.strip()] = val.strip()
    return out


def build_prompt(fx: dict, system_prompt: str) -> str:
    user_parts = []
    if fx["elements"]:
        user_parts.append("on-screen elements:")
        user_parts.extend(fx["elements"])
        user_parts.append("")
    user_parts.append(f"user said: {fx['user']}")
    return f"system: {system_prompt}\n\nuser: " + "\n".join(user_parts) + "\n\nassistant:"


def extract_json_response(content: str) -> dict | None:
    """Pull the first balanced JSON object out of model output."""
    # Find first '{' and try to parse forward until we get valid JSON
    start = content.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(content[start:i+1])
                    except json.JSONDecodeError:
                        break
        start = content.find("{", start + 1)
    return None


def evaluate_one(fx_path: Path, hf_dir: str, lora_path: str, system_prompt: str) -> dict:
    fx = parse_fixture(fx_path.read_text())
    prompt = build_prompt(fx, system_prompt)
    try:
        result = subprocess.run(
            [TINYGPT, "hf-load", hf_dir, "--lora", lora_path, "--sample",
             "--prompt", prompt, "--tokens", "180", "--temperature", "0.0"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return {"name": fx_path.stem, "pass": False,
                    "failures": [f"exit {result.returncode}"],
                    "content": result.stderr[:200]}
        full = result.stdout
        # Take content AFTER the prompt as printed
        if prompt in full:
            content = full.split(prompt, 1)[1]
        else:
            content = full
        content = re.split(r"\n\(\d+ tokens? in", content)[0].strip()
    except Exception as e:
        return {"name": fx_path.stem, "pass": False, "failures": [str(e)], "content": ""}

    failures = []
    spoken = content
    # If JSON expected (not free_text mode), extract spokenText + IDs
    point_id = None
    click_id = None
    if not fx["free_text"]:
        parsed = extract_json_response(content)
        if parsed is None:
            failures.append("no JSON object found in output")
        else:
            spoken = str(parsed.get("spokenText", ""))
            point_id = parsed.get("pointAtElementId")
            click_id = parsed.get("clickElementId")

    # Check spoken expectations
    exp = fx["expects"]
    if "SPOKEN_MUST_CONTAIN" in exp:
        sub = exp["SPOKEN_MUST_CONTAIN"]
        if sub.lower() not in spoken.lower():
            failures.append(f"missing substring: {sub}")
    if "SPOKEN_MUST_NOT_CONTAIN" in exp:
        sub = exp["SPOKEN_MUST_NOT_CONTAIN"]
        if sub.lower() in spoken.lower():
            failures.append(f"forbidden substring matched: {sub}")
    if "SPOKEN_MUST_MATCH_REGEX" in exp:
        pat = exp["SPOKEN_MUST_MATCH_REGEX"]
        if not re.search(pat, spoken):
            failures.append(f"regex not matched: {pat}")
    if "SPOKEN_MAX_WORDS" in exp:
        cap = int(exp["SPOKEN_MAX_WORDS"])
        wc = len(spoken.split())
        if wc > cap:
            failures.append(f"too many words: {wc} > {cap}")
    if "EXPECT_POINT_ID" in exp and point_id is not None:
        want = int(exp["EXPECT_POINT_ID"])
        if point_id != want:
            failures.append(f"pointAtElementId: got {point_id}, want {want}")
    if "EXPECT_POINT_ID_ONE_OF" in exp and point_id is not None:
        want = [int(x) for x in exp["EXPECT_POINT_ID_ONE_OF"].split(",")]
        if point_id not in want:
            failures.append(f"pointAtElementId: got {point_id}, want one of {want}")
    if "EXPECT_CLICK_ID" in exp and click_id is not None:
        want = int(exp["EXPECT_CLICK_ID"])
        if click_id != want:
            failures.append(f"clickElementId: got {click_id}, want {want}")
    if "EXPECT_CLICK_ID_ONE_OF" in exp and click_id is not None:
        want = [int(x) for x in exp["EXPECT_CLICK_ID_ONE_OF"].split(",")]
        if click_id not in want:
            failures.append(f"clickElementId: got {click_id}, want one of {want}")

    return {"name": fx_path.stem, "pass": len(failures) == 0,
            "failures": failures, "content": content[:300],
            "spoken": spoken[:200], "point_id": point_id, "click_id": click_id}


def main():
    if len(sys.argv) < 3:
        print("usage: pace-eval-fm-fixtures.py <hf-base-dir> <lora-path> [system-prompt-path]", file=sys.stderr)
        return 2
    hf_dir, lora = sys.argv[1], sys.argv[2]
    sys_prompt_path = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_PROMPT
    sys_prompt = sys_prompt_path.read_text().strip()

    fixtures = sorted(FM_FIX_DIR.glob("*.txt"))
    print(f"=== pace v3 eval ({len(fixtures)} fm-fixtures, lora={Path(lora).name}) ===\n")
    passed = 0
    for fx in fixtures:
        r = evaluate_one(fx, hf_dir, lora, sys_prompt)
        status = "PASS" if r["pass"] else "FAIL"
        print(f"[{status}] {r['name']}")
        if not r["pass"]:
            for f in r["failures"]:
                print(f"    - {f}")
            if "spoken" in r:
                print(f"    spoken: {r['spoken'][:120]}")
            if r.get("point_id") is not None or r.get("click_id") is not None:
                print(f"    IDs: point={r.get('point_id')} click={r.get('click_id')}")
        else:
            passed += 1
    print(f"\n=== {passed}/{len(fixtures)} fm-fixtures passed ===")
    return 0 if passed == len(fixtures) else 1


if __name__ == "__main__":
    sys.exit(main())
