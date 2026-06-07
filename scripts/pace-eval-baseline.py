#!/usr/bin/env python3
"""pace-eval-baseline.py — fm-fixture eval against an OpenAI-compatible
HTTP endpoint. Uses comma-split MUST_CONTAIN / NOT_CONTAIN."""
import json, re, sys, urllib.request
from pathlib import Path

FM_FIX_DIR = Path("/Users/sarthak/Desktop/fleet/clickyLocal/evals/fm-fixtures")
SYSP_PATH = Path("/Users/sarthak/Desktop/fleet/tinygpt/grammars/pace-system-prompt-v3.txt")


def parse_fx(text):
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
                ["EXPECT_POINT_ID","EXPECT_CLICK_ID","SPOKEN_MUST_CONTAIN",
                 "SPOKEN_MUST_NOT_CONTAIN","SPOKEN_MUST_MATCH_REGEX","SPOKEN_MAX_WORDS"]):
            k,_,v = line.partition(":")
            out["expects"][k.strip()] = v.strip()
    return out


def build_user_msg(fx):
    parts = []
    if fx["elements"]:
        parts.append("on-screen elements:")
        parts.extend(fx["elements"])
        parts.append("")
    parts.append(f"user said: {fx['user']}")
    return "\n".join(parts)


def extract_json(text):
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try: return json.loads(text[start:i+1])
                    except json.JSONDecodeError: break
        start = text.find("{", start + 1)
    return None


def check_must_contain(spoken, exp_value):
    """Comma-separated tokens — ALL must be present."""
    missing = []
    for tok in [t.strip().lower() for t in exp_value.split(",") if t.strip()]:
        if tok not in spoken.lower():
            missing.append(tok)
    return missing


def check_must_not_contain(spoken, exp_value):
    """Comma-separated tokens — NONE may be present."""
    matched = []
    for tok in [t.strip().lower() for t in exp_value.split(",") if t.strip()]:
        if tok in spoken.lower():
            matched.append(tok)
    return matched


def eval_one(fx_path, url, model_id, sys_prompt):
    fx = parse_fx(fx_path.read_text())
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": build_user_msg(fx)},
        ],
        "temperature": 0.0, "max_tokens": 250, "stream": False,
    }
    try:
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
            headers={"Content-Type":"application/json"}, method="POST")
        r = urllib.request.urlopen(req, timeout=180).read()
        d = json.loads(r)
        content = d["choices"][0]["message"]["content"]
    except Exception as e:
        return {"name": fx_path.stem, "pass": False, "failures": [str(e)], "content": ""}

    fails, spoken = [], content
    pid, cid = None, None
    # ALWAYS try to extract JSON first — if model wrapped action-tag in JSON
    # (which grammar enforcement causes), pull spokenText out so action
    # regex tests run against the actual response content, not JSON keys.
    parsed = extract_json(content)
    if parsed is not None:
        spoken = str(parsed.get("spokenText", content))
        pid = parsed.get("pointAtElementId"); cid = parsed.get("clickElementId")
    elif not fx["free_text"]:
        fails.append("no JSON in output")

    exp = fx["expects"]
    if "SPOKEN_MUST_CONTAIN" in exp:
        for t in check_must_contain(spoken, exp["SPOKEN_MUST_CONTAIN"]):
            fails.append(f"missing: {t}")
    if "SPOKEN_MUST_NOT_CONTAIN" in exp:
        for t in check_must_not_contain(spoken, exp["SPOKEN_MUST_NOT_CONTAIN"]):
            fails.append(f"forbidden: {t}")
    if "SPOKEN_MUST_MATCH_REGEX" in exp and not re.search(exp["SPOKEN_MUST_MATCH_REGEX"], spoken):
        fails.append(f"regex miss: {exp['SPOKEN_MUST_MATCH_REGEX']}")
    if "SPOKEN_MAX_WORDS" in exp:
        cap = int(exp["SPOKEN_MAX_WORDS"]); wc = len(spoken.split())
        if wc > cap: fails.append(f"words {wc} > {cap}")
    if "EXPECT_POINT_ID" in exp and pid is not None and pid != int(exp["EXPECT_POINT_ID"]):
        fails.append(f"point: got {pid} want {exp['EXPECT_POINT_ID']}")
    if "EXPECT_POINT_ID_ONE_OF" in exp and pid is not None:
        want = [int(x) for x in exp["EXPECT_POINT_ID_ONE_OF"].split(",")]
        if pid not in want: fails.append(f"point: got {pid} want one_of {want}")
    if "EXPECT_CLICK_ID" in exp and cid is not None and cid != int(exp["EXPECT_CLICK_ID"]):
        fails.append(f"click: got {cid} want {exp['EXPECT_CLICK_ID']}")
    if "EXPECT_CLICK_ID_ONE_OF" in exp and cid is not None:
        want = [int(x) for x in exp["EXPECT_CLICK_ID_ONE_OF"].split(",")]
        if cid not in want: fails.append(f"click: got {cid} want one_of {want}")
    return {"name": fx_path.stem, "pass": not fails, "failures": fails,
            "spoken": spoken[:120], "pid": pid, "cid": cid}


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:1234/v1/chat/completions"
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen/qwen3-30b-a3b"
    sysp_path = Path(sys.argv[3]) if len(sys.argv) > 3 else SYSP_PATH
    sysp = sysp_path.read_text().strip()

    fxs = sorted(FM_FIX_DIR.glob("*.txt"))
    print(f"=== {model} on {len(fxs)} fm-fixtures (prompt: {sysp_path.name}) ===\n")
    passed = 0
    for fx in fxs:
        r = eval_one(fx, url, model, sysp)
        s = "PASS" if r["pass"] else "FAIL"
        print(f"[{s}] {r['name']}")
        if not r["pass"]:
            for f in r["failures"]: print(f"    - {f}")
            if r.get("spoken"): print(f"    spoken: {r['spoken']}")
            if r.get("pid") is not None or r.get("cid") is not None:
                print(f"    pid={r.get('pid')} cid={r.get('cid')}")
        else:
            passed += 1
    print(f"\n=== {passed}/{len(fxs)} fm-fixtures passed ===")
    return 0 if passed == len(fxs) else 1


if __name__ == "__main__":
    sys.exit(main())
