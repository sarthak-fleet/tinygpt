#!/usr/bin/env python3
"""Same as pace-eval-baseline.py but adds response_format JSON schema."""
import json, re, sys, urllib.request
from pathlib import Path

FM_FIX_DIR = Path("/Users/sarthak/Desktop/fleet/clickyLocal/evals/fm-fixtures")
SYSP_PATH = Path("/Users/sarthak/Desktop/fleet/tinygpt/grammars/pace-system-prompt-v3.txt")
SCHEMA_PATH = Path("/Users/sarthak/Desktop/fleet/tinygpt/grammars/pace-fm-response.schema.json")


def parse_fx(text):
    out = {"user": "", "elements": [], "expects": {}, "free_text": False}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("USER:"): out["user"] = line.removeprefix("USER:").strip()
        elif line.startswith("ELEMENT:"): out["elements"].append(line.removeprefix("ELEMENT:").strip())
        elif line.startswith("FREE_TEXT_MODE:"): out["free_text"] = "true" in line.lower()
        elif ":" in line and any(line.startswith(p) for p in
                ["EXPECT_POINT_ID","EXPECT_CLICK_ID","SPOKEN_MUST_CONTAIN",
                 "SPOKEN_MUST_NOT_CONTAIN","SPOKEN_MUST_MATCH_REGEX","SPOKEN_MAX_WORDS"]):
            k,_,v = line.partition(":"); out["expects"][k.strip()] = v.strip()
    return out


def check_must_contain(s, v):
    return [t for t in [x.strip().lower() for x in v.split(",") if x.strip()] if t not in s.lower()]
def check_must_not_contain(s, v):
    return [t for t in [x.strip().lower() for x in v.split(",") if x.strip()] if t in s.lower()]


def eval_one(fx_path, url, model_id, sys_prompt, schema):
    fx = parse_fx(fx_path.read_text())
    user_parts = []
    if fx["elements"]:
        user_parts.append("on-screen elements:")
        user_parts.extend(fx["elements"])
        user_parts.append("")
    user_parts.append(f"user said: {fx['user']}")
    body = {
        "model": model_id,
        "messages":[
            {"role":"system","content":sys_prompt},
            {"role":"user","content":"\n".join(user_parts)},
        ],
        "temperature":0.0, "max_tokens":250, "stream":False,
    }
    if not fx["free_text"]:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name":"PaceFMTurnResponse","strict":True,"schema":schema}
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
    if not fx["free_text"]:
        try:
            d = json.loads(content)
            spoken = str(d.get("spokenText","")); pid = d.get("pointAtElementId"); cid = d.get("clickElementId")
        except json.JSONDecodeError:
            fails.append("not valid JSON")

    exp = fx["expects"]
    if "SPOKEN_MUST_CONTAIN" in exp:
        for t in check_must_contain(spoken, exp["SPOKEN_MUST_CONTAIN"]):
            fails.append(f"missing: {t}")
    if "SPOKEN_MUST_NOT_CONTAIN" in exp:
        for t in check_must_not_contain(spoken, exp["SPOKEN_MUST_NOT_CONTAIN"]):
            fails.append(f"forbidden: {t}")
    if "SPOKEN_MUST_MATCH_REGEX" in exp and not re.search(exp["SPOKEN_MUST_MATCH_REGEX"], spoken):
        fails.append(f"regex: {exp['SPOKEN_MUST_MATCH_REGEX']}")
    if "SPOKEN_MAX_WORDS" in exp and len(spoken.split()) > int(exp["SPOKEN_MAX_WORDS"]):
        fails.append(f"words: {len(spoken.split())} > {exp['SPOKEN_MAX_WORDS']}")
    if "EXPECT_POINT_ID" in exp and pid is not None and pid != int(exp["EXPECT_POINT_ID"]):
        fails.append(f"point: {pid} vs {exp['EXPECT_POINT_ID']}")
    if "EXPECT_POINT_ID_ONE_OF" in exp and pid is not None:
        ok = [int(x) for x in exp["EXPECT_POINT_ID_ONE_OF"].split(",")]
        if pid not in ok: fails.append(f"point: {pid} not in {ok}")
    if "EXPECT_CLICK_ID" in exp and cid is not None and cid != int(exp["EXPECT_CLICK_ID"]):
        fails.append(f"click: {cid} vs {exp['EXPECT_CLICK_ID']}")
    if "EXPECT_CLICK_ID_ONE_OF" in exp and cid is not None:
        ok = [int(x) for x in exp["EXPECT_CLICK_ID_ONE_OF"].split(",")]
        if cid not in ok: fails.append(f"click: {cid} not in {ok}")
    return {"name": fx_path.stem, "pass": not fails, "failures": fails,
            "spoken": spoken[:120], "pid": pid, "cid": cid}


def main():
    url = "http://127.0.0.1:1234/v1/chat/completions"
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen/qwen3-30b-a3b"
    sysp_path = Path(sys.argv[2]) if len(sys.argv) > 2 else SYSP_PATH
    sysp = sysp_path.read_text().strip()
    schema = json.loads(SCHEMA_PATH.read_text())

    fxs = sorted(FM_FIX_DIR.glob("*.txt"))
    print(f"=== {model} + response_format json_schema (prompt: {sysp_path.name}) ===\n")
    passed = 0
    for fx in fxs:
        r = eval_one(fx, url, model, sysp, schema)
        s = "PASS" if r["pass"] else "FAIL"
        print(f"[{s}] {r['name']}")
        if not r["pass"]:
            for f in r["failures"]: print(f"    - {f}")
            if r.get("spoken"): print(f"    spoken: {r['spoken']}")
        else:
            passed += 1
    print(f"\n=== {passed}/{len(fxs)} fm-fixtures passed ===")


if __name__ == "__main__":
    main()
