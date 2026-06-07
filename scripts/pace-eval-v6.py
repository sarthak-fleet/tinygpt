#!/usr/bin/env python3
"""v6 eval: model emits labels, this script does the deterministic label→ID
lookup before comparing to fixture EXPECT_POINT_ID / EXPECT_CLICK_ID.

Label matching:
1. Exact match (case-insensitive) on element label field
2. Fuzzy substring match (model's label is contained in element's label OR vice versa)
3. Returns -1 if no element matches (correct behavior for "target not present")
"""
import json, re, sys, urllib.request
from pathlib import Path

FM_FIX_DIR = Path("/Users/sarthak/Desktop/fleet/clickyLocal/evals/fm-fixtures")
SYSP_PATH = Path("/Users/sarthak/Desktop/fleet/tinygpt/grammars/pace-system-prompt-v6-label.txt")
SCHEMA_PATH = Path("/Users/sarthak/Desktop/fleet/tinygpt/grammars/pace-fm-label-response.schema.json")


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


def element_id_for_label(elements, label):
    """Deterministic lookup: model-emitted label → numeric ID.
    - Empty/missing label → -1 (no target).
    - Exact case-insensitive match preferred.
    - Else: substring match (model's label is contained in element's label OR vice versa).
    - Else: -1.
    Element format: [N] role|x,y|label|text
    """
    if not label or not label.strip():
        return -1
    target = label.strip().lower()
    parsed = []
    for el in elements:
        m = re.match(r"\[(\d+)\]\s*\w+\|[\d,]+\|([^|]+)", el)
        if m:
            parsed.append((int(m.group(1)), m.group(2).strip().lower()))
    # exact match
    for eid, el_label in parsed:
        if el_label == target:
            return eid
    # substring containment (model's label in element's label)
    for eid, el_label in parsed:
        if target in el_label:
            return eid
    # reverse containment (element's label in model's label)
    for eid, el_label in parsed:
        if el_label in target:
            return eid
    return -1


def extract_json(text):
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            parsed, _ = decoder.raw_decode(text[start:])
            return parsed
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


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
        "messages": [
            {"role":"system","content":sys_prompt},
            {"role":"user","content":"\n".join(user_parts)},
        ],
        "temperature":0.0, "max_tokens":250, "stream":False,
    }
    if not fx["free_text"]:
        body["response_format"] = {
            "type":"json_schema",
            "json_schema":{"name":"PaceFMLabelResponse","strict":True,"schema":schema}
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
    pid, cid = -1, -1
    point_label, click_label = "", ""
    parsed = extract_json(content)
    if parsed is not None:
        spoken = str(parsed.get("spokenText", content))
        point_label = str(parsed.get("pointAtLabel", "") or "")
        click_label = str(parsed.get("clickLabel", "") or "")
        # Deterministic lookup label → ID
        pid = element_id_for_label(fx["elements"], point_label)
        cid = element_id_for_label(fx["elements"], click_label)
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
        fails.append(f"regex: {exp['SPOKEN_MUST_MATCH_REGEX']}")
    if "SPOKEN_MAX_WORDS" in exp:
        cap = int(exp["SPOKEN_MAX_WORDS"]); wc = len(spoken.split())
        if wc > cap: fails.append(f"words {wc} > {cap}")
    if "EXPECT_POINT_ID" in exp and pid != int(exp["EXPECT_POINT_ID"]):
        fails.append(f"point: got {pid} (from label '{point_label}') want {exp['EXPECT_POINT_ID']}")
    if "EXPECT_POINT_ID_ONE_OF" in exp:
        want = [int(x) for x in exp["EXPECT_POINT_ID_ONE_OF"].split(",")]
        if pid not in want: fails.append(f"point: got {pid} (from '{point_label}') want one_of {want}")
    if "EXPECT_CLICK_ID" in exp and cid != int(exp["EXPECT_CLICK_ID"]):
        fails.append(f"click: got {cid} (from label '{click_label}') want {exp['EXPECT_CLICK_ID']}")
    if "EXPECT_CLICK_ID_ONE_OF" in exp:
        want = [int(x) for x in exp["EXPECT_CLICK_ID_ONE_OF"].split(",")]
        if cid not in want: fails.append(f"click: got {cid} (from '{click_label}') want one_of {want}")
    return {"name": fx_path.stem, "pass": not fails, "failures": fails,
            "spoken": spoken[:120], "pid": pid, "cid": cid,
            "point_label": point_label, "click_label": click_label}


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/v1/chat/completions"
    model = sys.argv[2] if len(sys.argv) > 2 else "tinygpt"
    sysp = SYSP_PATH.read_text().strip()
    schema = json.loads(SCHEMA_PATH.read_text())

    fxs = sorted(FM_FIX_DIR.glob("*.txt"))
    print(f"=== v6 (label-based) eval against {len(fxs)} fm-fixtures ===\n")
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
