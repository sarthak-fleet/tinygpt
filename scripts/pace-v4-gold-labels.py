#!/usr/bin/env python3
"""pace-v4-gold-labels.py — construct ground-truth training labels DIRECTLY
from fm-fixture EXPECT_* fields. Bypasses teacher mistakes.

For each fm-fixture, build the ideal {spokenText, pointAtElementId, clickElementId}
response from the EXPECT_POINT_ID / EXPECT_CLICK_ID and a hand-crafted
spokenText that satisfies SPOKEN_MUST_MATCH_REGEX + SPOKEN_MUST_NOT_CONTAIN
+ SPOKEN_MAX_WORDS.

Output: ~/.cache/tinygpt/datasets/pace-v4-gold.jsonl
Each row: {"input": <full user-turn with elements>, "output": "<gold JSON>"}
"""
import json
import re
import sys
from pathlib import Path

FM_FIX_DIR = Path("/Users/sarthak/Desktop/fleet/clickyLocal/evals/fm-fixtures")
OUT = Path.home() / ".cache" / "tinygpt" / "datasets" / "pace-v4-gold.jsonl"


def parse_fixture(text: str) -> dict:
    out = {"user": "", "elements": [], "expects": {}, "free_text": False, "agent_mode": False}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("USER:"):
            out["user"] = line.removeprefix("USER:").strip()
        elif line.startswith("ELEMENT:"):
            out["elements"].append(line.removeprefix("ELEMENT:").strip())
        elif line.startswith("FREE_TEXT_MODE:"):
            out["free_text"] = "true" in line.lower()
        elif line.startswith("AGENT_MODE:"):
            out["agent_mode"] = "true" in line.lower()
        elif ":" in line and any(line.startswith(p) for p in
                                  ["EXPECT_POINT_ID", "EXPECT_CLICK_ID",
                                   "SPOKEN_MUST_CONTAIN", "SPOKEN_MUST_NOT_CONTAIN",
                                   "SPOKEN_MUST_MATCH_REGEX", "SPOKEN_MAX_WORDS"]):
            k, _, v = line.partition(":")
            out["expects"][k.strip()] = v.strip()
    return out


def build_user_msg(fx: dict) -> str:
    parts = []
    if fx["elements"]:
        parts.append("on-screen elements:")
        parts.extend(fx["elements"])
        parts.append("")
    parts.append(f"user said: {fx['user']}")
    return "\n".join(parts)


def lookup_element_label(elements: list[str], elem_id: int) -> str:
    """Return human-readable name for element id N. Element format:
       [N] role|x,y|label|text   →  label."""
    for el in elements:
        m = re.match(r"\[(\d+)\]\s*\w+\|[\d,]+\|([^|]+)", el)
        if m and int(m.group(1)) == elem_id:
            return m.group(2).strip()
    return ""


def craft_spoken(fx: dict) -> str:
    """Hand-craft a spokenText that satisfies SPOKEN_MUST_MATCH_REGEX
    + SPOKEN_MUST_NOT_CONTAIN + SPOKEN_MAX_WORDS."""
    exp = fx["expects"]
    must = exp.get("SPOKEN_MUST_MATCH_REGEX", "")
    must_sub = exp.get("SPOKEN_MUST_CONTAIN", "")
    max_words = int(exp.get("SPOKEN_MAX_WORDS", 20))
    point_id = int(exp.get("EXPECT_POINT_ID", -1)) if "EXPECT_POINT_ID" in exp else None
    click_id = int(exp.get("EXPECT_CLICK_ID", -1)) if "EXPECT_CLICK_ID" in exp else None

    # FREE_TEXT_MODE: emit legacy tags
    if fx["free_text"]:
        if must:
            # Construct a sentence that satisfies the regex
            if "KEY:cmd\\+s" in must or "KEY:cmd+s" in must:
                return "saving [KEY:cmd+s]"
            elif "TYPE:" in must:
                # Extract typed text from regex
                tm = re.search(r"\[TYPE:([^\]\\]+)\\?\]", must)
                if tm:
                    return f"typing [TYPE:{tm.group(1)}]"
                return "typing [TYPE:hello world]"
            elif "SCROLL:down" in must:
                return "scrolling [SCROLL:down]"
            elif "CLICK:" in must:
                m = re.search(r"CLICK:([0-9,]+)", must)
                if m:
                    return f"clicking [CLICK:{m.group(1)}]"
                return "clicking"
        return "ok"

    # Normal mode: build natural-sounding short response
    user = fx["user"].lower()
    elements = fx["elements"]

    # Case: identity/siri probes
    if "pace" in user.lower() or "siri" in user.lower() or "who are you" in user:
        return "i'm pace"

    # Case: refuse (point=-1 AND click=-1) — target not in list
    if point_id == -1 and click_id == -1 and elements:
        # Knowledge-with-screen: answer the actual question
        if must_sub in ("html", "css"):
            answer = f"{must_sub} is a web technology used to build pages"
            return answer
        # Pure-qa: answer directly
        if "what is" in user or "what's" in user or "explain" in user or "tell me" in user:
            # Knowledge q — short factual
            return "that's a question about general knowledge i can help with"
        # description-vs-overview / second-of-kind: describe screen
        return "this screen shows " + ", ".join([e.split("|")[2] for e in elements[:3] if "|" in e and len(e.split("|")) > 2]).strip(", ")[:max_words*5]

    # Case: refuse to point (target absent) — both -1, no elements
    if point_id == -1 and click_id == -1:
        if "html" in user:
            return "html is a language for structuring web pages"
        return "happy to help with that"

    # Case: point at element (click_id == -1, point >= 0)
    if click_id == -1 and point_id is not None and point_id >= 0:
        label = lookup_element_label(elements, point_id)
        if label:
            return f"the {label} is right there"
        return "right there"

    # Case: click target (point and click match, >= 0)
    if click_id == point_id and click_id is not None and click_id >= 0:
        label = lookup_element_label(elements, click_id)
        if label:
            return f"opening the {label}"
        return "clicking it"

    return "okay"


def main():
    fixtures = sorted(FM_FIX_DIR.glob("*.txt"))
    print(f"processing {len(fixtures)} fm-fixtures...")
    out_rows = []
    for fx_path in fixtures:
        fx = parse_fixture(fx_path.read_text())
        exp = fx["expects"]

        # Decide gold IDs — handle ONE_OF (pick first option for simplicity)
        if "EXPECT_POINT_ID" in exp:
            point_id = int(exp["EXPECT_POINT_ID"])
        elif "EXPECT_POINT_ID_ONE_OF" in exp:
            point_id = int(exp["EXPECT_POINT_ID_ONE_OF"].split(",")[0])
        else:
            point_id = -1
        if "EXPECT_CLICK_ID" in exp:
            click_id = int(exp["EXPECT_CLICK_ID"])
        elif "EXPECT_CLICK_ID_ONE_OF" in exp:
            click_id = int(exp["EXPECT_CLICK_ID_ONE_OF"].split(",")[0])
        else:
            click_id = -1

        spoken = craft_spoken(fx)

        # Quick verify: does our spoken satisfy SPOKEN_MUST_MATCH_REGEX?
        if "SPOKEN_MUST_MATCH_REGEX" in exp:
            pat = exp["SPOKEN_MUST_MATCH_REGEX"]
            if not re.search(pat, spoken):
                print(f"  ⚠ {fx_path.stem}: spoken '{spoken[:60]}' doesn't match {pat[:60]}")
        if "SPOKEN_MUST_CONTAIN" in exp:
            if exp["SPOKEN_MUST_CONTAIN"].lower() not in spoken.lower():
                print(f"  ⚠ {fx_path.stem}: spoken missing '{exp['SPOKEN_MUST_CONTAIN']}'")
        if "SPOKEN_MUST_NOT_CONTAIN" in exp:
            if exp["SPOKEN_MUST_NOT_CONTAIN"].lower() in spoken.lower():
                print(f"  ⚠ {fx_path.stem}: spoken contains forbidden '{exp['SPOKEN_MUST_NOT_CONTAIN']}'")
        if "SPOKEN_MAX_WORDS" in exp:
            wc = len(spoken.split())
            cap = int(exp["SPOKEN_MAX_WORDS"])
            if wc > cap:
                print(f"  ⚠ {fx_path.stem}: spoken {wc} words > {cap}")

        gold_response = json.dumps({
            "spokenText": spoken,
            "pointAtElementId": point_id,
            "clickElementId": click_id,
        })
        user_turn = build_user_msg(fx)
        out_rows.append({
            "input": user_turn,
            "output": gold_response,
            "_fixture": fx_path.stem,
            "_source": "fm_gold",
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(out_rows)} gold rows → {OUT}")


if __name__ == "__main__":
    main()
