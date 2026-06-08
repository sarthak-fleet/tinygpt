#!/usr/bin/env python3
"""FakePace — a rule-based 'planner' that uses NO model inference.

Implements the same response shape as Pace's v6 label-based architecture
({spokenText, pointAtLabel, clickLabel} JSON or FREE_TEXT_MODE string),
using only:

  - regex over the user prompt
  - substring matching against the element list
  - a tiny knowledge-lookup template

If this endpoint scores comparably to v3/v5/v6 LoRA artifacts on the
fm-fixtures, it proves the fixtures are testing framework + grammar,
not model contribution. That's the eval methodology gate for #270.

Usage: import + call respond(user, elements, free_text_mode=False, sys_prompt_v6=True),
       OR run as CLI to score against fm-fixtures.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Action-tag templates kept identical to v6-label system prompt so the
# regex assertions in fm-fixtures pass.
KEY_MAP = {
    "command": "cmd", "cmd": "cmd",
    "control": "ctrl", "ctrl": "ctrl",
    "shift": "shift",
    "option": "alt", "alt": "alt",
    "enter": "Return", "return": "Return",
    "escape": "Escape", "esc": "Escape",
    "tab": "Tab",
}

CLICK_VERBS = ("click", "tap", "press", "open", "launch", "hit", "choose", "select", "go to")
STOP_WORDS = {"the", "a", "an", "this", "that", "please", "on", "and", "to", "for", "of"}
# Generic role/type words that appear in many labels — don't count them as
# meaningful matches. e.g. "click the elephant button" shouldn't match
# "search button" just because both contain "button".
GENERIC_LABEL_WORDS = {
    "button", "link", "field", "input", "text", "area", "menu", "item",
    "view", "tab", "icon", "image", "row", "cell", "label", "control",
    "switch", "checkbox", "radio", "slider", "panel", "section", "header",
    "footer", "bar", "box", "list", "group",
}


def parse_user(user: str) -> str:
    return user.strip().lower()


def fake_pace(user: str, elements: list[dict], free_text_mode: bool = False) -> str:
    """Return a response string. JSON-shape if not free_text_mode, else
    a free-text action-tag string (matching v6-label-system-prompt rules).
    """
    u = parse_user(user)

    # --- Identity probes ---
    if re.search(r"\b(who are you|what are you|are you siri|are you apple|are you a chatbot)\b", u):
        return _wrap("i'm pace", "", "", free_text_mode)

    # --- Explicit Siri/AI mention but still expects Pace identity ---
    if "siri" in u or "apple intelligence" in u:
        return _wrap("i'm pace, not siri", "", "", free_text_mode)

    # --- Action tags: keyboard ---
    # "press cmd plus shift plus t", "hit cmd s", "press enter", "press escape"
    key_match = re.search(
        r"(?:press|hit|use(?: the)?(?: keyboard)?(?: shortcut)?(?: for)?)\s+"
        r"((?:command|cmd|control|ctrl|shift|alt|option)?(?:\s*(?:\+|plus)\s*"
        r"(?:command|cmd|control|ctrl|shift|alt|option))*\s*"
        r"(?:\+|plus|\s)?\s*[a-z0-9]+)",
        u,
    )
    if key_match:
        spec = key_match.group(1)
        parts = re.split(r"\s*(?:\+|plus|\s)+\s*", spec.strip())
        parts = [p for p in parts if p]
        canon = [KEY_MAP.get(p, p) for p in parts]
        # Normalize ordering: modifiers first, key last
        mods = [p for p in canon if p in ("cmd", "ctrl", "shift", "alt")]
        keys = [p for p in canon if p not in ("cmd", "ctrl", "shift", "alt")]
        if not keys:
            keys = [canon[-1]] if canon else []
        tag = "+".join(mods + keys)
        return _wrap(f"[KEY:{tag}] doing that", "", "", free_text_mode)

    # --- Action: scroll ---
    scroll_match = re.search(r"scroll\s+(up|down)(?:\s+(?:\w+\s+)?(\d+)\s+times?)?", u)
    if not scroll_match:
        scroll_match = re.search(r"scroll\s+(back\s+up|down)", u)
    if not scroll_match:
        scroll_match = re.search(r"(?:go to|jump to)\s+the\s+(top|bottom)", u)
    if scroll_match:
        if "bottom" in u or "down" in scroll_match.group(0):
            direction = "down"
        elif "top" in u or "up" in scroll_match.group(0):
            direction = "up"
        else:
            direction = scroll_match.group(1)
        # Optional count
        cnt = re.search(r"\b(\d+)\s+times?\b", u) or re.search(r"\bthree\s+times?\b", u)
        if cnt and cnt.group(0).startswith("three"):
            tag = f"[SCROLL:{direction}:3]"
        elif cnt:
            tag = f"[SCROLL:{direction}:{cnt.group(1)}]"
        elif "bottom" in u:
            tag = f"[SCROLL:down:5]"
        else:
            tag = f"[SCROLL:{direction}]"
        return _wrap(f"{tag} scrolling", "", "", free_text_mode)

    # --- Action: type ---
    type_match = re.search(r"type\s+(?:the\s+message\s+|the\s+text\s+|in\s+)?[\"']?(.+?)[\"']?$", u)
    if not type_match:
        type_match = re.search(r"(?:write|enter|fill in)\s+(?:the\s+message\s+|my\s+name\s+as\s+|in\s+)?[\"']?(.+?)[\"']?$", u)
    if type_match and u.startswith(("type", "write", "enter", "fill")):
        text = type_match.group(1).strip().strip(".").strip("?!")
        # Strip leading "the ... field" context if user said "enter X in the Y field"
        if " in the " in text:
            text = text.split(" in the ")[0].strip()
        return _wrap(f"[TYPE:{text}] typing", "", "", free_text_mode)

    # --- Action: OPEN_APP ---
    open_app_match = re.search(r"(?:open|launch|start)\s+(safari|notes|calendar|finder|mail|messages|music|photos|maps|chrome|firefox|terminal|xcode)\b", u)
    if open_app_match:
        app = open_app_match.group(1).capitalize()
        return _wrap(f"[OPEN_APP:{app}] opening {app}", "", "", free_text_mode)

    # --- Compound action: "click X and type Y" / "click X then scroll" etc. ---
    # Detect the click-then-action chain pattern before plain click handling.
    chain_match = re.search(
        r"(click|tap|press)\s+(?:the\s+)?(.+?)\s+(?:and|then)\s+(type|enter|scroll)\s+(.+?)$",
        u,
    )
    if chain_match:
        target_phrase = chain_match.group(2).strip()
        verb2 = chain_match.group(3)
        arg2 = chain_match.group(4).strip().strip(".?!")
        # Find the element for the click target.
        # Build a fake "click X" user prompt so _find_element can score it.
        target_el = _find_element(f"click the {target_phrase}", elements)
        if target_el is not None:
            x, y = target_el["pos"].split(",")
            tags = [f"[CLICK:{x},{y}]"]
            if verb2 == "scroll":
                direction = "down" if "down" in arg2 else "up"
                tags.append(f"[SCROLL:{direction}]")
            else:  # type / enter
                tags.append(f"[TYPE:{arg2}]")
            spoken = f"{tags[0]} {tags[1]} doing both"
            return _wrap(spoken, target_el["label"], target_el["label"], free_text_mode)

    # --- Click intent → resolve target via substring match ---
    has_click_verb = any(re.search(rf"\b{v}\b", u) for v in CLICK_VERBS)
    if has_click_verb:
        # Empty screen → refuse
        if not elements:
            target_phrase = _extract_target_noun(u)
            return _wrap(
                f"i can't see {target_phrase or 'that'} on this screen",
                "", "", free_text_mode,
            )
        # Try to match user phrase to an element label.
        target = _find_element(u, elements)
        if target is not None:
            label = target["label"]
            return _wrap(f"opening the {label}", label, label, free_text_mode)
        # No match → refuse
        target_phrase = _extract_target_noun(u)
        return _wrap(
            f"i can't see {target_phrase or 'that'} on this screen",
            "", "", free_text_mode,
        )

    # --- Pure QA / knowledge — emit templated response keyed off the question term ---
    # Detect question words.
    qa_match = re.search(r"what(?:'s|\s+is)\s+(.+?)[\?\.]?$", u)
    if not qa_match:
        qa_match = re.search(r"(?:tell me about|describe|explain)\s+(.+?)[\?\.]?$", u)
    if qa_match:
        topic = qa_match.group(1).strip()
        # Strip common context phrases
        topic = re.sub(r"\b(this|that|the|in (this|the) screen|here|on screen)\b", "", topic).strip()
        topic = topic or "this"
        return _wrap(_qa_answer(topic), "", "", free_text_mode)

    # --- Generic fallback: short helpful response ---
    return _wrap("i'm not sure what you'd like me to do", "", "", free_text_mode)


def _wrap(spoken: str, point: str, click: str, free_text_mode: bool) -> str:
    if free_text_mode:
        return spoken
    return json.dumps({"spokenText": spoken, "pointAtLabel": point, "clickLabel": click},
                       ensure_ascii=False, separators=(",", ":"))


def _qa_answer(topic: str) -> str:
    """Returns a short answer that contains the topic word.
    Just ensure the topic word appears + answer feels generic-knowledgeable."""
    # All we need: must contain topic for SPOKEN_MUST_CONTAIN checks.
    # Be concise (under SPOKEN_MAX_WORDS most fixtures use, typically 40).
    return f"{topic} is a common topic — happy to explain more if you want"


def _find_element(user: str, elements: list[dict]) -> dict | None:
    """Substring match user against element labels. Picks the best match."""
    u = parse_user(user)
    # Strip click verbs and stopwords to extract intent target.
    intent = u
    for v in CLICK_VERBS:
        intent = re.sub(rf"\b{v}\b", " ", intent)
    intent_words = {w for w in re.findall(r"\b\w+\b", intent)
                      if w not in STOP_WORDS and len(w) > 1}

    # Score each element. Generic role words ("button", "link") count for
    # 0 to avoid matching "elephant button" → "search button". Each
    # specific overlap word counts 1. Whole-label substring counts 2.
    best, best_score = None, 0
    for el in elements:
        label = el["label"].lower()
        label_words = set(re.findall(r"\b\w+\b", label))
        specific_overlap = (label_words & intent_words) - GENERIC_LABEL_WORDS
        score = len(specific_overlap)
        if label in u or (len(label) > 4 and label in u):
            score += 2
        if score > best_score:
            best, best_score = el, score
    # Require at least one *specific* (non-role) match to avoid the
    # "elephant button" false positive.
    return best if best_score >= 1 else None


def _extract_target_noun(user: str) -> str:
    u = parse_user(user)
    for v in CLICK_VERBS:
        u = re.sub(rf"\b{v}\b", " ", u)
    u = re.sub(r"\b(the|a|an|please|on|to)\b", " ", u)
    return u.strip().strip("?!.").strip()


# ----------------- fm-fixtures parser + scorer (re-implements the
# scoring logic of pace-eval-v6.py without calling any model) -----------------


def parse_fixture(text: str) -> dict:
    out = {"user": "", "elements": [], "expects": {}, "free_text": False}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("USER:"):
            out["user"] = line[len("USER:"):].strip()
        elif line.startswith("ELEMENT:"):
            out["elements"].append(_parse_element(line[len("ELEMENT:"):].strip()))
        elif line.startswith("FREE_TEXT_MODE:"):
            out["free_text"] = "true" in line.lower()
        elif ":" in line and any(line.startswith(p) for p in
                                  ("EXPECT_POINT_ID", "EXPECT_CLICK_ID",
                                    "EXPECT_POINT_ID_ONE_OF", "EXPECT_CLICK_ID_ONE_OF",
                                    "SPOKEN_MUST_CONTAIN", "SPOKEN_MUST_NOT_CONTAIN",
                                    "SPOKEN_MUST_MATCH_REGEX", "SPOKEN_MAX_WORDS",
                                    "BODY_MUST_CONTAIN", "BODY_MUST_BE_EMPTY",
                                    "BODY_MUST_NOT_BE_EMPTY", "BODY_MIN_WORDS")):
            k, _, v = line.partition(":")
            out["expects"][k.strip()] = v.strip()
    return out


def _parse_element(line: str) -> dict:
    """[N] role|x,y|label|text → dict"""
    m = re.match(r"\[(\d+)\]\s+([^|]+)\|(\d+,\d+)\|([^|]*)\|(.*)", line)
    if not m:
        return {"id": -1, "role": "", "pos": "", "label": "", "text": ""}
    return {
        "id": int(m.group(1)),
        "role": m.group(2).strip(),
        "pos": m.group(3),
        "label": m.group(4).strip(),
        "text": m.group(5).strip(),
    }


def element_id_for_label(elements: list[dict], label: str) -> int:
    """Mimics pace-eval-v6.py's lookup: exact case-insensitive match,
    else substring containment in either direction, else -1."""
    if not label or not label.strip():
        return -1
    lab = label.strip().lower()
    for el in elements:
        if el["label"].lower() == lab:
            return el["id"]
    for el in elements:
        el_lab = el["label"].lower()
        if lab in el_lab or el_lab in lab:
            return el["id"]
    return -1


def score_response(response: str, expects: dict, elements: list[dict], free_text_mode: bool) -> tuple[bool, list[str]]:
    """Returns (passed, list of failure reasons). Mirrors pace-eval-v6.py's
    smart-extraction strategy: try JSON first, fall back to free-text."""
    reasons = []

    # Extract spokenText + labels + bodyText from the response.
    spoken = response
    point_label = ""
    click_label = ""
    body_text = ""
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            spoken = parsed.get("spokenText", response)
            point_label = parsed.get("pointAtLabel", "")
            click_label = parsed.get("clickLabel", "")
            body_text = parsed.get("bodyText", "")
    except Exception:
        # Free text — use the whole response as spokenText.
        pass

    # ID checks
    if "EXPECT_POINT_ID" in expects:
        want = int(expects["EXPECT_POINT_ID"])
        got = element_id_for_label(elements, point_label)
        if got != want:
            reasons.append(f"point: got {got} (from label {point_label!r}) want {want}")

    if "EXPECT_CLICK_ID" in expects:
        want = int(expects["EXPECT_CLICK_ID"])
        got = element_id_for_label(elements, click_label)
        if got != want:
            reasons.append(f"click: got {got} (from label {click_label!r}) want {want}")

    if "EXPECT_POINT_ID_ONE_OF" in expects:
        opts = {int(s.strip()) for s in expects["EXPECT_POINT_ID_ONE_OF"].split(",")}
        got = element_id_for_label(elements, point_label)
        if got not in opts:
            reasons.append(f"point: got {got} want one of {opts}")

    if "EXPECT_CLICK_ID_ONE_OF" in expects:
        opts = {int(s.strip()) for s in expects["EXPECT_CLICK_ID_ONE_OF"].split(",")}
        got = element_id_for_label(elements, click_label)
        if got not in opts:
            reasons.append(f"click: got {got} want one of {opts}")

    # Spoken-text checks
    if "SPOKEN_MUST_CONTAIN" in expects:
        for needle in expects["SPOKEN_MUST_CONTAIN"].split(","):
            n = needle.strip().lower()
            if n and n not in spoken.lower():
                reasons.append(f"spoken missing {n!r}")

    if "SPOKEN_MUST_NOT_CONTAIN" in expects:
        for needle in expects["SPOKEN_MUST_NOT_CONTAIN"].split(","):
            n = needle.strip().lower()
            if n and n in spoken.lower():
                reasons.append(f"spoken contains forbidden {n!r}")

    if "SPOKEN_MUST_MATCH_REGEX" in expects:
        pat = expects["SPOKEN_MUST_MATCH_REGEX"]
        if not re.search(pat, spoken):
            reasons.append(f"spoken does not match regex {pat!r}")

    if "SPOKEN_MAX_WORDS" in expects:
        max_w = int(expects["SPOKEN_MAX_WORDS"])
        n_words = len(spoken.split())
        if n_words > max_w:
            reasons.append(f"spoken {n_words} words exceeds max {max_w}")

    # Body-text checks (v9+ schema with bodyText field)
    if "BODY_MUST_BE_EMPTY" in expects:
        if body_text.strip():
            reasons.append(f"bodyText not empty: {body_text[:80]!r}")

    if "BODY_MUST_NOT_BE_EMPTY" in expects:
        if not body_text.strip():
            reasons.append("bodyText is empty (expected compose content)")

    if "BODY_MUST_CONTAIN" in expects:
        for needle in expects["BODY_MUST_CONTAIN"].split(","):
            n = needle.strip().lower()
            if n and n not in body_text.lower():
                reasons.append(f"body missing {n!r}")

    if "BODY_MIN_WORDS" in expects:
        min_w = int(expects["BODY_MIN_WORDS"])
        n_words = len(body_text.split())
        if n_words < min_w:
            reasons.append(f"body {n_words} words below min {min_w}")

    return len(reasons) == 0, reasons


def run_eval(fixtures_dir: Path, verbose: bool = False) -> dict:
    fixtures = sorted(p for p in fixtures_dir.glob("*.txt"))
    results = []
    for f in fixtures:
        fx = parse_fixture(f.read_text())
        response = fake_pace(fx["user"], fx["elements"], free_text_mode=fx["free_text"])
        passed, reasons = score_response(response, fx["expects"], fx["elements"], fx["free_text"])
        results.append({"fixture": f.stem, "passed": passed, "reasons": reasons,
                          "response": response, "user": fx["user"]})
        marker = "PASS" if passed else "FAIL"
        print(f"[{marker}] {f.stem}")
        if verbose or not passed:
            for r in reasons:
                print(f"    - {r}")
            if not passed:
                print(f"    response: {response[:200]}")

    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)
    print(f"\n=== FakePace baseline: {n_pass}/{n_total} fm-fixtures passed ===")
    return {"n_pass": n_pass, "n_total": n_total, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path,
                          default=Path("/Users/sarthak/Desktop/fleet/clickyLocal/evals/fm-fixtures"))
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--json", action="store_true",
                          help="emit results JSON to stdout (instead of human-readable summary)")
    args = parser.parse_args()
    result = run_eval(args.fixtures, verbose=args.verbose)
    if args.json:
        print(json.dumps(result, indent=2))
