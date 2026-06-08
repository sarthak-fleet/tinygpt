#!/usr/bin/env python3
"""FakePaceVLM — rule-based "VLM" using ONLY:

  - AX tree (provided in the fixture; mirrors what AXUIElementCopy* gives Pace at runtime)
  - Apple Vision OCR text (provided in the fixture; raw text extracted from screenshot)
  - NSWorkspace.frontmostApp (provided in the fixture as APP_FROMOST)
  - Simple heuristics

NO model inference. NO vision encoder. Just rules.

This is the eval gate (task #272). Any future Pace VLM specialist
must score meaningfully ABOVE this baseline on `fm-vlm-fixtures-v1/`
to claim it's adding value beyond what AX + OCR + heuristics can do.

Workload Pace's VLM serves (per the design conversation):
  1. Read what's written (= OCR)
  2. Tell where to click (= AX bbox lookup, or VLM grounding when AX blind)
  3. Understand what user is doing (= activity inference from app + state)

The first two are mostly free via macOS APIs. The third is what a
real VLM has to earn. FakePaceVLM does the obvious heuristics on (3)
so we know what the model has to beat.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Activity verbs by app — a simple lookup table that handles the common
# cases without any model. Pace's VLM has to do better than this OR
# handle apps not in this table.
APP_ACTIVITY_HINTS: dict[str, dict[str, str]] = {
    "Mail":     {"default": "reading email", "compose": "composing email"},
    "Messages": {"default": "messaging"},
    "Safari":   {"default": "browsing the web"},
    "Chrome":   {"default": "browsing the web"},
    "Firefox":  {"default": "browsing the web"},
    "Arc":      {"default": "browsing the web"},
    "Xcode":    {"default": "writing code"},
    "Cursor":   {"default": "writing code"},
    "VS Code":  {"default": "writing code"},
    "Terminal": {"default": "running commands"},
    "Pages":    {"default": "writing a document"},
    "Numbers":  {"default": "editing a spreadsheet"},
    "Keynote":  {"default": "preparing a presentation"},
    "Notes":    {"default": "taking notes"},
    "Notion":   {"default": "writing in a workspace"},
    "Slack":    {"default": "messaging your team"},
    "Discord":  {"default": "chatting on discord"},
    "Zoom":     {"default": "in a video call"},
    "Spotify":  {"default": "listening to music"},
    "Apple Music": {"default": "listening to music"},
    "Photos":   {"default": "looking at photos"},
    "Calendar": {"default": "checking your calendar"},
    "Reminders":{"default": "managing reminders"},
    "Figma":    {"default": "designing"},
    "Lightroom":{"default": "editing photos"},
    "Finder":   {"default": "browsing files"},
}


def _parse_element_line(line: str) -> dict | None:
    """[N] role|x,y|label|text"""
    m = re.match(r"\[(\d+)\]\s+([^|]+)\|([\d,]+)\|([^|]*)\|(.*)", line.strip())
    if not m:
        return None
    return {
        "id": int(m.group(1)),
        "role": m.group(2).strip(),
        "pos": m.group(3),
        "label": m.group(4).strip(),
        "text": m.group(5).strip(),
    }


def parse_fixture(text: str) -> dict:
    """Parse a fm-vlm-fixture-v1 .txt file."""
    out = {
        "user": "",
        "app_frontmost": "",
        "ax_blind": False,           # True iff AX tree empty for an interactive screen
        "ax_tree": [],
        "ocr_text": "",
        "expects": {},
    }
    section = None  # 'ax' or 'ocr' or None
    ocr_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            section = None
            continue
        if stripped.startswith("USER:"):
            out["user"] = stripped[len("USER:"):].strip(); section = None
        elif stripped.startswith("APP_FRONTMOST:"):
            out["app_frontmost"] = stripped[len("APP_FRONTMOST:"):].strip(); section = None
        elif stripped.startswith("AX_BLIND:"):
            out["ax_blind"] = "true" in stripped.lower(); section = None
        elif stripped.startswith("AX_TREE:"):
            section = "ax"
        elif stripped.startswith("OCR_TEXT:"):
            section = "ocr"
        elif stripped.startswith(("EXPECT_", "SPOKEN_")):
            k, _, v = stripped.partition(":")
            out["expects"][k.strip()] = v.strip()
            section = None
        elif section == "ax":
            el = _parse_element_line(stripped)
            if el is not None:
                out["ax_tree"].append(el)
        elif section == "ocr":
            ocr_lines.append(line)
    out["ocr_text"] = "\n".join(ocr_lines).strip()
    return out


# ----------------- Rule-based "VLM" -----------------


def fake_pace_vlm(fx: dict) -> dict:
    """Return structured response: {activity, app, elements, spoken}."""
    user = fx["user"].lower()
    app = fx["app_frontmost"]
    ax = fx["ax_tree"]
    ocr = fx["ocr_text"]

    # 1. App identity is free.
    app_label = app or "(unknown app)"

    # 2. Activity heuristic by app.
    activity = "(unknown activity)"
    if app in APP_ACTIVITY_HINTS:
        activity = APP_ACTIVITY_HINTS[app]["default"]
        # crude refinement: if OCR contains compose-y signals, switch to "composing"
        if app == "Mail":
            ocr_lower = ocr.lower()
            if any(s in ocr_lower for s in ("to:", "subject:", "compose", "draft")):
                activity = APP_ACTIVITY_HINTS[app].get("compose", activity)

    # 3. Elements come from AX directly. If AX is blind, we have nothing
    #    (this is the "VLM adds value here" case — leave empty).
    elements_summary: list[str] = []
    if not fx["ax_blind"]:
        for el in ax[:10]:  # cap; Pace VLM also caps
            elements_summary.append(f"{el['label']} ({el['role']})")

    # 4. Spoken response composition (intent-aware).
    spoken = ""
    user_lower = user
    if any(q in user_lower for q in ("what am i doing", "what's on", "what do you see", "describe")):
        if elements_summary:
            spoken = f"you're {activity} in {app_label}; i can see {len(ax)} elements"
        else:
            spoken = f"you're {activity} in {app_label}"
    elif any(q in user_lower for q in ("read", "what does it say", "what's the text")):
        if ocr:
            # Take first ~40 words of OCR.
            words = ocr.split()
            spoken = " ".join(words[:40])
        else:
            spoken = "i don't see any text on screen right now"
    elif any(q in user_lower for q in ("which app", "what app", "where am i")):
        spoken = f"you're in {app_label}"
    elif user_lower.startswith(("click", "tap", "press", "open", "hit")) and ax:
        # element grounding via substring match (same logic as planner FakePace)
        target = re.sub(r"\b(click|tap|press|open|hit|the|a|an)\b", " ", user_lower).strip()
        target_words = set(re.findall(r"\b\w+\b", target))
        best, best_score = None, 0
        for el in ax:
            label_words = set(re.findall(r"\b\w+\b", el["label"].lower()))
            overlap = len(label_words & target_words - {"button", "field", "menu", "icon"})
            if overlap > best_score:
                best, best_score = el, overlap
        if best and best_score >= 1:
            spoken = f"clicking the {best['label']}"
        else:
            spoken = "i can't see that here"
    else:
        # Default: describe state.
        spoken = f"you're {activity} in {app_label}"

    return {
        "activity": activity,
        "app": app_label,
        "elements": elements_summary,
        "spoken": spoken,
    }


# ----------------- Scoring -----------------


def score_response(resp: dict, expects: dict, fx: dict) -> tuple[bool, list[str]]:
    fails: list[str] = []
    spoken = resp.get("spoken", "")

    if "EXPECT_ACTIVITY" in expects:
        want = expects["EXPECT_ACTIVITY"].lower()
        got = resp.get("activity", "").lower()
        if want not in got and got not in want:
            fails.append(f"activity: got {got!r} want {want!r}")

    if "EXPECT_APP" in expects:
        want = expects["EXPECT_APP"].lower()
        got = resp.get("app", "").lower()
        if want != got and want not in got:
            fails.append(f"app: got {got!r} want {want!r}")

    if "EXPECT_ELEMENTS_MIN" in expects:
        want = int(expects["EXPECT_ELEMENTS_MIN"])
        got = len(resp.get("elements", []))
        if got < want:
            fails.append(f"elements: got {got} want >= {want}")

    if "SPOKEN_MUST_CONTAIN" in expects:
        for needle in expects["SPOKEN_MUST_CONTAIN"].split(","):
            n = needle.strip().lower()
            if n and n not in spoken.lower():
                fails.append(f"spoken missing {n!r}")

    if "SPOKEN_MUST_NOT_CONTAIN" in expects:
        for needle in expects["SPOKEN_MUST_NOT_CONTAIN"].split(","):
            n = needle.strip().lower()
            if n and n in spoken.lower():
                fails.append(f"spoken contains forbidden {n!r}")

    if "SPOKEN_MUST_MATCH_REGEX" in expects:
        if not re.search(expects["SPOKEN_MUST_MATCH_REGEX"], spoken):
            fails.append(f"spoken does not match regex {expects['SPOKEN_MUST_MATCH_REGEX']!r}")

    return len(fails) == 0, fails


def run(fixtures_dir: Path, verbose: bool = False) -> dict:
    fxs = sorted(p for p in fixtures_dir.glob("*.txt"))
    results = []
    for path in fxs:
        fx = parse_fixture(path.read_text())
        resp = fake_pace_vlm(fx)
        passed, fails = score_response(resp, fx["expects"], fx)
        results.append({"fixture": path.stem, "passed": passed, "response": resp,
                          "fails": fails})
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {path.stem}")
        if verbose or not passed:
            print(f"    spoken: {resp['spoken'][:160]}")
            for r in fails:
                print(f"    ✗ {r}")

    n_pass = sum(1 for r in results if r["passed"])
    n = len(results)
    print(f"\n=== FakePaceVLM baseline: {n_pass}/{n}  ({100*n_pass/n:.1f}%) ===")
    return {"n_pass": n_pass, "n_total": n, "results": results}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures-dir", type=Path,
                     default=Path("/Users/sarthak/Desktop/fleet/clickyLocal/evals/fm-vlm-fixtures-v1"))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    run(args.fixtures_dir, verbose=args.verbose)
