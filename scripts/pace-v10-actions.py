#!/usr/bin/env python3
"""Build Pace v10 SFT corpus: parameterized actions schema.

v10 schema:
    {"spokenText": str, "intent": "action"|"dictate"|"edit"|"answer",
     "payload": {...intent-specific shape...}}

Sources:
  1. v9 dataset converted to v10 shape (case A,B,C non-compose stay as
     intent=answer or intent=action(AX.press) ; case D compose becomes
     intent=action(Mail.draft)).
  2. Hand-crafted seeds across all 12 v1 actions (this file).
  3. (Future) Teacher-generated variations from Qwen3-14B at 10× per seed.

Run:
    python3 scripts/pace-v10-actions.py \\
      --v9-in ~/.cache/tinygpt/datasets/pace-v9-sft.jsonl \\
      --out ~/.cache/tinygpt/datasets/pace-v10-sft.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_V9 = Path.home() / ".cache/tinygpt/datasets/pace-v9-sft.jsonl"
DEFAULT_OUT = Path.home() / ".cache/tinygpt/datasets/pace-v10-sft.jsonl"


# =====================================================================
# Row builder
# =====================================================================


def _row(user: str, elements: list[str], spoken: str, intent: str, payload: dict) -> dict:
    """Build a single v10 SFT row."""
    el_lines = [f"[{i}] {e}" for i, e in enumerate(elements)]
    parts = []
    if elements:
        parts.append("on-screen elements:")
        parts.extend(el_lines)
        parts.append("")
    parts.append(f"user said: {user}")
    response = {"spokenText": spoken, "intent": intent, "payload": payload}
    return {
        "instruction": "\n".join(parts),
        "response": json.dumps(response, separators=(",", ":"), ensure_ascii=False),
    }


def _action(spoken: str, name: str, args: dict) -> tuple[str, str, dict]:
    return spoken, "action", {"name": name, "args": args}


def _answer(spoken: str) -> tuple[str, str, dict]:
    return spoken, "answer", {"text": ""}


def _dictate(spoken: str, text: str) -> tuple[str, str, dict]:
    return spoken, "dictate", {"text": text}


def _edit(spoken: str, reference: str, transform: str, argument: str = "") -> tuple[str, str, dict]:
    p = {"reference": reference, "transform": transform}
    if argument:
        p["argument"] = argument
    return spoken, "edit", p


# =====================================================================
# Common screen contexts
# =====================================================================

DOCK_GENERIC = [
    "dock_icon|24,1000|Mail|Inbox",
    "dock_icon|72,1000|Calendar|Schedule",
    "dock_icon|120,1000|Safari|Browse",
    "dock_icon|168,1000|Notes|Note app",
    "dock_icon|216,1000|Reminders|Lists",
    "dock_icon|264,1000|Slack|Chat",
]

FORM_SCREEN = [
    "text_field|400,200|email|jane@example.com",
    "text_field|400,250|name|Jane Doe",
    "button|400,400|Submit|submit form",
    "button|500,400|Cancel|cancel",
]

LONG_DOC = [
    "text_area|240,300|editor|a long document with several paragraphs of text",
    "button|600,40|Save|save document",
]

NOTES_OPEN = [
    "window_title|400,30|Notes",
    "row|200,120|Shopping list",
    "row|200,150|Project ideas",
]


# =====================================================================
# Hand-crafted seeds — ~10 per action minimum for v1 smoke
# =====================================================================


def build_seeds() -> list[dict]:
    rows = []

    def add(user, elements, spec):
        spoken, intent, payload = spec
        rows.append(_row(user, elements, spoken, intent, payload))

    # ---------------- AX.press ----------------
    add("click save", LONG_DOC,
        _action("clicking save", "AX.press", {"target": "Save"}))
    add("press the submit button", FORM_SCREEN,
        _action("pressing submit", "AX.press", {"target": "Submit"}))
    add("hit cancel", FORM_SCREEN,
        _action("cancelling", "AX.press", {"target": "Cancel"}))
    add("open mail", DOCK_GENERIC,
        _action("opening mail", "AX.press", {"target": "Mail"}))
    add("open the shopping list", NOTES_OPEN,
        _action("opening the shopping list", "AX.press", {"target": "Shopping list"}))
    add("tap on slack", DOCK_GENERIC,
        _action("opening slack", "AX.press", {"target": "Slack"}))
    add("press calendar", DOCK_GENERIC,
        _action("opening calendar", "AX.press", {"target": "Calendar"}))
    add("choose project ideas", NOTES_OPEN,
        _action("opening project ideas", "AX.press", {"target": "Project ideas"}))

    # ---------------- AX.scroll ----------------
    add("scroll down", LONG_DOC,
        _action("scrolling down", "AX.scroll", {"direction": "down"}))
    add("go to the top", LONG_DOC,
        _action("scrolling to the top", "AX.scroll", {"direction": "top"}))
    add("scroll up a bit", LONG_DOC,
        _action("scrolling up", "AX.scroll", {"direction": "up", "amount": 2}))
    add("jump to the bottom", LONG_DOC,
        _action("scrolling to the bottom", "AX.scroll", {"direction": "bottom"}))
    add("scroll left", LONG_DOC,
        _action("scrolling left", "AX.scroll", {"direction": "left"}))

    # ---------------- App.launch ----------------
    add("launch xcode", DOCK_GENERIC,
        _action("opening xcode", "App.launch", {"name": "Xcode"}))
    add("open figma", DOCK_GENERIC,
        _action("opening figma", "App.launch", {"name": "Figma"}))
    add("start zoom", DOCK_GENERIC,
        _action("opening zoom", "App.launch", {"name": "Zoom"}))
    add("open vscode", DOCK_GENERIC,
        _action("opening vs code", "App.launch", {"name": "Visual Studio Code"}))
    add("get me chrome", DOCK_GENERIC,
        _action("opening chrome", "App.launch", {"name": "Google Chrome"}))

    # ---------------- App.activate ----------------
    add("switch to safari", DOCK_GENERIC,
        _action("switching to safari", "App.activate", {"name": "Safari"}))
    add("bring up mail", DOCK_GENERIC,
        _action("bringing up mail", "App.activate", {"name": "Mail"}))
    add("focus slack", DOCK_GENERIC,
        _action("focusing slack", "App.activate", {"name": "Slack"}))

    # ---------------- Mail.draft ----------------
    add("draft an email to john about the project status", DOCK_GENERIC,
        _action("drafting an email to john", "Mail.draft", {
            "to": ["__resolve:john"],
            "subject": "project status",
            "body": "wanted to give you a quick update on where we are. the planner specialist shipped this week and we just got the screen-reader half running end to end. next milestone is the demo build by end of week. let me know if you want a walkthrough before then.",
        }))
    add("send an email to priya about design review", DOCK_GENERIC,
        _action("writing a message to priya about the design review", "Mail.draft", {
            "to": ["__resolve:priya"],
            "subject": "design review",
            "body": "thanks for putting time on the calendar today. one thing i wanted to flag ahead of the review — the empty-state flow still feels off and i'd love your read on it before we ship.",
        }))
    add("email marcus thanking him for the intro", DOCK_GENERIC,
        _action("writing a thank-you to marcus", "Mail.draft", {
            "to": ["__resolve:marcus"],
            "subject": "thank you",
            "body": "really appreciate the intro to sara — we had a great chat yesterday and she's exactly the right person for what we're trying to build. owe you one.",
        }))
    add("send a quick note to mom", DOCK_GENERIC,
        _action("writing a quick note to mom", "Mail.draft", {
            "to": ["__resolve:mom"],
            "subject": "",
            "body": "hey mom, just thinking of you. hope work's been okay this week. call you over the weekend.",
        }))
    add("follow up with the recruiter from acme", DOCK_GENERIC,
        _action("drafting a follow-up to acme", "Mail.draft", {
            "to": ["__resolve:acme recruiter"],
            "subject": "follow-up",
            "body": "wanted to circle back on our conversation last week. i've had time to think it over and i'm still interested in moving forward — happy to schedule the next round whenever your team has availability.",
        }))

    # ---------------- Cal.event ----------------
    add("schedule a meeting with priya at 3pm tomorrow", DOCK_GENERIC,
        _action("scheduling a meeting with priya for tomorrow at 3pm", "Cal.event", {
            "title": "Meeting with Priya",
            "start": "tomorrow 3pm",
        }))
    add("add a coffee with mom at 2pm friday", DOCK_GENERIC,
        _action("adding coffee with mom for friday at 2pm", "Cal.event", {
            "title": "Coffee with mom",
            "start": "friday 2pm",
        }))
    add("block calendar for focus time today from 10 to noon", DOCK_GENERIC,
        _action("blocking focus time today, 10 to noon", "Cal.event", {
            "title": "Focus time",
            "start": "today 10am",
            "end": "today 12pm",
        }))
    add("dinner with sarah at 7 friday, the italian place downtown", DOCK_GENERIC,
        _action("adding dinner with sarah for friday at 7", "Cal.event", {
            "title": "Dinner with Sarah",
            "start": "friday 7pm",
            "location": "Italian place downtown",
        }))

    # ---------------- Reminders.add ----------------
    add("remind me to pick up groceries", DOCK_GENERIC,
        _action("adding groceries reminder", "Reminders.add", {
            "title": "Pick up groceries",
        }))
    add("remind me to call john on friday", DOCK_GENERIC,
        _action("reminder to call john friday", "Reminders.add", {
            "title": "Call John",
            "due": "friday",
        }))
    add("add a high priority reminder to submit the form by tonight", DOCK_GENERIC,
        _action("adding high-priority form reminder for tonight", "Reminders.add", {
            "title": "Submit the form",
            "due": "tonight",
            "priority": "high",
        }))
    add("remind me about the dentist appointment thursday at 2", DOCK_GENERIC,
        _action("adding dentist reminder for thursday at 2pm", "Reminders.add", {
            "title": "Dentist appointment",
            "due": "thursday 2pm",
        }))

    # ---------------- Notes.create ----------------
    add("create a note about the dinner plans", DOCK_GENERIC,
        _action("creating a note about the dinner plans", "Notes.create", {
            "title": "Dinner plans",
            "body": "dinner plans — need to finalize restaurant, time, and guest list.",
        }))
    add("make a note titled ideas with some bullet points", DOCK_GENERIC,
        _action("creating a note titled ideas", "Notes.create", {
            "title": "Ideas",
            "body": "- \n- \n- ",
        }))
    add("write down what i'm thinking about the project pivot", DOCK_GENERIC,
        _action("creating a project pivot note", "Notes.create", {
            "title": "Project pivot thinking",
            "body": "main question: do we narrow the v1 scope to just voice and executor, or include rag and vision? leaning narrow — faster ship, can iterate publicly. risk: positioning gets muddled if we expand later.",
        }))

    # ---------------- Shortcut.run ----------------
    add("run my morning routine shortcut", DOCK_GENERIC,
        _action("running morning routine", "Shortcut.run", {"name": "Morning routine"}))
    add("trigger focus mode", DOCK_GENERIC,
        _action("triggering focus mode", "Shortcut.run", {"name": "Focus mode"}))
    add("start the shutdown shortcut", DOCK_GENERIC,
        _action("running shutdown shortcut", "Shortcut.run", {"name": "Shutdown"}))

    # ---------------- Window.snap ----------------
    add("snap left", DOCK_GENERIC,
        _action("snapping left", "Window.snap", {"position": "left"}))
    add("make this fullscreen", DOCK_GENERIC,
        _action("going fullscreen", "Window.snap", {"position": "full"}))
    add("tile right", DOCK_GENERIC,
        _action("tiling right", "Window.snap", {"position": "right"}))
    add("center this window", DOCK_GENERIC,
        _action("centering the window", "Window.snap", {"position": "center"}))
    add("snap top left", DOCK_GENERIC,
        _action("snapping top-left", "Window.snap", {"position": "top-left"}))

    # ---------------- Clipboard.read ----------------
    add("what's in my clipboard", DOCK_GENERIC,
        _action("reading the clipboard", "Clipboard.read", {}))
    add("read what i copied", DOCK_GENERIC,
        _action("reading the clipboard", "Clipboard.read", {}))
    add("what did i copy last", DOCK_GENERIC,
        _action("reading the clipboard", "Clipboard.read", {}))

    # ---------------- AX.setValue ----------------
    add("fill in jane doe for the name", FORM_SCREEN,
        _action("filling in jane doe", "AX.setValue", {"target": "name", "value": "Jane Doe"}))
    add("type my email in the email field", FORM_SCREEN,
        _action("typing your email", "AX.setValue", {"target": "email", "value": "__resolve:user-email"}))

    # ---------------- intent=dictate ----------------
    add("hi sarah just checking in", DOCK_GENERIC,
        _dictate("(dictating)", "Hi Sarah, just checking in."))
    add("let me know when you have a sec", DOCK_GENERIC,
        _dictate("(dictating)", "Let me know when you have a sec."))
    add("here are the three things i'm worried about colon one two three", DOCK_GENERIC,
        _dictate("(dictating)", "Here are the three things I'm worried about: 1) 2) 3)"))

    # ---------------- intent=edit ----------------
    add("make this more formal", LONG_DOC,
        _edit("rewriting more formal", "selection", "formal"))
    add("delete the last sentence", LONG_DOC,
        _edit("deleting the last sentence", "last sentence", "delete"))
    add("turn this into bullet points", LONG_DOC,
        _edit("converting to bullet points", "selection", "bullets"))
    add("shorten that paragraph", LONG_DOC,
        _edit("shortening the paragraph", "this paragraph", "shorter"))
    add("rewrite that more casually", LONG_DOC,
        _edit("rewriting more casual", "selection", "casual"))

    # ---------------- intent=answer (no action needed) ----------------
    add("what's html", [],
        _answer("html is hypertext markup language — the standard way web pages are written. it tells the browser what's a heading, a paragraph, a link, an image, and so on."))
    add("what time is it", DOCK_GENERIC,
        _answer("i don't have a clock specialist yet — check your menu bar."))
    add("describe what i'm looking at", LONG_DOC,
        _answer("you've got a long document open in an editor, with a save button up top."))
    add("how do you work", [],
        _answer("i'm pace. you talk to me, i figure out what you want, and i do it or answer you."))

    # ---------------- intent disambiguation: negative cases ----------------
    # User says compose-ish words but it's NOT a draft request
    add("what's the difference between draft and final", [],
        _answer("a draft is the working version, the final is what you actually send."))
    add("can you send a slack message — wait nevermind", DOCK_GENERIC,
        _answer("got it, holding off."))

    return rows


# =====================================================================
# v9 → v10 conversion
# =====================================================================


def v9_to_v10_row(v9_row: dict) -> dict | None:
    """Convert a v9-schema row to v10 schema.

    Returns None if the v9 row doesn't carry a parseable JSON response
    (those legacy plain-text v5 carry-overs stay in v9 corpus, not v10).
    """
    try:
        resp = json.loads(v9_row["response"])
    except (json.JSONDecodeError, KeyError):
        return None
    if not isinstance(resp, dict):
        return None

    spoken = resp.get("spokenText", "")
    click  = resp.get("clickLabel", "")
    point  = resp.get("pointAtLabel", "")
    body   = resp.get("bodyText", "")

    if body:
        # v9 compose row → v10 Mail.draft (skip — better to regenerate with
        # explicit `to` field in build_seeds() above)
        return None
    if click:
        v10_resp = {
            "spokenText": spoken, "intent": "action",
            "payload": {"name": "AX.press", "args": {"target": click}}
        }
    elif point:
        # v9 case B without click → answer with HUD pointer (lossy: we
        # drop the point info since v10 doesn't have it; the planner can
        # reference visible elements in spokenText if relevant)
        v10_resp = {"spokenText": spoken, "intent": "answer", "payload": {"text": ""}}
    else:
        v10_resp = {"spokenText": spoken, "intent": "answer", "payload": {"text": ""}}

    return {
        "instruction": v9_row["instruction"],
        "response": json.dumps(v10_resp, separators=(",", ":"), ensure_ascii=False),
    }


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v9-in", dest="v9_in", type=Path, default=DEFAULT_V9)
    p.add_argument("--out", dest="output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--no-v9", action="store_true",
                     help="Skip v9 corpus conversion (seeds-only)")
    args = p.parse_args()

    seeds = build_seeds()
    print(f"hand-crafted seeds: {len(seeds)}")

    v9_converted: list[dict] = []
    if not args.no_v9 and args.v9_in.exists():
        v9 = load_jsonl(args.v9_in)
        for r in v9:
            converted = v9_to_v10_row(r)
            if converted is not None:
                v9_converted.append(converted)
        print(f"v9 rows converted: {len(v9_converted)} (out of {len(v9)})")

    merged = v9_converted + seeds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"wrote {len(merged)} v10 rows → {args.output}")
    print()
    print("NEXT: teacher-multiply each seed via Qwen3-14B at LM Studio")
    print("      to reach 5-10k. Blocked on v9 training finishing first")
    print("      (memory pressure — Qwen3-14B + Qwen3-0.6B-DoRA = OOM).")


if __name__ == "__main__":
    main()
