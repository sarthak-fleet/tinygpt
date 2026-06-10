#!/usr/bin/env python3
"""Generate contrastive clarify-discipline training seeds (v11 JSON format).

The clarify failure is universal (every model measured scores ~0-5%): given
"send the draft" with three drafts visible, models pick one instead of
asking. Training data must teach the BOUNDARY, not blanket caution, so every
ambiguous seed has a matched unambiguous twin (one candidate / explicit
target -> act). Over-asking is the known over-correction failure mode.

All scenarios are surface-disjoint from the held-out eval suites
(evals/fm-fixtures-ambig-h2) — verified by the collision check in the
pipeline. Output rows: {"instruction", "response", "_meta"}.

Usage: python3 scripts/build-clarify-seeds.py --out ~/.cache/tinygpt/datasets/clarify-seeds.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def el(i, role, pos, label, text=""):
    return f"[{i}] {role}|{pos}|{label}|{text}"


def instruction(user, elements):
    parts = []
    if elements:
        parts.append("on-screen elements:")
        parts.extend(elements)
        parts.append("")
    parts.append(f"user said: {user}")
    return "\n".join(parts)


def clarify(spoken, question, topic):
    return json.dumps({"spokenText": spoken, "intent": "clarify",
                       "payload": {"question": question, "topic": topic}})


def action(spoken, name, args):
    return json.dumps({"spokenText": spoken, "intent": "action",
                       "payload": {"name": name, "args": args}})


ROWS = []


def add(kind, user, elements, response, scenario):
    ROWS.append({
        "instruction": instruction(user, elements),
        "response": response,
        "_meta": {"intent_class": "clarify" if kind == "ambiguous" else "action",
                  "contrast_pair": scenario, "kind": kind},
    })


# ---- Category 1: multiple matching elements vs exactly one ----
MULTI = [
    ("pdf report", "open the report", "file",
     [("Q3-board-report.pdf", "report"), ("metrics-report.pdf", "report"), ("readme.md", "doc")],
     [("Q3-board-report.pdf", "report"), ("holiday-photos.zip", "archive"), ("readme.md", "doc")]),
    ("slack channel", "post this in the channel", "list_item",
     [("#eng-frontend", "channel"), ("#eng-backend", "channel"), ("#random", "channel")],
     [("#eng-frontend", "channel"), ("Sam (DM)", "dm"), ("Threads", "view")]),
    ("contact card", "call them", "row",
     [("Jordan Lee — mobile", "contact"), ("Jordan Smith — work", "contact"), ("Settings", "menu")],
     [("Jordan Lee — mobile", "contact"), ("Recents", "view"), ("Settings", "menu")]),
    ("photo album", "share the album", "grid_item",
     [("Trip to Goa", "album"), ("Trip to Leh", "album"), ("Screenshots", "album")],
     [("Trip to Goa", "album"), ("Camera", "button"), ("Search", "field")]),
    ("zoom link", "join the call", "row",
     [("Standup — 9:30", "meeting"), ("Architecture sync — 9:30", "meeting"), ("Calendar", "app")],
     [("Standup — 9:30", "meeting"), ("Tomorrow: 1:1", "meeting tomorrow"), ("Calendar", "app")]),
    ("bookmark", "open my saved page", "list_item",
     [("HN — Show thread", "bookmark"), ("Recipe — dal", "bookmark"), ("Banking", "bookmark")],
     [("HN — Show thread", "bookmark"), ("History", "menu"), ("Downloads", "menu")]),
    ("terminal tab", "kill the server in the terminal", "tab",
     [("dev-server :3000", "session"), ("dev-server :8080", "session"), ("logs", "session")],
     [("dev-server :3000", "session"), ("zsh — idle", "session"), ("logs", "session")]),
    ("attachment", "download the attachment", "row",
     [("invoice-march.pdf", "attachment"), ("invoice-april.pdf", "attachment"), ("signature.png", "inline")],
     [("invoice-march.pdf", "attachment"), ("Reply", "button"), ("Archive", "button")]),
]

for scenario, user, role, ambiguous_set, unambiguous_set in MULTI:
    amb_els = [el(i, role, f"{100 + i * 60},200", lab, txt) for i, (lab, txt) in enumerate(ambiguous_set)]
    first_two = f"{ambiguous_set[0][0]} or {ambiguous_set[1][0]}"
    add("ambiguous", user, amb_els,
        clarify(f"i see more than one — {first_two}?",
                f"which one: {first_two}?", "target"), scenario)
    un_els = [el(i, role, f"{100 + i * 60},200", lab, txt) for i, (lab, txt) in enumerate(unambiguous_set)]
    add("unambiguous", user, un_els,
        action(f"on it", "AX.press", {"target": unambiguous_set[0][0]}), scenario)

# ---- Category 2: missing essential argument vs argument present ----
MISSING = [
    ("mail-recipient", "email the budget sheet", "email the budget sheet to finance@acme.com",
     "who should i send it to?", "recipient",
     "Mail.draft", {"to": ["finance@acme.com"], "subject": "Budget sheet"}),
    ("event-time", "schedule a sync with the design team", "schedule a sync with the design team thursday at 3pm",
     "when should i put it?", "time",
     "Cal.event", {"title": "Sync with design team", "start": "thursday 3pm"}),
    ("reminder-time", "remind me to renew the passport", "remind me tomorrow morning to renew the passport",
     "for when?", "time",
     "Reminders.add", {"title": "Renew the passport", "due": "tomorrow morning"}),
    ("note-content", "start a note for the offsite", "start a note for the offsite: agenda, budget, travel",
     "what should go in it?", "content",
     "Notes.create", {"title": "Offsite", "body": "agenda, budget, travel"}),
    ("message-body", "reply to dana", "reply to dana saying i'll review it tonight",
     "what should i say?", "content",
     "Mail.draft", {"to": ["__resolve:dana"], "body": "I'll review it tonight."}),
    ("timer-duration", "start a countdown", "start a twenty minute countdown",
     "how long?", "duration",
     "Shortcut.run", {"name": "Timer", "input": "20 minutes"}),
]

for scenario, amb_user, clear_user, q, topic, name, args in MISSING:
    add("ambiguous", amb_user, [],
        clarify(q, q, topic), scenario)
    add("unambiguous", clear_user, [],
        action("done", name, args), scenario)

# ---- Category 3: pronoun with no referent vs visible referent ----
PRONOUN = [
    ("archive-it", "archive it", "row",
     [], [("Re: contract renewal", "selected email")]),
    ("forward-that", "forward that to legal", "row",
     [], [("NDA draft v3", "selected email")]),
    ("rename-this", "rename this to final", "file",
     [], [("draft-v7.key", "selected file")]),
    ("pin-it", "pin it to the top", "list_item",
     [], [("Sprint goals", "selected note")]),
    ("duplicate-that", "duplicate that slide", "thumbnail",
     [], [("Slide 12 — roadmap", "selected slide")]),
]

for scenario, user, role, _, referent_set in PRONOUN:
    add("ambiguous", user, [],
        clarify("which one do you mean?", "which item should i use?", "target"), scenario)
    ref_els = [el(0, role, "300,300", referent_set[0][0], referent_set[0][1])]
    add("unambiguous", user, ref_els,
        action("done", "AX.press", {"target": referent_set[0][0]}), scenario)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for row in ROWS:
            f.write(json.dumps(row) + "\n")
    amb = sum(1 for r in ROWS if r["_meta"]["kind"] == "ambiguous")
    print(f"wrote {len(ROWS)} seeds ({amb} clarify / {len(ROWS) - amb} action twins) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main()) if False else main()
