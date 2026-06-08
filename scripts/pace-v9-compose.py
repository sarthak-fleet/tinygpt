#!/usr/bin/env python3
"""Build Pace v9 SFT corpus: v8 rows (with bodyText="" backfilled) + ~50
compose/draft examples that exercise the new bodyText field.

The v9 schema adds a fourth field:
    {"spokenText","pointAtLabel","clickLabel","bodyText"}

For every non-compose case the label-emitter behavior is unchanged; bodyText
is just "". For draft/compose cases, the model emits the actual message text
in bodyText. The Pace runtime streams bodyText into the open compose window
(mailto:, AX setValue, or keystroke) as chunks arrive — so the user sees the
message being typed.

The examples here are HAND-CRAFTED. Teacher-labeling compose intents would
inherit the teacher's tone (formal, hedgy); we want the Pace voice (lowercase,
warm, short, written-for-ear). Curated examples set the model's voice
explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V8 = Path.home() / ".cache/tinygpt/datasets/pace-v8-sft.jsonl"
DEFAULT_OUT = Path.home() / ".cache/tinygpt/datasets/pace-v9-sft.jsonl"


def _row(user: str, elements: list[str], response: dict) -> dict:
    el_lines = [f"[{i}] {e}" for i, e in enumerate(elements)]
    parts = []
    if elements:
        parts.append("on-screen elements:")
        parts.extend(el_lines)
        parts.append("")
    parts.append(f"user said: {user}")
    instruction = "\n".join(parts)
    return {
        "instruction": instruction,
        "response": json.dumps(response, separators=(",", ":"), ensure_ascii=False),
    }


def _compose(spoken: str, body: str) -> dict:
    return {"spokenText": spoken, "pointAtLabel": "", "clickLabel": "", "bodyText": body}


# =====================================================================
# Common screen contexts that show user is not actively composing yet
# (Pace launches compose in parallel; the screen reflects whatever was open)
# =====================================================================

DESKTOP_GENERIC = [
    "dock_icon|24,1000|Mail|Inbox",
    "dock_icon|72,1000|Calendar|Schedule",
    "dock_icon|120,1000|Safari|Browse",
    "dock_icon|168,1000|Notes|Notes app",
]

SLACK_OPEN = [
    "menu_bar|600,12|Slack|Slack workspace",
    "channel_name|140,80|#engineering",
    "message|140,200|Priya|did the build pass?",
]

MAIL_INBOX_OPEN = [
    "window_title|400,30|Inbox — Gmail",
    "row|200,120|John Doe|Re: project status",
    "row|200,150|Marcus|Lunch tomorrow?",
    "button|750,30|Compose|new email",
]

NOTES_OPEN = [
    "window_title|400,30|Notes",
    "row|200,120|Shopping list",
    "row|200,150|Project ideas",
]


# =====================================================================
# Draft / compose examples — the v9 unlock
# =====================================================================

COMPOSE_ROWS = [
    # --- Email: explicit recipient + topic ---
    _row(
        "draft a mail to john about the project status",
        DESKTOP_GENERIC,
        _compose(
            "drafting an email to john",
            "wanted to give you a quick update on where we are. the planner specialist shipped this week and we just got the screen-reader half running end to end. next milestone is the demo build by end of week. let me know if you want a walkthrough before then.",
        ),
    ),
    _row(
        "send an email to priya about the design review",
        DESKTOP_GENERIC,
        _compose(
            "writing a message to priya about the design review",
            "thanks for putting time on the calendar today. one thing i wanted to flag ahead of the review — the empty-state flow still feels off and i'd love your read on it before we ship. happy to talk it through whenever works.",
        ),
    ),
    _row(
        "draft an email to my manager about taking friday off",
        DESKTOP_GENERIC,
        _compose(
            "drafting a time-off note to your manager",
            "wanted to give you a heads up that i'm planning to take friday off — nothing urgent, just need the long weekend. i'll wrap up the open items by thursday eod and slack the team if anything comes up. let me know if that works.",
        ),
    ),
    _row(
        "write an email to marcus thanking him for the intro",
        DESKTOP_GENERIC,
        _compose(
            "writing a thank-you note to marcus",
            "really appreciate the intro to sara — we had a great chat yesterday and she's exactly the right person for what we're trying to build. owe you one.",
        ),
    ),
    _row(
        "follow up with the recruiter from acme corp",
        DESKTOP_GENERIC,
        _compose(
            "drafting a follow-up to the acme recruiter",
            "wanted to circle back on our conversation last week. i've had time to think it over and i'm still interested in moving forward — happy to schedule the next round whenever your team has availability.",
        ),
    ),
    _row(
        "reply to the last email from sarah",
        MAIL_INBOX_OPEN,
        _compose(
            "drafting a reply to sarah",
            "thanks for sending this over. taking a closer look now and will get back to you with thoughts by tomorrow. let me know if there's anything specific you want me to weigh in on first.",
        ),
    ),
    _row(
        "email the team about the deployment going out tonight",
        DESKTOP_GENERIC,
        _compose(
            "drafting an email to the team about tonight's deployment",
            "heads up — pushing the new release tonight around 9pm. should be transparent to users but i'll be online if anything breaks. ping me on slack if you see anything weird in the dashboards tomorrow morning.",
        ),
    ),
    _row(
        "draft a thank you note to the catering team",
        DESKTOP_GENERIC,
        _compose(
            "writing a thank-you to the catering team",
            "just wanted to say thanks for everything yesterday — the food was great and people are still talking about it. really appreciate how smoothly you ran the whole thing.",
        ),
    ),
    _row(
        "send a message to my parents that i landed safely",
        DESKTOP_GENERIC,
        _compose(
            "letting your parents know you landed",
            "hey, just landed and through customs. heading to the hotel now. will call you tomorrow morning your time. love you both.",
        ),
    ),
    _row(
        "write an apology email to the client about the delay",
        DESKTOP_GENERIC,
        _compose(
            "drafting an apology to the client",
            "wanted to reach out directly about the timeline slip on your end. we underestimated the integration work and i should have flagged it sooner. here's where things actually stand — i'll have a corrected delivery date and a credit on this month's invoice by tomorrow morning.",
        ),
    ),
    _row(
        "draft a reply saying yes i can join the meeting tomorrow",
        DESKTOP_GENERIC,
        _compose(
            "confirming you can make the meeting tomorrow",
            "yes i can make it — see you then.",
        ),
    ),
    _row(
        "send a quick note to mom",
        DESKTOP_GENERIC,
        _compose(
            "writing a quick note to mom",
            "hey mom, just thinking of you. hope work's been okay this week. call you over the weekend.",
        ),
    ),

    # --- Vague topic, model has to invent something plausible ---
    _row(
        "send the team a friday update",
        DESKTOP_GENERIC,
        _compose(
            "writing the team's friday update",
            "quick wrap on the week. we got the planner specialist landed and the eval moved from 60 to 73 percent. next week is screen-reader and the compose flow. ping me if anything's blocking you over the weekend.",
        ),
    ),
    _row(
        "write a short note to my landlord about the leaking faucet",
        DESKTOP_GENERIC,
        _compose(
            "writing a note to your landlord about the faucet",
            "hi — the kitchen sink has been leaking since yesterday, slow drip from under the faucet. happy to be home anytime this week if you want to send someone by. thanks.",
        ),
    ),
    _row(
        "draft a slack message to my team that i'm running late",
        SLACK_OPEN,
        _compose(
            "letting the team know you're running late",
            "running about ten minutes late — start without me, i'll catch up.",
        ),
    ),

    # --- Notes / non-email compose ---
    _row(
        "add a note about the dentist appointment next thursday",
        NOTES_OPEN,
        _compose(
            "adding a note for the dentist appointment",
            "dentist — next thursday at 2pm. dr. patel's office. bring insurance card.",
        ),
    ),
    _row(
        "write a note about the book recommendations from dinner",
        NOTES_OPEN,
        _compose(
            "writing down the book recs from dinner",
            "books from dinner with maya:\nthinking fast and slow — kahneman\nhow to take smart notes — ahrens\nthe order of time — rovelli",
        ),
    ),

    # --- Edge cases / negative-bodyText (case A,B,C with bodyText="") ---
    _row(
        "what does mail mean",
        DESKTOP_GENERIC,
        {
            "spokenText": "mail is apple's built-in email app — you use it to send and receive messages across your accounts.",
            "pointAtLabel": "",
            "clickLabel": "",
            "bodyText": "",
        },
    ),
    _row(
        "open mail",
        DESKTOP_GENERIC,
        {
            "spokenText": "opening mail",
            "pointAtLabel": "Mail",
            "clickLabel": "Mail",
            "bodyText": "",
        },
    ),
    _row(
        "compose",
        MAIL_INBOX_OPEN,
        # Ambiguous bare "compose" — without a target, treat as click on the
        # Compose button (case B), not as a draft-this-message intent.
        {
            "spokenText": "opening a new email",
            "pointAtLabel": "Compose",
            "clickLabel": "Compose",
            "bodyText": "",
        },
    ),
]


# =====================================================================
# Assembly: take v8 corpus, backfill bodyText="" for old rows, add new ones
# =====================================================================


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def backfill_body(row: dict) -> dict:
    """Add bodyText='' to the response so the schema matches v9."""
    try:
        resp = json.loads(row["response"])
    except (json.JSONDecodeError, KeyError):
        return row
    if "bodyText" not in resp:
        resp["bodyText"] = ""
    row["response"] = json.dumps(resp, separators=(",", ":"), ensure_ascii=False)
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input", type=Path, default=DEFAULT_V8)
    p.add_argument("--out", dest="output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"missing input corpus: {args.input}")

    base = load_jsonl(args.input)
    base = [backfill_body(r) for r in base]
    merged = base + COMPOSE_ROWS

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"base rows (v8 with bodyText backfilled): {len(base)}")
    print(f"new compose/draft rows: {len(COMPOSE_ROWS)}")
    print(f"wrote {len(merged)} rows → {args.output}")


if __name__ == "__main__":
    main()
