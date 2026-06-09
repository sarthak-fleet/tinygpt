#!/usr/bin/env python3
"""build-v11-seed-jsonl.py — convert the 60 fm-fixtures-{oos,ambig,
destructive} files into v11 seed training rows.

Each row matches the v10 corpus shape:
  {"instruction": "...", "response": "<json string>"}

Where response is a JSON string of:
  {spokenText, intent, payload}

Intent enum extended from v10's 2 classes ("action", "answer") to 5:
  - "action"               unchanged from v10
  - "answer"               unchanged from v10
  - "out_of_scope"         NEW — refuse cleanly, payload.reason
  - "clarify"              NEW — ask back, payload.question + topic
  - "confirm_destructive"  NEW — flag before firing, payload.action + target

Output: ~/.cache/tinygpt/datasets/pace-v11-seed.jsonl
Total rows: 60 (30 OOS + 20 ambig + 10 destructive)

These are the BASE seeds for v11 training; merge with v10's 404-row
happy-path corpus + the eventual hand-augmented additional 90 (per
docs/prds/pace-planner-v11-training-data.md). For initial training
they alone won't hit the ship gate, but they bootstrap the new intent
classes.
"""
import json
import re
from pathlib import Path

PACE_EVAL = Path("/Users/sarthak/Desktop/fleet/pace/evals")
OUT_FILE = Path.home() / ".cache/tinygpt/datasets/pace-v11-seed.jsonl"


# ---- parser (mirrors eval_pace_unhappy.parse_fixture) ----------------------
def parse_fixture(text: str) -> dict:
    fx = {
        "user": "",
        "elements": [],
        "expect_intent": None,
        "expect_clarify_topic": None,
        "expect_confirm_target": None,
        "reason": None,
    }
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("USER:"):
            fx["user"] = line[len("USER:"):].strip()
        elif line.startswith("ELEMENT:"):
            body = line[len("ELEMENT:"):].strip()
            m = re.match(r"\[(\d+)\]\s+([^|]+)\|([^|]+)\|([^|]+)\|(.*)", body)
            if m:
                fx["elements"].append({
                    "id": int(m.group(1)),
                    "role": m.group(2).strip(),
                    "pos":  m.group(3).strip(),
                    "label": m.group(4).strip(),
                    "text": m.group(5).strip(),
                })
        elif line.startswith("EXPECT_INTENT:"):
            fx["expect_intent"] = line.split(":", 1)[1].strip()
        elif line.startswith("EXPECT_CLARIFY_TOPIC:"):
            fx["expect_clarify_topic"] = line.split(":", 1)[1].strip()
        elif line.startswith("EXPECT_CONFIRM_TARGET:"):
            fx["expect_confirm_target"] = line.split(":", 1)[1].strip()
        elif line.startswith("REASON:"):
            fx["reason"] = line.split(":", 1)[1].strip()
    return fx


# ---- instruction builder (mirrors v10 training shape) ----------------------
def build_instruction(fx: dict) -> str:
    parts = []
    if fx["elements"]:
        parts.append("on-screen elements:")
        for el in fx["elements"]:
            parts.append(f"[{el['id']}] {el['role']}|{el['pos']}|{el['label']}|{el['text']}")
        parts.append("")
    parts.append(f"user said: {fx['user']}")
    return "\n".join(parts)


# ---- per-class response templates -----------------------------------------
# These hand-crafted templates produce natural Pace voice (lowercase, casual,
# warm). Tone is "i can't / let's confirm / which one did you mean" — never
# robotic, never apologetic-spam.
OOS_VOICE_BY_SLUG = {
    # cloud knowledge
    "cloud-weather-today":   "i can't check weather. you'd want a weather app for that.",
    "cloud-news-headlines":  "i can't pull news for you. open safari or news for that.",
    "cloud-stock-price":     "i can't look up stock prices. try stocks app or a browser.",
    "cloud-trivia-question": "that's a knowledge question, not a mac action. i can't answer it.",
    "cloud-currency-rate":   "i can't do currency conversion. open calculator or search the web.",
    "cloud-define-word":     "i can't define words. spotlight or dictionary can.",
    "cloud-time-in-city":    "i can't check time zones. world clock app does that.",
    "cloud-math-problem":    "i can't compute. spotlight does math, try that.",
    # external services
    "ext-order-uber":         "i can't order rides. uber app handles that.",
    "ext-book-flight":        "i can't book flights. try the airline's site in safari.",
    "ext-post-tweet":         "i can't post to social media. open the app yourself.",
    "ext-order-pizza":        "i can't place food orders. try the restaurant's app.",
    "ext-call-uber-friend":   "i can't dispatch rides to other people.",
    "ext-shazam-song":        "i can't identify songs. shazam can.",
    "ext-translate-spanish":  "i can't translate. translate app handles that.",
    # non-mac devices
    "dev-iphone-silent":      "i only control your mac, not your phone.",
    "dev-turn-off-lights":    "i can't control smart home devices.",
    "dev-thermostat":         "i can't control thermostats — i'm just on this mac.",
    "dev-tv-channel":         "i don't control your tv.",
    "dev-watch-timer":        "i can't control your watch from here.",
    # conversational
    "conv-tell-joke":         "i'm here to do mac stuff, not chat.",
    "conv-meaning-of-life":   "that's not something i can help with.",
    "conv-how-are-you":       "i'm here to help you, what do you need?",
    "conv-sentient":          "i'm just pace. i help you with your mac.",
    # monitoring
    "mon-notify-when":        "i can't watch for events in the background.",
    "mon-track-spending":     "i can't track stuff for you over time.",
    "mon-remind-when-arrive": "i can't trigger on your location.",
    # recall
    "recall-yesterday":            "i don't keep history of what you did.",
    "recall-last-conversation":    "i don't remember past conversations.",
    "recall-clipboard-history":    "i can only see what's on your clipboard right now.",
}

AMBIG_VOICE_BY_SLUG = {
    # pronoun-without-referent — reference what's likely on screen
    "pronoun-send-it":       "what do you want to send?",
    "pronoun-open-that":     "which one did you mean?",
    "pronoun-play-this":     "what should i play?",
    "pronoun-close-them":    "which windows do you want to close?",
    "pronoun-share-it":      "what do you want to share with sarah?",
    # missing recipient
    "recipient-send-message": "who do you want to send the message to?",
    "recipient-share-link":   "who do you want to share the link with?",
    "recipient-email-doc":    "who do you want to email this to?",
    "recipient-text-someone": "who do you want to text?",
    # multi-element
    "multi-which-button":   "which button did you mean — save, cancel, or submit?",
    "multi-which-email":    "which email did you mean — the one from alice, bob, or carol?",
    "multi-which-tab":      "which tab do you want to close?",
    "multi-which-document": "which document did you mean?",
    # time / qty
    "time-remind-later":    "when do you want me to remind you?",
    "time-schedule-meeting":"when do you want the meeting?",
    "time-set-alarm":       "what time should the alarm be?",
    "qty-scroll-down":      "how far do you want to scroll?",
    # subject content
    "subject-write-email":  "who do you want to email?",
    "subject-make-event":   "when's the event?",
    "subject-take-note":    "what should the note say?",
}

DESTRUCT_VOICE_BY_SLUG = {
    "delete-all-email":     ("delete every email in your inbox", "Mail.deleteAll"),
    "empty-trash":          ("empty the trash", "Finder.emptyTrash"),
    "force-quit-finder":    ("force quit finder", "App.forceQuit"),
    "erase-document":       ("erase this document", "Doc.erase"),
    "delete-folder":        ("delete the downloads folder", "Finder.deleteFolder"),
    "uninstall-app":        ("uninstall this app", "App.uninstall"),
    "delete-meeting":       ("cancel and delete the meeting", "Cal.deleteEvent"),
    "discard-all-changes":  ("discard all unsaved changes", "Doc.discardChanges"),
    "wipe-downloads":       ("wipe your downloads folder", "Finder.wipeFolder"),
    "delete-photo-library": ("delete your entire photo library", "Photos.deleteLibrary"),
}


# ---- response builders ----------------------------------------------------
def build_oos_response(fx: dict, slug: str) -> str:
    spoken = OOS_VOICE_BY_SLUG.get(slug, "i can't do that on your mac.")
    reason = fx.get("reason") or "no pace action covers this"
    return json.dumps({
        "spokenText": spoken,
        "intent": "out_of_scope",
        "payload": {"reason": reason},
    }, ensure_ascii=False)


def build_clarify_response(fx: dict, slug: str) -> str:
    spoken = AMBIG_VOICE_BY_SLUG.get(slug, "which one did you mean?")
    topic = fx.get("expect_clarify_topic") or "your request"
    return json.dumps({
        "spokenText": spoken,
        "intent": "clarify",
        "payload": {"question": spoken, "topic": topic},
    }, ensure_ascii=False)


def build_confirm_destructive_response(fx: dict, slug: str) -> str:
    target_desc, action_name = DESTRUCT_VOICE_BY_SLUG.get(
        slug, ("do that", "Generic.destructive"))
    spoken = f"that will {target_desc} — say yes to confirm."
    target = fx.get("expect_confirm_target") or target_desc
    return json.dumps({
        "spokenText": spoken,
        "intent": "confirm_destructive",
        "payload": {"action": action_name, "target": target},
    }, ensure_ascii=False)


# ---- main -----------------------------------------------------------------
def main():
    rows = []
    seen_slugs = set()

    for d, builder in [
        (PACE_EVAL / "fm-fixtures-oos", build_oos_response),
        (PACE_EVAL / "fm-fixtures-ambig", build_clarify_response),
        (PACE_EVAL / "fm-fixtures-destructive", build_confirm_destructive_response),
    ]:
        files = sorted(d.glob("*.txt"))
        for fp in files:
            slug = fp.stem
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            fx = parse_fixture(fp.read_text())
            if not fx["expect_intent"]:
                continue
            inst = build_instruction(fx)
            resp = builder(fx, slug)
            rows.append({
                "instruction": inst,
                "response": resp,
                "_meta": {"source_fixture": str(fp.relative_to(PACE_EVAL.parent)),
                          "intent_class": fx["expect_intent"]},
            })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"wrote {len(rows)} rows to {OUT_FILE}")

    # quick distribution check
    by_class = {}
    for r in rows:
        c = r["_meta"]["intent_class"]
        by_class[c] = by_class.get(c, 0) + 1
    for c, n in sorted(by_class.items()):
        print(f"  {c:24s} {n}")


if __name__ == "__main__":
    main()
