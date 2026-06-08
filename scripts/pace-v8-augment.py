#!/usr/bin/env python3
"""Build Pace v8 SFT data: v5's 248 rows + ~50 hand-crafted v2-targeted
examples covering the failure modes the previous LoRAs missed.

Failure axes from docs/learn/eval-matrix-2026-06-08.md:
  - Semantic disambiguation (intent → app via world knowledge)
  - Multi-element reasoning (parse element text, pick by superlative)
  - Abstract reference (goal → action mapping)

Examples are HAND-CRAFTED (not teacher-labeled) because Qwen3-14B
teacher itself only scores 60% on v2 — its labels for the v5-failure
cases would be unreliable.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V5 = Path.home() / ".cache/tinygpt/datasets/pace-v5-sft.jsonl"
DEFAULT_OUT = Path.home() / ".cache/tinygpt/datasets/pace-v8-sft.jsonl"


def _row(user: str, elements: list[str], response):
    """Build a single training row matching v5's shape."""
    el_lines = []
    for i, e in enumerate(elements):
        el_lines.append(f"[{i}] {e}")
    parts = []
    if elements:
        parts.append("on-screen elements:")
        parts.extend(el_lines)
        parts.append("")
    parts.append(f"user said: {user}")
    instruction = "\n".join(parts)
    if isinstance(response, dict):
        response = json.dumps(response, separators=(",", ":"), ensure_ascii=False)
    return {"instruction": instruction, "response": response}


def _click(spoken: str, label: str) -> dict:
    return {"spokenText": spoken, "pointAtLabel": label, "clickLabel": label}


def _qa(spoken: str) -> dict:
    return {"spokenText": spoken, "pointAtLabel": "", "clickLabel": ""}


# =====================================================================
# Semantic disambiguation — intent → app/element via world knowledge
# =====================================================================

DOCK_APPS_PRODUCTIVITY = [
    "dock_icon|24,1000|Pages|Document creation",
    "dock_icon|72,1000|Numbers|Spreadsheets",
    "dock_icon|120,1000|Keynote|Presentations",
    "dock_icon|168,1000|Reminders|Lists and tasks",
]
DOCK_APPS_GENERAL = [
    "dock_icon|24,1000|Mail|Inbox",
    "dock_icon|72,1000|Xcode|Development",
    "dock_icon|120,1000|Safari|Web browser",
    "dock_icon|168,1000|Spotify|Music streaming",
]
DOCK_APPS_VARIED = [
    "dock_icon|24,1000|Calendar|Schedule",
    "dock_icon|72,1000|Messages|iMessage",
    "dock_icon|120,1000|Photos|Image library",
    "dock_icon|168,1000|Notes|Quick notes",
]
DOCK_APPS_BROWSERS = [
    "dock_icon|24,1000|Safari|Apple browser",
    "dock_icon|72,1000|Chrome|Google browser",
    "dock_icon|120,1000|Firefox|Mozilla browser",
    "dock_icon|168,1000|Arc|Modern browser",
]


SEMANTIC_ROWS = [
    # browser-related
    _row("open my browser", DOCK_APPS_GENERAL,
          _click("opening safari", "Safari")),
    _row("i want to check a website", DOCK_APPS_GENERAL,
          _click("opening safari", "Safari")),
    _row("let me look something up online", DOCK_APPS_GENERAL,
          _click("opening safari", "Safari")),
    _row("open chrome", DOCK_APPS_BROWSERS,
          _click("opening chrome", "Chrome")),
    # spreadsheets
    _row("i need to do a budget", DOCK_APPS_PRODUCTIVITY,
          _click("opening numbers", "Numbers")),
    _row("let me make a spreadsheet", DOCK_APPS_PRODUCTIVITY,
          _click("opening numbers", "Numbers")),
    _row("i want to track expenses", DOCK_APPS_PRODUCTIVITY,
          _click("opening numbers", "Numbers")),
    # presentations
    _row("i'm presenting tomorrow", DOCK_APPS_PRODUCTIVITY,
          _click("opening keynote", "Keynote")),
    _row("i need to make slides", DOCK_APPS_PRODUCTIVITY,
          _click("opening keynote", "Keynote")),
    _row("let me put together a deck", DOCK_APPS_PRODUCTIVITY,
          _click("opening keynote", "Keynote")),
    # word processor
    _row("i need to write a letter", DOCK_APPS_PRODUCTIVITY,
          _click("opening pages", "Pages")),
    _row("let me draft a document", DOCK_APPS_PRODUCTIVITY,
          _click("opening pages", "Pages")),
    _row("i want to write an essay", DOCK_APPS_PRODUCTIVITY,
          _click("opening pages", "Pages")),
    # code editor
    _row("open the app i use to write code", DOCK_APPS_GENERAL,
          _click("opening xcode", "Xcode")),
    _row("let me debug something", DOCK_APPS_GENERAL,
          _click("opening xcode", "Xcode")),
    _row("i need my ide", DOCK_APPS_GENERAL,
          _click("opening xcode", "Xcode")),
    # email
    _row("take me to my email", DOCK_APPS_GENERAL,
          _click("opening mail", "Mail")),
    _row("i need to send an email", DOCK_APPS_GENERAL,
          _click("opening mail", "Mail")),
    _row("check my inbox", DOCK_APPS_GENERAL,
          _click("opening mail", "Mail")),
    # music
    _row("open my music app", DOCK_APPS_GENERAL,
          _click("opening spotify", "Spotify")),
    _row("i want to play some music", DOCK_APPS_GENERAL,
          _click("opening spotify", "Spotify")),
    _row("put on some tunes", DOCK_APPS_GENERAL,
          _click("opening spotify", "Spotify")),
    # calendar / scheduling
    _row("what's on my schedule", DOCK_APPS_VARIED,
          _click("opening calendar", "Calendar")),
    _row("when's my next meeting", DOCK_APPS_VARIED,
          _click("opening calendar", "Calendar")),
    # photos
    _row("i want to look at my pictures", DOCK_APPS_VARIED,
          _click("opening photos", "Photos")),
    _row("show me my photos", DOCK_APPS_VARIED,
          _click("opening photos", "Photos")),
    # notes
    _row("let me jot something down", DOCK_APPS_VARIED,
          _click("opening notes", "Notes")),
    # messages
    _row("i need to text someone", DOCK_APPS_VARIED,
          _click("opening messages", "Messages")),
    _row("send a quick message", DOCK_APPS_VARIED,
          _click("opening messages", "Messages")),
]


# =====================================================================
# Multi-element reasoning — parse text fields, pick by superlative
# =====================================================================

PLANS = [
    "button|240,200|Free plan|$0/mo",
    "button|400,200|Pro plan|$15/mo",
    "button|560,200|Enterprise plan|$99/mo",
]
PLANS_ALT = [
    "button|240,200|Basic|$5/mo",
    "button|400,200|Premium|$25/mo",
    "button|560,200|Ultimate|$50/mo",
]
EMAILS = [
    "email_row|240,100|Alice|2 hours ago",
    "email_row|240,140|Bob|yesterday",
    "email_row|240,180|Carol|3 days ago",
    "email_row|240,220|Dave|last week",
]
VIDEOS = [
    "video_card|240,200|Episode 1|12 minutes",
    "video_card|400,200|Episode 2|45 minutes",
    "video_card|560,200|Episode 3|22 minutes",
]
MOVIES = [
    "movie_card|240,200|Inception|7.8 stars",
    "movie_card|400,200|Interstellar|8.6 stars",
    "movie_card|560,200|Tenet|7.4 stars",
]
PRODUCTS = [
    "product_card|240,200|Phone Lite|$299",
    "product_card|400,200|Phone Pro|$799",
    "product_card|560,200|Phone Ultra|$1499",
]


REASONING_ROWS = [
    # cheapest / most expensive
    _row("click the cheapest plan", PLANS,
          _click("opening free plan", "Free plan")),
    _row("pick the cheapest option", PLANS_ALT,
          _click("opening basic", "Basic")),
    _row("select the most expensive option", PLANS,
          _click("opening enterprise plan", "Enterprise plan")),
    _row("buy the premium one", PLANS_ALT,
          _click("opening ultimate", "Ultimate")),
    _row("show me the entry-level product", PRODUCTS,
          _click("opening phone lite", "Phone Lite")),
    _row("get the top-of-the-line phone", PRODUCTS,
          _click("opening phone ultra", "Phone Ultra")),
    _row("which phone is the cheapest", PRODUCTS,
          _qa("phone lite at $299 is the cheapest")),
    # newest / oldest emails
    _row("open the latest email", EMAILS,
          _click("opening alice's message", "Alice")),
    _row("show me the most recent message", EMAILS,
          _click("opening alice's message", "Alice")),
    _row("archive the oldest message", EMAILS,
          _click("archiving dave's message", "Dave")),
    _row("delete the oldest email", EMAILS,
          _click("removing dave's message", "Dave")),
    # longest / shortest videos
    _row("play the longest one", VIDEOS,
          _click("playing episode 2", "Episode 2")),
    _row("play the shortest video", VIDEOS,
          _click("playing episode 1", "Episode 1")),
    _row("watch the 45-minute one", VIDEOS,
          _click("playing episode 2", "Episode 2")),
    # ratings
    _row("pick the highest rated movie", MOVIES,
          _click("opening interstellar", "Interstellar")),
    _row("show me the best reviewed film", MOVIES,
          _click("opening interstellar", "Interstellar")),
    _row("what's the lowest rated movie here", MOVIES,
          _click("opening tenet", "Tenet")),
    _row("which movie has the best rating", MOVIES,
          _qa("interstellar with 8.6 stars has the best rating")),
]


# =====================================================================
# Abstract reference — goal → action mapping
# =====================================================================

FINANCE_BUTTONS = [
    "button|240,200|Buy|Purchase assets",
    "button|400,200|Sell|Liquidate holdings",
    "button|560,200|Transfer|Send to another account",
]
SHOPPING_ACTIONS = [
    "button|240,200|Add to cart|Save for later",
    "button|400,200|Checkout|Complete purchase",
    "button|560,200|Save for later|Wishlist",
]
DOC_ACTIONS = [
    "button|240,200|Save|Save document",
    "button|400,200|Save as|Save with new name",
    "button|560,200|Export|Convert to other format",
    "button|720,200|Share|Send to others",
]


ABSTRACT_ROWS = [
    # finance
    _row("i need to pay my electric bill", FINANCE_BUTTONS,
          _click("opening transfer", "Transfer")),
    _row("send money to my friend", FINANCE_BUTTONS,
          _click("opening transfer", "Transfer")),
    _row("i want to invest", FINANCE_BUTTONS,
          _click("opening buy", "Buy")),
    _row("let me cash out", FINANCE_BUTTONS,
          _click("opening sell", "Sell")),
    _row("i'd like to liquidate this position", FINANCE_BUTTONS,
          _click("opening sell", "Sell")),
    # shopping
    _row("i'm done shopping", SHOPPING_ACTIONS,
          _click("opening checkout", "Checkout")),
    _row("i'll think about this and come back", SHOPPING_ACTIONS,
          _click("opening save for later", "Save for later")),
    _row("add this to my list", SHOPPING_ACTIONS,
          _click("opening add to cart", "Add to cart")),
    # documents
    _row("send this to alice", DOC_ACTIONS,
          _click("opening share", "Share")),
    _row("save a copy under a new name", DOC_ACTIONS,
          _click("opening save as", "Save as")),
    _row("convert this to pdf", DOC_ACTIONS,
          _click("opening export", "Export")),
    _row("just save what i have", DOC_ACTIONS,
          _click("opening save", "Save")),
]


# =====================================================================
# Assembly
# =====================================================================

def all_new_rows() -> list[dict]:
    return SEMANTIC_ROWS + REASONING_ROWS + ABSTRACT_ROWS


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input", type=Path, default=DEFAULT_V5,
                     help="base corpus (default v5)")
    p.add_argument("--out", dest="output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"missing input corpus: {args.input}")

    base = load_jsonl(args.input)
    added = all_new_rows()
    merged = base + added

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"base rows (from {args.input.name}): {len(base)}")
    print(f"  semantic disambiguation: {len(SEMANTIC_ROWS)}")
    print(f"  multi-element reasoning: {len(REASONING_ROWS)}")
    print(f"  abstract reference:      {len(ABSTRACT_ROWS)}")
    print(f"added rows total: {len(added)}")
    print(f"wrote {len(merged)} rows → {args.output}")


if __name__ == "__main__":
    main()
