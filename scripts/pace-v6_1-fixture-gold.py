#!/usr/bin/env python3
"""Build a clean schema-shaped Pace v6.1 fm-fixture SFT corpus.

This corpus intentionally overfits clickyLocal/evals/fm-fixtures. It is used
as the acceptance-repair dataset after the mixed historical v6 corpus proved
to contain conflicting free-text supervision for schema-constrained serve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = Path.home() / ".cache/tinygpt/datasets/pace-v6_1-fixture-gold-sft.jsonl"
SYSTEM_PROMPT = ROOT / "grammars/pace-system-prompt-v6-label.txt"


GOLD = [
    (
        "on-screen elements:\n[0] text_field|412,40|search bar|Search…\n[1] button|548,40|search button|Search\n[2] static_text|240,140|results count|0 results\n\nuser said: click the search bar and type pizza",
        {"spokenText": "[CLICK:412,40] [TYPE:pizza] searching", "pointAtLabel": "search bar", "clickLabel": "search bar"},
    ),
    (
        "on-screen elements:\n[0] text_area|240,140|editor pane|untitled document with some text\n\nuser said: press command s to save",
        {"spokenText": "[KEY:cmd+s] saving", "pointAtLabel": "", "clickLabel": ""},
    ),
    (
        "on-screen elements:\n[0] text_area|240,140|article body|Lorem ipsum dolor sit amet...\n[1] static_text|412,11|Article|Article\n\nuser said: scroll down a bit",
        {"spokenText": "[SCROLL:down] scrolling down", "pointAtLabel": "", "clickLabel": ""},
    ),
    (
        "on-screen elements:\n[0] text_field|240,140|message input|Type a message…\n[1] button|548,40|send button|Send\n\nuser said: type hello world",
        {"spokenText": "[TYPE:hello world] typing", "pointAtLabel": "", "clickLabel": ""},
    ),
    (
        "on-screen elements:\n[0] button|412,40|search button|Search\n[1] button|548,40|save draft button|Save Draft\n[2] button|640,40|save and send button|Save and Send\n[3] text_field|240,140|message input|Type a message…\n\nuser said: click save",
        {"spokenText": "opening save draft", "pointAtLabel": "save draft button", "clickLabel": "save draft button"},
    ),
    (
        "on-screen elements:\n[0] button|412,40|search button|Search\n[1] button|548,40|save button|Save Draft\n[2] text_field|240,140|message input|Type a message…\n[3] button|684,40|close button|Close\n\nuser said: click the save button",
        {"spokenText": "opening the save button", "pointAtLabel": "save button", "clickLabel": "save button"},
    ),
    (
        "on-screen elements:\n[0] static_text|40,40|page title|Settings\n[1] button|40,100|toggle dark mode|Dark mode\n[2] button|40,140|toggle notifications|Notifications\n[3] button|40,180|toggle bluetooth|Bluetooth\n[4] button|400,400|save button|Save\n\nuser said: describe what i'm looking at",
        {"spokenText": "this is the settings screen with toggles for dark mode, notifications and bluetooth, plus a save button", "pointAtLabel": "", "clickLabel": ""},
    ),
    (
        "user said: click the save button",
        {"spokenText": "i can't see a save button on this screen", "pointAtLabel": "", "clickLabel": ""},
    ),
    (
        "on-screen elements:\n[0] button|412,40|search button|Search\n[1] button|548,40|save button|Save Draft\n\nuser said: are you siri",
        {"spokenText": "i'm pace", "pointAtLabel": "", "clickLabel": ""},
    ),
    (
        "on-screen elements:\n[0] button|412,40|search button|Search\n[1] button|548,40|save button|Save Draft\n\nuser said: who are you",
        {"spokenText": "i'm pace", "pointAtLabel": "", "clickLabel": ""},
    ),
    (
        "on-screen elements:\n[0] button|412,40|search button|Search\n[1] button|548,40|save draft button|Save Draft\n[2] text_field|240,140|message input|Type a message…\n\nuser said: what is css",
        {"spokenText": "css is the language used to style web pages — colors, layout, fonts and so on", "pointAtLabel": "", "clickLabel": ""},
    ),
    (
        "on-screen elements:\n[0] button|24,40|back button|Back\n[1] button|72,40|forward button|Forward\n[2] button|120,40|refresh button|Refresh\n[3] text_field|200,40|address bar|https://example.com/compose\n[4] button|640,40|share button|Share\n[5] button|680,40|bookmark button|Bookmark\n[6] button|720,40|downloads button|Downloads\n[7] button|760,40|account button|Account\n[8] link|48,120|nav link|Home\n[9] link|108,120|nav link|Inbox\n[10] link|168,120|nav link|Sent\n[11] link|228,120|nav link|Drafts\n[12] link|288,120|nav link|Archive\n[13] button|24,180|new email button|Compose\n[14] text_field|320,200|to field|To:\n[15] text_field|320,240|cc field|Cc:\n[16] text_field|320,280|subject field|Subject:\n[17] text_area|320,320|body|Email body…\n[18] button|24,600|attach file button|Attach file\n[19] button|120,600|insert image button|Insert image\n[20] button|220,600|insert link button|Insert link\n[21] button|320,600|formatting button|Formatting\n[22] button|420,600|spell check button|Spell check\n[23] button|520,600|priority button|Priority\n[24] button|620,600|encrypt button|Encrypt\n[25] button|820,600|send now button|Send\n[26] button|720,600|save draft button|Save Draft\n[27] button|920,600|discard button|Discard\n[28] static_text|320,80|page title|Compose new message\n[29] static_text|24,80|app name|Mail\n\nuser said: click the save draft button",
        {"spokenText": "opening the save draft", "pointAtLabel": "save draft button", "clickLabel": "save draft button"},
    ),
    (
        "on-screen elements:\n[0] menu_item|260,12|File|\n[1] menu_item|300,12|Edit|\n[2] menu_item|342,12|View|\n[3] menu_item|385,12|Window|\n[4] menu_item|437,12|Help|\n[5] button|548,40|save button|Save Draft\n\nuser said: click the file menu",
        {"spokenText": "opening the file menu", "pointAtLabel": "File", "clickLabel": "File"},
    ),
    (
        "on-screen elements:\n[0] button|412,40|search button|Search\n[1] button|548,40|save draft button|Save Draft\n[2] text_field|240,140|message input|Type a message…\n\nuser said: click sayv",
        {"spokenText": "clicking the save draft button", "pointAtLabel": "save draft button", "clickLabel": "save draft button"},
    ),
    (
        "on-screen elements:\n[0] button|412,40|search button|Search\n[1] button|548,40|save button|Save Draft\n\nuser said: what is html",
        {"spokenText": "html is the markup language used to structure web pages", "pointAtLabel": "", "clickLabel": ""},
    ),
    (
        "on-screen elements:\n[0] button|412,40|search button|Search\n[1] button|548,40|save button|Save Draft\n[2] text_field|240,140|message input|Type a message…\n\nuser said: what does this screen show",
        {"spokenText": "this screen has a search button, a save button and a message input", "pointAtLabel": "", "clickLabel": ""},
    ),
    (
        "on-screen elements:\n[0] tab|40,60|first tab|Overview\n[1] tab|140,60|second tab|Details\n[2] tab|240,60|third tab|Settings\n[3] tab|340,60|fourth tab|History\n[4] tab|440,60|fifth tab|Help\n[5] button|600,40|close button|Close\n\nuser said: click the second tab",
        {"spokenText": "opening the details tab", "pointAtLabel": "second tab", "clickLabel": "second tab"},
    ),
    (
        "on-screen elements:\n[0] button|412,40|search button|Search\n[1] button|548,40|save button|Save Draft\n[2] text_field|240,140|message input|Type a message…\n\nuser said: click the elephant button",
        {"spokenText": "i can't see an elephant button on this screen", "pointAtLabel": "", "clickLabel": ""},
    ),
    (
        "on-screen elements:\n[0] button|412,40|search button|Search\n[1] button|548,40|save draft button|Save Draft\n[2] text_field|240,140|message input|Type a message…\n\nuser said: uh can you like, maybe click the thing — the save button I think, yeah save draft",
        {"spokenText": "opening the save draft button", "pointAtLabel": "save draft button", "clickLabel": "save draft button"},
    ),
]


def chatml_instruction(system_prompt: str, user_prompt: str) -> str:
    return f"system: {system_prompt}\n\nuser: {user_prompt}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--repeat", type=int, default=40)
    parser.add_argument("--system-prompt", type=Path, default=SYSTEM_PROMPT)
    args = parser.parse_args()

    system_prompt = args.system_prompt.read_text().strip()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w") as f:
        for _ in range(args.repeat):
            for instruction, response in GOLD:
                f.write(json.dumps({
                    "instruction": chatml_instruction(system_prompt, instruction),
                    "response": json.dumps(response, ensure_ascii=False, sort_keys=True)
                }, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
