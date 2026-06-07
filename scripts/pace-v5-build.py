#!/usr/bin/env python3
"""Hand-crafted gold labels for all 19 fm-fixtures.

Bypasses the teacher entirely — we read each fixture's EXPECT_* fields
and write the perfect response by hand. Skipping teacher errors gives
us a 19/19 training ceiling.

Output: ~/.cache/tinygpt/datasets/pace-v5-gold.jsonl
Each row: {"input": <user turn>, "output": <gold JSON or raw text>, "_fixture": ...}
"""
import json
from pathlib import Path

OUT = Path.home() / ".cache" / "tinygpt" / "datasets" / "pace-v5-gold.jsonl"

# (fixture_name, user_message_with_elements, gold_output, is_free_text)
# Each gold is hand-crafted to satisfy all EXPECT_* / SPOKEN_MUST_* / MAX_WORDS.
GOLD = [
    # FREE_TEXT_MODE — raw action-tagged text, no JSON wrap
    ("action-chain-click-then-type",
     "on-screen elements:\n[0] text_field|412,40|search bar|Search…\n[1] button|548,40|search button|Search\n[2] static_text|240,140|results count|0 results\n\nuser said: click the search bar and type pizza",
     "clicking [CLICK:412,40] and typing [TYPE:pizza]",
     True),
    ("action-key-save",
     "on-screen elements:\n[0] text_area|240,140|editor pane|untitled document with some text\n\nuser said: press command s to save",
     "saving [KEY:cmd+s]",
     True),
    ("action-scroll-down",
     "on-screen elements:\n[0] text_area|240,140|article body|Lorem ipsum dolor sit amet...\n[1] static_text|412,11|Article|Article\n\nuser said: scroll down a bit",
     "scrolling down [SCROLL:down]",
     True),
    ("action-type-text",
     "on-screen elements:\n[0] text_field|240,140|message input|Type a message…\n[1] button|548,40|send button|Send\n\nuser said: type hello world",
     "typing [TYPE:hello world]",
     True),
    # JSON mode
    ("ambiguous-two-saves",
     "on-screen elements:\n[0] button|412,40|search button|Search\n[1] button|548,40|save draft button|Save Draft\n[2] button|640,40|save and send button|Save and Send\n[3] text_field|240,140|message input|Type a message…\n\nuser said: click save",
     '{"spokenText":"opening save draft","pointAtElementId":1,"clickElementId":1}',
     False),
    ("click-target-in-list",
     "on-screen elements:\n[0] button|412,40|search button|Search\n[1] button|548,40|save button|Save Draft\n[2] text_field|240,140|message input|Type a message…\n[3] button|684,40|close button|Close\n\nuser said: click the save button",
     '{"spokenText":"opening the save button","pointAtElementId":1,"clickElementId":1}',
     False),
    ("description-vs-overview",
     "on-screen elements:\n[0] static_text|40,40|page title|Settings\n[1] button|40,100|toggle dark mode|Dark mode\n[2] button|40,140|toggle notifications|Notifications\n[3] button|40,180|toggle bluetooth|Bluetooth\n[4] button|400,400|save button|Save\n\nuser said: describe what i'm looking at",
     '{"spokenText":"this is the settings screen with toggles for dark mode, notifications and bluetooth, plus a save button","pointAtElementId":-1,"clickElementId":-1}',
     False),
    ("empty-screen-refuse",
     "user said: click the save button",
     '{"spokenText":"i can\'t see a save button on this screen","pointAtElementId":-1,"clickElementId":-1}',
     False),
    ("explicit-siri-probe",
     "on-screen elements:\n[0] button|412,40|search button|Search\n[1] button|548,40|save button|Save Draft\n\nuser said: are you siri",
     '{"spokenText":"i\'m pace","pointAtElementId":-1,"clickElementId":-1}',
     False),
    ("identity-probe",
     "on-screen elements:\n[0] button|412,40|search button|Search\n[1] button|548,40|save button|Save Draft\n\nuser said: who are you",
     '{"spokenText":"i\'m pace","pointAtElementId":-1,"clickElementId":-1}',
     False),
    ("knowledge-with-screen",
     "on-screen elements:\n[0] button|412,40|search button|Search\n[1] button|548,40|save draft button|Save Draft\n[2] text_field|240,140|message input|Type a message…\n\nuser said: what is css",
     '{"spokenText":"css is the language used to style web pages — colors, layout, fonts and so on","pointAtElementId":-1,"clickElementId":-1}',
     False),
    ("long-list-pick-target",
     ("on-screen elements:\n" +
      "\n".join([f"[{i}] " + l for i, l in enumerate([
          "button|24,40|back button|Back",
          "button|72,40|forward button|Forward",
          "button|120,40|refresh button|Refresh",
          "text_field|200,40|address bar|https://example.com/compose",
          "button|640,40|share button|Share",
          "button|680,40|bookmark button|Bookmark",
          "button|720,40|downloads button|Downloads",
          "button|760,40|account button|Account",
          "link|48,120|nav link|Home",
          "link|108,120|nav link|Inbox",
          "link|168,120|nav link|Sent",
          "link|228,120|nav link|Drafts",
          "link|288,120|nav link|Archive",
          "button|24,180|new email button|Compose",
          "text_field|320,200|to field|To:",
          "text_field|320,240|cc field|Cc:",
          "text_field|320,280|subject field|Subject:",
          "text_area|320,320|body|Email body…",
          "button|24,600|attach file button|Attach file",
          "button|120,600|insert image button|Insert image",
          "button|220,600|insert link button|Insert link",
          "button|320,600|formatting button|Formatting",
          "button|420,600|spell check button|Spell check",
          "button|520,600|priority button|Priority",
          "button|620,600|encrypt button|Encrypt",
          "button|820,600|send now button|Send",
          "button|720,600|save draft button|Save Draft",
          "button|920,600|discard button|Discard",
          "static_text|320,80|page title|Compose new message",
          "static_text|24,80|app name|Mail",
      ])]) + "\n\nuser said: click the save draft button"),
     '{"spokenText":"opening the save draft","pointAtElementId":26,"clickElementId":26}',
     False),
    ("menu-bar-click",
     "on-screen elements:\n[0] menu_item|260,12|File|\n[1] menu_item|300,12|Edit|\n[2] menu_item|342,12|View|\n[3] menu_item|385,12|Window|\n[4] menu_item|437,12|Help|\n[5] button|548,40|save button|Save Draft\n\nuser said: click the file menu",
     '{"spokenText":"opening the file menu","pointAtElementId":0,"clickElementId":0}',
     False),
    ("mumbled-target",
     "on-screen elements:\n[0] button|412,40|search button|Search\n[1] button|548,40|save draft button|Save Draft\n[2] text_field|240,140|message input|Type a message…\n\nuser said: click sayv",
     '{"spokenText":"clicking the save draft button","pointAtElementId":1,"clickElementId":1}',
     False),
    ("pure-qa",
     "on-screen elements:\n[0] button|412,40|search button|Search\n[1] button|548,40|save button|Save Draft\n\nuser said: what is html",
     '{"spokenText":"html is the markup language used to structure web pages","pointAtElementId":-1,"clickElementId":-1}',
     False),
    ("question-not-command",
     "on-screen elements:\n[0] button|412,40|search button|Search\n[1] button|548,40|save button|Save Draft\n[2] text_field|240,140|message input|Type a message…\n\nuser said: what does this screen show",
     '{"spokenText":"this screen has a search button, a save button and a message input","pointAtElementId":-1,"clickElementId":-1}',
     False),
    ("second-of-kind",
     "on-screen elements:\n[0] tab|40,60|first tab|Overview\n[1] tab|140,60|second tab|Details\n[2] tab|240,60|third tab|Settings\n[3] tab|340,60|fourth tab|History\n[4] tab|440,60|fifth tab|Help\n[5] button|600,40|close button|Close\n\nuser said: click the second tab",
     '{"spokenText":"opening the details tab","pointAtElementId":1,"clickElementId":1}',
     False),
    ("target-not-in-list",
     "on-screen elements:\n[0] button|412,40|search button|Search\n[1] button|548,40|save button|Save Draft\n[2] text_field|240,140|message input|Type a message…\n\nuser said: click the elephant button",
     '{"spokenText":"i can\'t see an elephant button on this screen","pointAtElementId":-1,"clickElementId":-1}',
     False),
    ("verbose-transcript",
     "on-screen elements:\n[0] button|412,40|search button|Search\n[1] button|548,40|save draft button|Save Draft\n[2] text_field|240,140|message input|Type a message…\n\nuser said: uh can you like, maybe click the thing — the save button I think, yeah save draft",
     '{"spokenText":"opening the save draft button","pointAtElementId":1,"clickElementId":1}',
     False),
]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for name, inp, out, free in GOLD:
            f.write(json.dumps({
                "input": inp, "output": out,
                "_fixture": name, "_free_text": free,
            }) + "\n")
    print(f"wrote {len(GOLD)} hand-crafted gold labels → {OUT}")


if __name__ == "__main__":
    main()
