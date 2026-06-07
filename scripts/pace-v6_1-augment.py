#!/usr/bin/env python3
"""Build Pace planner v6.1 SFT data by appending action-tag examples.

The existing v6 corpus is label-based:
  {"instruction": "...", "response": "..."}

Some rows are free-text action tags, while JSON-mode rows emit:
  {"spokenText": "...", "pointAtLabel": "...", "clickLabel": "..."}

This script preserves the v6 corpus, appends v6.1 rows, and validates every
JSON-mode response against grammars/pace-fm-label-response.schema.json without
adding a jsonschema dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = Path.home() / ".cache/tinygpt/datasets/pace-v6-sft.jsonl"
DEFAULT_OUT = Path.home() / ".cache/tinygpt/datasets/pace-v6_1-sft.jsonl"
SCHEMA_PATH = ROOT / "grammars/pace-fm-label-response.schema.json"
SYSTEM_PROMPT = ROOT / "grammars/pace-system-prompt-v6-label.txt"


def screen(elements: list[str], user: str) -> str:
    parts: list[str] = []
    if elements:
        parts.append("on-screen elements:")
        parts.extend(elements)
        parts.append("")
    parts.append(f"user said: {user}")
    return "\n".join(parts)


def chatml_instruction(system_prompt: str, user_prompt: str) -> str:
    return f"system: {system_prompt}\n\nuser: {user_prompt}"


def row(user: str, elements: list[str], response: str | dict[str, str]) -> dict[str, str]:
    if isinstance(response, dict):
        response_text = json.dumps(response, separators=(",", ":"), ensure_ascii=False)
    else:
        response_text = response
    return {"instruction": screen(elements, user), "response": response_text}


def action_rows() -> list[dict[str, str]]:
    editor = ["[0] text_area|240,140|editor pane|untitled document with some text"]
    article = [
        "[0] text_area|240,140|article body|Lorem ipsum dolor sit amet...",
        "[1] static_text|412,11|Article|Article",
    ]
    message = [
        "[0] text_field|240,140|message input|Type a message…",
        "[1] button|548,40|send button|Send",
    ]
    browser = [
        "[0] text_field|200,40|address bar|https://example.com",
        "[1] button|120,40|refresh button|Refresh",
        "[2] button|720,40|downloads button|Downloads",
    ]
    desktop = ["[0] desktop|0,0|desktop|Finder desktop"]
    search = [
        "[0] text_field|400,80|search bar|empty",
        "[1] button|548,80|search button|Search",
    ]
    form = [
        "[0] text_field|200,120|name field|Name",
        "[1] button|240,300|submit|Submit",
    ]

    def action_response(spoken: str, label: str = "") -> dict[str, str]:
        return {"spokenText": spoken, "pointAtLabel": label, "clickLabel": label}

    return [
        row("press command s to save", editor, action_response("[KEY:cmd+s] saving")),
        row("save the file", editor, action_response("[KEY:cmd+s] saving")),
        row("hit cmd s", editor, action_response("[KEY:cmd+s] saving")),
        row("use the keyboard shortcut for save", editor, action_response("[KEY:cmd+s] saving")),
        row("press enter", message, action_response("[KEY:Return] pressing enter")),
        row("hit return", message, action_response("[KEY:Return] pressing return")),
        row("press escape", form, action_response("[KEY:Escape] dismissing")),
        row("hit escape to close this", form, action_response("[KEY:Escape] dismissing")),
        row("press tab", form, action_response("[KEY:Tab] moving focus")),
        row("press shift tab", form, action_response("[KEY:shift+Tab] moving focus back")),
        row("press cmd plus shift plus t", browser, action_response("[KEY:cmd+shift+t] reopening tab")),
        row("reopen the last closed tab", browser, action_response("[KEY:cmd+shift+t] reopening tab")),
        row("copy this", editor, action_response("[KEY:cmd+c] copying")),
        row("paste it", message, action_response("[KEY:cmd+v] pasting")),
        row("select all", editor, action_response("[KEY:cmd+a] selecting all")),
        row("scroll down", article, action_response("[SCROLL:down] scrolling down")),
        row("scroll down a bit", article, action_response("[SCROLL:down] scrolling down")),
        row("scroll down three times", article, action_response("[SCROLL:down:3] scrolling down")),
        row("scroll up", article, action_response("[SCROLL:up] scrolling up")),
        row("scroll back up", article, action_response("[SCROLL:up] scrolling up")),
        row("go to the bottom of this page", article, action_response("[SCROLL:down:5] scrolling down")),
        row("type hello world", message, action_response("[TYPE:hello world] typing")),
        row("type the message thanks for the help", message, action_response("[TYPE:thanks for the help] typing")),
        row("enter pikachu in the search field", search, action_response("[TYPE:pikachu] typing")),
        row("write yes please", message, action_response("[TYPE:yes please] typing")),
        row("fill in my name as sarthak", form, action_response("[TYPE:sarthak] typing")),
        row("open safari", desktop, action_response("[OPEN_APP:Safari] opening Safari")),
        row("launch notes", desktop, action_response("[OPEN_APP:Notes] opening Notes")),
        row("open calendar", desktop, action_response("[OPEN_APP:Calendar] opening Calendar")),
        row("open finder", desktop, action_response("[OPEN_APP:Finder] opening Finder")),
        row("click the search bar and type pikachu", search, action_response("[CLICK:400,80] [TYPE:pikachu] searching", "search bar")),
        row("tap search and enter pizza", search, action_response("[CLICK:400,80] [TYPE:pizza] searching", "search bar")),
        row("click submit and press enter", form, action_response("[CLICK:240,300] [KEY:Return] submitting", "submit")),
        row("click the name field and type sarthak", form, action_response("[CLICK:200,120] [TYPE:sarthak] typing", "name field")),
        row("click the search field then scroll down", search, action_response("[CLICK:400,80] [SCROLL:down] scrolling", "search bar")),
    ]


def disambiguation_rows() -> list[dict[str, str]]:
    tabs = [
        "[0] tab|40,60|first tab|Overview",
        "[1] tab|140,60|second tab|Details",
        "[2] tab|240,60|third tab|Settings",
        "[3] tab|340,60|fourth tab|History",
        "[4] tab|440,60|fifth tab|Help",
    ]
    saves = [
        "[0] button|200,100|save · left|Save",
        "[1] button|400,100|save · right|Save",
        "[2] button|600,100|cancel|Cancel",
    ]
    files = [
        "[0] button|200,100|file · first|File",
        "[1] button|360,100|file · second|File",
        "[2] button|520,100|file · third|File",
    ]

    def json_response(spoken: str, label: str) -> dict[str, str]:
        return {"spokenText": spoken, "pointAtLabel": label, "clickLabel": label}

    return [
        row("click the second tab", tabs, json_response("opening the second tab", "second tab")),
        row("open the details tab", tabs, json_response("opening the details tab", "second tab")),
        row("click the third tab", tabs, json_response("opening the third tab", "third tab")),
        row("click the fourth tab", tabs, json_response("opening the fourth tab", "fourth tab")),
        row("click the second save button", saves, json_response("opening the right save button", "save · right")),
        row("press the right save button", saves, json_response("opening the right save button", "save · right")),
        row("click the left save button", saves, json_response("opening the left save button", "save · left")),
        row("click the first file button", files, json_response("opening the first file button", "file · first")),
        row("click the second file button", files, json_response("opening the second file button", "file · second")),
        row("click the third file button", files, json_response("opening the third file button", "file · third")),
    ]


def _wrap_plain_to_json(response: str) -> str:
    """Ensure every response is a JSON object matching the v6 schema.

    v6 base corpus has ~59 free-text action-tag rows (pre-v6 free-text format)
    mixed with ~172 JSON-mode rows. At inference, the v6 grammar forces JSON
    output — so the free-text rows give the model a training signal it can
    never actually emit. Wrapping plain strings into the JSON shape gives a
    consistent signal: always emit {spokenText, pointAtLabel, clickLabel}, with
    action tags living inside spokenText.

    For ambiguous wrapping cases (we don't know the original pointAt/click
    intent of a plain string), set both label fields to empty. The action tag
    inside spokenText carries the click coordinates / key / etc., so the
    executor doesn't need pointAt/click for those rows.
    """
    stripped = response.strip()
    if stripped.startswith("{"):
        return response  # already JSON
    spoken = stripped or "ok"  # spokenText must be 1..300 chars
    return json.dumps(
        {"spokenText": spoken, "pointAtLabel": "", "clickLabel": ""},
        separators=(",", ":"),
        ensure_ascii=False,
    )


def load_jsonl(path: Path, *, system_prompt: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj.get("instruction"), str) or not isinstance(obj.get("response"), str):
                raise ValueError(f"{path}:{lineno}: expected instruction/response strings")
            if not obj["instruction"].lstrip().lower().startswith("system:"):
                obj["instruction"] = chatml_instruction(system_prompt, obj["instruction"])
            obj["response"] = _wrap_plain_to_json(obj["response"])
            rows.append(obj)
    return rows


def validate_json_response(text: str, *, context: str) -> None:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return
    obj = json.loads(stripped)
    required = {"spokenText", "pointAtLabel", "clickLabel"}
    extra = set(obj) - required
    missing = required - set(obj)
    if extra or missing:
        raise ValueError(f"{context}: schema keys extra={sorted(extra)} missing={sorted(missing)}")
    for key in required:
        if not isinstance(obj[key], str):
            raise ValueError(f"{context}: {key} must be string")
    if not (1 <= len(obj["spokenText"]) <= 300):
        raise ValueError(f"{context}: spokenText length invalid")
    if len(obj["pointAtLabel"]) > 100 or len(obj["clickLabel"]) > 100:
        raise ValueError(f"{context}: label too long")


def validate_rows(rows: list[dict[str, str]]) -> None:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(SCHEMA_PATH)
    json.loads(SCHEMA_PATH.read_text())
    for idx, item in enumerate(rows, 1):
        validate_json_response(item["response"], context=f"row {idx}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", dest="output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--system-prompt", type=Path, default=SYSTEM_PROMPT)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"missing input corpus: {args.input}")

    system_prompt = args.system_prompt.read_text().strip()
    base = load_jsonl(args.input, system_prompt=system_prompt)
    added = [
        {**item, "instruction": chatml_instruction(system_prompt, item["instruction"])}
        for item in action_rows() + disambiguation_rows()
    ]
    merged = base + added
    validate_rows(merged)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"base rows: {len(base)}")
    print(f"added rows: {len(added)}")
    print(f"wrote rows: {len(merged)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
