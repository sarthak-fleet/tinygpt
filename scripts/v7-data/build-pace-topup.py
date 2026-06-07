#!/usr/bin/env python3
"""Build a small hand-curated Pace Planner v7 top-up corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query",
            "description": "Read or search a known source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "mode": {"type": "string"},
                    "query": {"type": "string"}
                },
                "required": ["source"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perform",
            "description": "Perform a UI action against a visible target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "action": {"type": "string"},
                    "params": {"type": "object"}
                },
                "required": ["target", "action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set",
            "description": "Set a value on a control or setting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "value": {"type": "string"}
                },
                "required": ["target", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compose",
            "description": "Draft text or structured content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {"type": "string"},
                    "content": {"type": "string"},
                    "target": {"type": "string"}
                },
                "required": ["format", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clarify",
            "description": "Ask for a missing value, choice, or confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "question": {"type": "string"},
                    "choices": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["kind", "question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "say",
            "description": "Say a short response to the user.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_memory",
            "description": "Query long-term user memory.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait for a condition or short delay.",
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string"},
                    "timeout_sec": {"type": "integer"}
                },
                "required": ["condition"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open",
            "description": "Open an app, URL, file, or visible screen.",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule",
            "description": "Create a time-bound reminder or calendar action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "when": {"type": "string"},
                    "what": {"type": "string"}
                },
                "required": ["when", "what"]
            }
        }
    },
]


EXAMPLES = [
    ("click the save button", {"verb": "perform", "args": {"target": "save button", "action": "click"}, "spoken_text": "opening save"}),
    ("press command s", {"verb": "perform", "args": {"target": "keyboard", "action": "key", "params": {"key": "cmd+s"}}, "spoken_text": "saving"}),
    ("scroll down three times", {"verb": "perform", "args": {"target": "current window", "action": "scroll", "params": {"direction": "down", "count": 3}}, "spoken_text": "scrolling"}),
    ("type hello world in the message box", {"verb": "perform", "args": {"target": "message box", "action": "type", "params": {"text": "hello world"}}, "spoken_text": "typing"}),
    ("set brightness to fifty percent", {"verb": "set", "args": {"target": "brightness", "value": "50%"}, "spoken_text": "setting brightness"}),
    ("open safari", {"verb": "open", "args": {"target": "Safari"}, "spoken_text": "opening Safari"}),
    ("what is on my screen", {"verb": "query", "args": {"source": "current_screen", "mode": "describe"}, "spoken_text": ""}),
    ("find invoices in my files", {"verb": "query", "args": {"source": "files", "mode": "search", "query": "invoices"}, "spoken_text": "searching files"}),
    ("write a short reply saying thanks", {"verb": "compose", "args": {"format": "reply", "content": "thanks"}, "spoken_text": "drafting"}),
    ("remind me tomorrow morning to send the deck", {"verb": "schedule", "args": {"when": "tomorrow morning", "what": "send the deck"}, "spoken_text": "setting reminder"}),
    ("what did I say my preferred editor was", {"verb": "query_memory", "args": {"query": "preferred editor"}, "spoken_text": ""}),
    ("wait until the download finishes", {"verb": "wait", "args": {"condition": "download finishes", "timeout_sec": 600}, "spoken_text": "waiting"}),
    ("which save button do you mean", {"verb": "clarify", "args": {"kind": "choice", "question": "Which save button?", "choices": ["left save", "right save"]}, "spoken_text": "which save button"}),
    ("delete this file", {"verb": "clarify", "args": {"kind": "confirmation", "question": "Are you sure you want to delete this file?", "choices": ["yes", "no"]}, "spoken_text": "please confirm"}),
    ("say all set", {"verb": "say", "args": {"text": "all set"}, "spoken_text": "all set"}),
]


def instruction(intent: str) -> str:
    return (
        "You have these tools available. Choose exactly one tool call.\n\n"
        f"Tools:\n{json.dumps({'tools': TOOLS}, ensure_ascii=False, sort_keys=True)}\n\n"
        f"User intent: {intent}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path.home() / ".cache/tinygpt/datasets/pace-v7-topup.jsonl")
    parser.add_argument("--repeat", type=int, default=12)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.out.open("w") as f:
        for _ in range(max(1, args.repeat)):
            for intent, response in EXAMPLES:
                f.write(json.dumps({
                    "instruction": instruction(intent),
                    "response": json.dumps(response, ensure_ascii=False, sort_keys=True)
                }, ensure_ascii=False) + "\n")
                count += 1
    print(f"wrote {count} rows to {args.out}")


if __name__ == "__main__":
    main()
