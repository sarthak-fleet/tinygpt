#!/usr/bin/env python3
"""Normalize public function-calling rows into Planner v7 SFT JSONL.

Input is intentionally tolerant because xLAM-style datasets appear in a few
shapes. The script looks for an intent/user prompt, a tools list, and a target
tool call. Output rows are `{instruction,response}` for `tinygpt sft`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REVISED_VERBS = {
    "query", "perform", "set", "compose", "clarify",
    "say", "query_memory", "wait", "open", "schedule",
}


def first_string(obj: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = obj.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return ""


def extract_tools(obj: dict[str, Any]) -> list[dict[str, Any]]:
    raw = obj.get("tools") or obj.get("available_tools") or obj.get("functions")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if "function" in item and isinstance(item["function"], dict):
            fn = item["function"]
        else:
            fn = item
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        })
    return out


def extract_call(obj: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [
        obj.get("tool_call"),
        obj.get("function_call"),
        obj.get("expected_call"),
        obj.get("answer"),
        obj.get("output"),
    ]
    for raw in candidates:
        call = parse_call(raw)
        if call:
            return call
    messages = obj.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            calls = message.get("tool_calls")
            if isinstance(calls, list) and calls:
                call = parse_call(calls[0])
                if call:
                    return call
            call = parse_call(message.get("content"))
            if call:
                return call
    return None


def parse_call(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict):
        return None
    if "function" in raw and isinstance(raw["function"], dict):
        raw = raw["function"]
    name = raw.get("name") or raw.get("tool_name") or raw.get("verb")
    args = raw.get("arguments") or raw.get("args") or raw.get("parameters") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"text": args}
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(args, dict):
        args = {"value": args}
    return {"verb": map_verb(name), "args": args}


def map_verb(name: str) -> str:
    lower = name.lower().replace("-", "_")
    if lower in REVISED_VERBS:
        return lower
    if lower.startswith(("get_", "list_", "find_", "read_", "search_")):
        return "query"
    if lower.startswith(("open_", "navigate_")):
        return "open"
    if lower.startswith(("set_", "update_")):
        return "set"
    if lower.startswith(("create_", "write_", "draft_")):
        return "compose"
    if lower.startswith(("ask_", "confirm_", "choose_")):
        return "clarify"
    if lower.startswith(("schedule_", "remind_")):
        return "schedule"
    return "perform"


def render_instruction(intent: str, tools: list[dict[str, Any]]) -> str:
    return (
        "You have these tools available. Choose exactly one tool call.\n\n"
        f"Tools:\n{json.dumps({'tools': tools}, ensure_ascii=False, sort_keys=True)}\n\n"
        f"User intent: {intent}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input", type=Path, required=True)
    parser.add_argument("--out", dest="output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    written = 0
    skipped = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open() as src, args.output.open("w") as dst:
        for line in src:
            if args.limit and written >= args.limit:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            intent = first_string(obj, ["query", "instruction", "user", "prompt", "intent"])
            tools = extract_tools(obj)
            call = extract_call(obj)
            if not intent or not tools or not call:
                skipped += 1
                continue
            response = {
                "verb": call["verb"],
                "args": call["args"],
                "spoken_text": ""
            }
            dst.write(json.dumps({
                "instruction": render_instruction(intent, tools),
                "response": json.dumps(response, ensure_ascii=False, sort_keys=True)
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} rows to {args.output}; skipped {skipped}")


if __name__ == "__main__":
    main()
