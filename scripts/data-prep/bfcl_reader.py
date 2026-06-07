from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


QUERY_KEYS = ("question", "user_query", "prompt", "task", "user_input", "instruction", "query")
TOOL_KEYS = ("function", "tool_name", "name", "answer", "ground_truth", "expected_function")


def read_bfcl(path: str | Path) -> Iterable[dict[str, Any]]:
    for row in _iter_json_rows(Path(path), suffixes={".json", ".jsonl"}):
        query = _extract_query(row)
        tool = _extract_tool(row)
        if query and tool:
            yield {"query": query, "tool": tool, "metadata": {"source": "bfcl"}}


def _iter_json_rows(path: Path, suffixes: set[str]) -> Iterable[dict[str, Any]]:
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.suffix.lower() in suffixes)
    for file in files:
        if file.suffix.lower() == ".jsonl":
            with file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        yield row
            continue

        try:
            parsed = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(parsed, list):
            for row in parsed:
                if isinstance(row, dict):
                    yield row
        elif isinstance(parsed, dict):
            for value in parsed.values():
                if isinstance(value, list):
                    for row in value:
                        if isinstance(row, dict):
                            yield row
                elif isinstance(value, dict):
                    yield value
            yield parsed


def _extract_query(row: dict[str, Any]) -> str | None:
    for key in QUERY_KEYS:
        value = row.get(key)
        text = _text_from_value(value)
        if text:
            return text
    return _first_user_message(row.get("messages")) or _first_user_message(row.get("question"))


def _extract_tool(row: dict[str, Any]) -> str | None:
    for key in TOOL_KEYS:
        value = row.get(key)
        name = _tool_from_value(value)
        if name:
            return name
    return None


def _text_from_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        return _first_user_message(value)
    return None


def _first_user_message(value: Any) -> str | None:
    if isinstance(value, dict):
        if value.get("role") == "user" and isinstance(value.get("content"), str):
            return value["content"].strip() or None
        return None
    if not isinstance(value, list):
        return None
    for item in value:
        if isinstance(item, list):
            found = _first_user_message(item)
            if found:
                return found
        elif isinstance(item, dict):
            if item.get("role") == "user" and isinstance(item.get("content"), str):
                return item["content"].strip() or None
    return None


def _tool_from_value(value: Any) -> str | None:
    if isinstance(value, str):
        return _strip_call(value)
    if isinstance(value, dict):
        for key in ("name", "tool_name", "function", "tool"):
            name = _tool_from_value(value.get(key))
            if name:
                return name
    if isinstance(value, list):
        for item in value:
            name = _tool_from_value(item)
            if name:
                return name
    return None


def _strip_call(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    return value.split("(", 1)[0].strip() or None

