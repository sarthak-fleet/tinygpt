from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


QUERY_KEYS = ("user_instruction", "instruction", "user_query", "query", "question", "goal", "prompt")
TOOL_KEYS = ("tools_used", "expected_tools", "actions", "tool", "tool_name", "name")


def read_tau_bench(path: str | Path) -> Iterable[dict[str, Any]]:
    for row in _iter_json_rows(Path(path)):
        query = _first_string(row, QUERY_KEYS)
        tool = _extract_tool(row)
        if query and tool:
            yield {"query": query, "tool": tool, "metadata": {"source": "tau-bench"}}


def _iter_json_rows(path: Path) -> Iterable[dict[str, Any]]:
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.suffix.lower() in {".json", ".jsonl"})
    for file in files:
        try:
            if file.suffix.lower() == ".jsonl":
                for line in file.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield row
                continue
            parsed = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(parsed, list):
            for row in parsed:
                if isinstance(row, dict):
                    yield row
        elif isinstance(parsed, dict):
            if "tasks" in parsed and isinstance(parsed["tasks"], list):
                for row in parsed["tasks"]:
                    if isinstance(row, dict):
                        yield row
            yield parsed


def _first_string(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_tool(row: dict[str, Any]) -> str | None:
    for key in TOOL_KEYS:
        name = _tool_from_value(row.get(key))
        if name:
            return name
    return None


def _tool_from_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        for key in ("name", "tool", "tool_name"):
            name = _tool_from_value(value.get(key))
            if name:
                return name
    if isinstance(value, list):
        for item in value:
            name = _tool_from_value(item)
            if name:
                return name
    return None

