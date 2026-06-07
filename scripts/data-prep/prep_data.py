#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from bfcl_reader import read_bfcl
from github_reader import read_github
from tau_bench_reader import read_tau_bench


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare TinyGPT tool-routing data from BFCL, tau-bench, and GitHub.")
    parser.add_argument("--bfcl", action="append", default=[], help="Path to BFCL JSON/JSONL file or directory.")
    parser.add_argument("--tau-bench", action="append", default=[], help="Path to tau-bench JSON/JSONL file or directory.")
    parser.add_argument("--github", action="append", default=[], help="GitHub owner/repo to ingest through the REST API.")
    parser.add_argument("--kinds", default="issues-prs,reviews,commits", help="Comma-separated GitHub kinds.")
    parser.add_argument("--dedup", action="store_true", help="Deduplicate exact (query, tool) pairs.")
    parser.add_argument("--quality-filter", action="store_true", help="Drop empty/very-short rows before writing.")
    parser.add_argument("--tools", help="Optional OpenAI-style tool schema; rows outside the catalog are dropped.")
    parser.add_argument("--num-workers", type=int, default=1, help="Reserved for datatrove/local parallel execution.")
    parser.add_argument("--limit", type=int, help="Stop after this many emitted rows.")
    parser.add_argument("--out", help="Output JSONL path.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and count rows without writing JSONL.")
    args = parser.parse_args(argv)
    if not args.dry_run and not args.out:
        parser.error("--out is required unless --dry-run is set")

    rows = _iter_sources(args)
    allowed_tools = _load_allowed_tools(args.tools) if args.tools else None
    if allowed_tools is not None:
        rows = _tool_filter(rows, allowed_tools)
    if args.quality_filter:
        rows = _quality_filter(rows)
    if args.dedup:
        rows = _dedup(rows)
    if args.dry_run:
        count = _count_rows(rows, limit=args.limit)
        print(f"prep-data: dry-run counted {count} rows", file=sys.stderr)
    else:
        count = _write_jsonl(rows, Path(args.out), limit=args.limit)
        print(f"prep-data: wrote {count} rows -> {args.out}", file=sys.stderr)
    if args.num_workers > 1:
        print("prep-data: local fallback reader is serial; install/use datatrove executors for parallel runs", file=sys.stderr)
    return 0


def _iter_sources(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    for path in args.bfcl:
        yield from read_bfcl(path)
    for path in args.tau_bench:
        yield from read_tau_bench(path)
    kinds = [kind.strip() for kind in args.kinds.split(",") if kind.strip()]
    for repo in args.github:
        yield from read_github(repo, kinds=kinds, limit=args.limit)


def _quality_filter(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        query = str(row.get("query") or "").strip()
        tool = str(row.get("tool") or "").strip()
        if len(query) >= 3 and tool:
            row["query"] = query
            row["tool"] = tool
            yield row


def _tool_filter(rows: Iterable[dict[str, Any]], allowed: set[str]) -> Iterable[dict[str, Any]]:
    for row in rows:
        if str(row.get("tool") or "") in allowed:
            yield row


def _dedup(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("query") or ""), str(row.get("tool") or ""))
        if key in seen:
            continue
        seen.add(key)
        yield row


def _write_jsonl(rows: Iterable[dict[str, Any]], out: Path, limit: int | None = None) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            if "query" not in row or "tool" not in row:
                continue
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break
    return count


def _count_rows(rows: Iterable[dict[str, Any]], limit: int | None = None) -> int:
    count = 0
    for row in rows:
        if "query" not in row or "tool" not in row:
            continue
        count += 1
        if limit is not None and count >= limit:
            break
    return count


def _load_allowed_tools(path: str) -> set[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        tools = payload.get("tools", [])
    elif isinstance(payload, list):
        tools = payload
    else:
        tools = []
    allowed: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            allowed.add(fn["name"])
        elif isinstance(tool.get("name"), str):
            allowed.add(tool["name"])
    return allowed


if __name__ == "__main__":
    raise SystemExit(main())
