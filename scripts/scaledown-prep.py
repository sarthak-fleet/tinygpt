#!/usr/bin/env python3
"""scaledown-prep.py — synthesize ScaleDown training pairs from cached data.

Outputs JSONL rows of shape:

    {
        "question": "<short user query / synthesized question>",
        "context": "<long document split into sentences>",
        "sentences": ["<s0>", "<s1>", ...],
        "selected": [3, 7, 12],
        "compressed": "<concatenation of selected sentences>"
    }

Used to bootstrap B25 ScaleDown SFT data while D3 (MS-MARCO + Natural
Questions canonical pull) stays blocked. Replace with the canonical
sources when they land — same output schema. See docs/recipes/b25-scaledown.md.

Currently supports one source format: --source hermes-fc
(reads ~/.cache/tinygpt/datasets/hermes-fc.jsonl).

For hermes-fc: each row has `instruction` (system + tools + user query)
and `response` (tool call(s)). The "compressed" form keeps only the tool
definitions whose names appear in the response, plus the system prompt
preamble and the user query. The full instruction is the "context."
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HOME = Path.home()
DEFAULT_HERMES = HOME / ".cache/tinygpt/datasets/hermes-fc.jsonl"
DEFAULT_FCC = HOME / ".cache/tinygpt/datasets/function-calling-chatml.jsonl"
DEFAULT_OUT = HOME / ".cache/tinygpt/datasets/scaledown-train-synthetic.jsonl"

# Regex compiled once.
TOOL_BLOCK_ALL_RE = re.compile(r"<tools>(.*?)</tools>", re.DOTALL)
TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
RESP_CALL_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z<])")


def split_sentences(text: str) -> list[str]:
    parts = SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_tool_defs(instruction: str) -> tuple[str, list[tuple[str, str]], str]:
    """Return (prefix, [(tool_name, tool_def_text)], suffix).

    prefix: everything before <tools>
    tool list: each (name, compact JSON for the single tool def)
    suffix: everything after </tools>

    hermes-fc packs tools as a single JSON array between the tags; we
    parse the array and re-emit each tool as its own compact JSON line.
    """
    # Hermes' system prompt mentions `<tools> </tools>` as a syntax
    # example before the real <tools>{json}</tools> block; pick the
    # longest match to skip the empty doc-example.
    matches = list(TOOL_BLOCK_ALL_RE.finditer(instruction))
    matches = [m for m in matches if m.group(1).strip()]
    if not matches:
        return instruction, [], ""
    m = max(matches, key=lambda mm: len(mm.group(1)))
    prefix = instruction[: m.start()].rstrip()
    suffix = instruction[m.end() :].lstrip()
    tools_blob = m.group(1).strip()

    tools: list[tuple[str, str]] = []
    try:
        parsed = json.loads(tools_blob)
    except json.JSONDecodeError:
        # fallback to regex name-scan (line-separated tool defs)
        for line in tools_blob.splitlines():
            if not line.strip():
                continue
            nm = TOOL_NAME_RE.search(line)
            if nm:
                tools.append((nm.group(1), line.rstrip()))
        return prefix, tools, suffix

    if not isinstance(parsed, list):
        return prefix, [], suffix
    for entry in parsed:
        # Hermes shape: {"type": "function", "function": {"name": ..., ...}}
        fn = entry.get("function", entry) if isinstance(entry, dict) else None
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str):
            continue
        tools.append((name, json.dumps(entry, separators=(",", ":"))))
    return prefix, tools, suffix


def called_tool_names(response: str) -> set[str]:
    return set(RESP_CALL_NAME_RE.findall(response))


def from_hermes(row: dict) -> dict | None:
    instr = row.get("instruction", "")
    resp = row.get("response", "")
    if not instr or not resp:
        return None

    prefix, tools, suffix = extract_tool_defs(instr)
    if not tools:
        return None
    called = called_tool_names(resp)
    if not called:
        return None
    # No "compression" if the call uses every tool listed.
    if called >= {n for n, _ in tools}:
        return None

    sentences: list[str] = []
    # prefix split into sentences
    sentences.extend(split_sentences(prefix))
    # one sentence per tool definition (so the mask is per-tool)
    tool_sentence_start = len(sentences)
    sentences.extend(td for _, td in tools)
    # suffix split into sentences
    suffix_sentence_start = len(sentences)
    sentences.extend(split_sentences(suffix))

    selected: list[int] = list(range(0, tool_sentence_start))  # prefix kept
    for offset, (name, _) in enumerate(tools):
        if name in called:
            selected.append(tool_sentence_start + offset)
    selected.extend(range(suffix_sentence_start, len(sentences)))  # suffix kept

    if len(selected) >= len(sentences):
        return None
    if not sentences:
        return None

    user_q_idx = next((i for i, s in enumerate(sentences) if s.lower().startswith("user:")), -1)
    question = sentences[user_q_idx] if user_q_idx >= 0 else "(implicit — tool selection)"

    compressed = "\n".join(sentences[i] for i in sorted(selected))
    return {
        "question": question,
        "context": "\n".join(sentences),
        "sentences": sentences,
        "selected": sorted(selected),
        "compressed": compressed,
        "_source": "hermes-fc",
        "_compression_ratio": round(len(compressed) / max(1, len("\n".join(sentences))), 3),
    }


def from_fcc(row: dict) -> dict | None:
    """function-calling-chatml shape:
    {system_message, function_description (JSON), conversations: [{from, value}]}.

    Compression signal: of the multi-turn dialog, keep only turns that
    contain a function call, function response, the final assistant
    answer, and the immediately preceding user query. Drop intermediate
    explanatory turns.
    """
    convs = row.get("conversations")
    if not isinstance(convs, list) or len(convs) < 3:
        return None

    sentences: list[str] = []
    selected: list[int] = []

    # Always keep the system + function description as one preamble sentence.
    sys_msg = (row.get("system_message") or "").strip()
    fn_desc = (row.get("function_description") or "").strip()
    if sys_msg or fn_desc:
        sentences.append(f"system: {sys_msg}\n{fn_desc}")
        selected.append(0)

    user_question = ""
    for i, turn in enumerate(convs):
        if not isinstance(turn, dict):
            continue
        role = turn.get("from", "?")
        val = (turn.get("value") or "").strip()
        if not val:
            continue
        idx = len(sentences)
        sentences.append(f"{role}: {val}")
        if role == "human" and not user_question:
            user_question = val[:200]
        if role == "human":
            selected.append(idx)
        elif "function_call" in val or "function_response" in val:
            selected.append(idx)
        elif i == len(convs) - 1 and role == "gpt":
            selected.append(idx)  # final assistant answer

    selected = sorted(set(selected))
    if not selected or len(selected) >= len(sentences) or len(sentences) < 4:
        return None

    compressed = "\n".join(sentences[i] for i in selected)
    full = "\n".join(sentences)
    ratio = round(len(compressed) / max(1, len(full)), 3)
    return {
        "question": user_question or "(implicit — function call extraction)",
        "context": full,
        "sentences": sentences,
        "selected": selected,
        "compressed": compressed,
        "_source": "function-calling-chatml",
        "_compression_ratio": ratio,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["hermes-fc", "fcc"], default="hermes-fc")
    ap.add_argument("--input", type=Path, default=None,
                    help="defaults: hermes-fc → DEFAULT_HERMES, fcc → DEFAULT_FCC")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = no limit")
    ap.add_argument(
        "--max-keep-ratio",
        type=float,
        default=1.0,
        help="reject rows that keep more than this fraction of the original "
        "context (0.6 = at least 40%% compression).",
    )
    args = ap.parse_args()

    if args.input is None:
        args.input = DEFAULT_HERMES if args.source == "hermes-fc" else DEFAULT_FCC

    if not args.input.exists():
        print(f"input missing: {args.input}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    in_count = 0
    out_count = 0
    sum_ratio = 0.0
    with args.input.open() as fin, args.out.open("w") as fout:
        for line in fin:
            in_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if args.source == "hermes-fc":
                synth = from_hermes(row)
            elif args.source == "fcc":
                synth = from_fcc(row)
            else:
                synth = None
            if synth is None:
                continue
            if synth["_compression_ratio"] > args.max_keep_ratio:
                continue
            fout.write(json.dumps(synth, ensure_ascii=False) + "\n")
            out_count += 1
            sum_ratio += synth["_compression_ratio"]
            if args.max_rows and out_count >= args.max_rows:
                break

    avg = (sum_ratio / out_count) if out_count else 0.0
    in_size = args.input.stat().st_size / 1e6
    out_size = args.out.stat().st_size / 1e6 if args.out.exists() else 0.0
    print(f"read   {in_count:>7} rows from {args.input}  ({in_size:.1f} MB)")
    print(f"wrote  {out_count:>7} rows to   {args.out}  ({out_size:.1f} MB)")
    print(f"yield  {(out_count / in_count * 100 if in_count else 0):.1f}%")
    print(f"avg compression ratio: {avg:.2f}  (1.0 = no compression, 0.5 = halved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
