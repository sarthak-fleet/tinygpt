#!/usr/bin/env python3
"""Streaming partial-JSON parser — prototype for #274.

For Pace's JARVIS-class response path: the planner emits JSON of the
shape {"spokenText": "...", "pointAtLabel": "...", "clickLabel": "..."}.
Pace wants to start TTS on `spokenText` AS IT GENERATES, not after
the full JSON arrives. That cuts ~200-400ms off perceived latency.

This parser consumes the serve's SSE token stream and emits structured
events for each field:

  on_field_start(name)              # the key just opened
  on_field_chunk(name, partial)     # incremental text inside the value
  on_field_complete(name, value)    # the value's closing quote arrived

Usage as a library:

    parser = PartialJSONStream()
    parser.on_chunk = lambda f, t: print(f"{f}: {t}", end="", flush=True)
    parser.on_complete = lambda f, v: print(f"\n[{f} = {v!r}]")
    for tok in tinygpt_serve_sse_stream(prompt):
        parser.feed(tok)

Usage as a CLI to validate against a live serve:

    python3 scripts/partial_json_stream.py \\
      --serve-url http://127.0.0.1:8765/v1/chat/completions \\
      --prompt "click the save button" \\
      --elements "[0] button|548,40|save button|Save Draft"

State machine: just enough JSON to handle Pace's flat-string schema.
Does NOT handle nested objects/arrays in values (Pace doesn't need them).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional


class PartialJSONStream:
    """Incremental parser for `{"key": "value", "key2": "value2", ...}` shapes.

    Fires three callbacks:
      on_start(field_name)       — emitted when we open the value-string for `field_name`
      on_chunk(field_name, text) — emitted with each new character(s) inside the value
      on_complete(field_name, value) — emitted when the closing `"` arrives
    """

    # State machine states
    BEFORE_OBJECT = 0   # waiting for the opening '{'
    IN_OBJECT = 1       # inside object, between fields
    READING_KEY = 2     # inside a key-string
    AFTER_KEY = 3       # after closing key-quote, before ':'
    AFTER_COLON = 4     # after ':', before value
    READING_VALUE = 5   # inside a value-string
    AFTER_VALUE = 6     # after closing value-quote, before ',' or '}'

    def __init__(self,
                 on_start: Optional[Callable[[str], None]] = None,
                 on_chunk: Optional[Callable[[str, str], None]] = None,
                 on_complete: Optional[Callable[[str, str], None]] = None):
        self.on_start = on_start or (lambda f: None)
        self.on_chunk = on_chunk or (lambda f, t: None)
        self.on_complete = on_complete or (lambda f, v: None)
        self._reset()

    def _reset(self):
        self.state = self.BEFORE_OBJECT
        self.key_buf = ""
        self.value_buf = ""
        self.current_field: Optional[str] = None
        self.escape_next = False  # inside string, after `\`
        self.complete_fields: dict[str, str] = {}

    def feed(self, chunk: str) -> None:
        """Feed incremental text chunks. Triggers callbacks as events fire."""
        for ch in chunk:
            self._step(ch)

    def _step(self, ch: str) -> None:
        s = self.state
        if s == self.BEFORE_OBJECT:
            if ch == '{':
                self.state = self.IN_OBJECT
        elif s == self.IN_OBJECT:
            if ch == '"':
                self.key_buf = ""
                self.state = self.READING_KEY
            elif ch == '}':
                self.state = self.BEFORE_OBJECT  # done
        elif s == self.READING_KEY:
            if self.escape_next:
                self.key_buf += ch
                self.escape_next = False
            elif ch == '\\':
                self.escape_next = True
            elif ch == '"':
                self.state = self.AFTER_KEY
            else:
                self.key_buf += ch
        elif s == self.AFTER_KEY:
            if ch == ':':
                self.state = self.AFTER_COLON
        elif s == self.AFTER_COLON:
            if ch == '"':
                # Value-string begins.
                self.current_field = self.key_buf
                self.value_buf = ""
                self.on_start(self.current_field)
                self.state = self.READING_VALUE
            elif ch.isspace():
                pass  # ignore whitespace
            # Note: non-string values (numbers, bool, null) not supported in this prototype.
            # Pace's schema only has strings for now.
        elif s == self.READING_VALUE:
            if self.escape_next:
                # Decode common escapes; for everything else, pass through verbatim.
                decoded = {
                    'n': '\n', 't': '\t', 'r': '\r',
                    '"': '"', '\\': '\\', '/': '/',
                }.get(ch, ch)
                self.value_buf += decoded
                self.on_chunk(self.current_field, decoded)
                self.escape_next = False
            elif ch == '\\':
                self.escape_next = True
            elif ch == '"':
                # Value complete.
                val = self.value_buf
                fld = self.current_field
                self.complete_fields[fld] = val
                self.on_complete(fld, val)
                self.current_field = None
                self.value_buf = ""
                self.state = self.AFTER_VALUE
            else:
                self.value_buf += ch
                self.on_chunk(self.current_field, ch)
        elif s == self.AFTER_VALUE:
            if ch == ',':
                self.state = self.IN_OBJECT
            elif ch == '}':
                self.state = self.BEFORE_OBJECT


# ----------------- Live serve test -----------------


def stream_from_serve(url: str, prompt: str, model: str, system_prompt: str,
                       max_tokens: int = 200) -> None:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"},
                                  method="POST")

    parser = PartialJSONStream(
        on_start=lambda f: print(f"\n[start  {f}]", flush=True),
        on_chunk=lambda f, t: print(t, end="", flush=True),
        on_complete=lambda f, v: print(f"\n[complete {f} = {v!r}]", flush=True),
    )

    t_first_chunk = None
    t_first_field_start = None
    t_first_spoken_chunk = None
    n_chunks = 0
    t0 = time.time()

    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = event.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if not delta:
                continue
            n_chunks += 1
            if t_first_chunk is None:
                t_first_chunk = time.time()

            # Wrap on_start to record first field-start timing.
            def _on_start(f):
                nonlocal t_first_field_start
                if t_first_field_start is None:
                    t_first_field_start = time.time()
                print(f"\n[start  {f}]", flush=True)

            def _on_chunk(f, t):
                nonlocal t_first_spoken_chunk
                if f == "spokenText" and t_first_spoken_chunk is None:
                    t_first_spoken_chunk = time.time()
                print(t, end="", flush=True)

            parser.on_start = _on_start
            parser.on_chunk = _on_chunk
            parser.feed(delta)

    print()
    print()
    print(f"=== timing ===")
    print(f"  first SSE chunk:   {(t_first_chunk - t0) * 1000:.0f}ms after request" if t_first_chunk else "  (no chunks)")
    print(f"  first field start: {(t_first_field_start - t0) * 1000:.0f}ms" if t_first_field_start else "")
    print(f"  first spokenText:  {(t_first_spoken_chunk - t0) * 1000:.0f}ms" if t_first_spoken_chunk else "")
    print(f"  total chunks:      {n_chunks}")
    print(f"  fields completed:  {list(parser.complete_fields.keys())}")


# ----------------- Unit tests -----------------


def _run_unit_tests():
    """Verify the parser on synthetic streams without a serve."""
    events: list = []
    parser = PartialJSONStream(
        on_start=lambda f: events.append(("start", f)),
        on_chunk=lambda f, t: events.append(("chunk", f, t)),
        on_complete=lambda f, v: events.append(("complete", f, v)),
    )

    # Stream the JSON one char at a time to simulate worst-case token boundaries.
    json_str = '{"spokenText":"clicking save","pointAtLabel":"save","clickLabel":"save"}'
    for ch in json_str:
        parser.feed(ch)

    # Expected: 3 fields × (start, N chunks, complete)
    starts = [e for e in events if e[0] == "start"]
    completes = [e for e in events if e[0] == "complete"]
    assert starts == [("start", "spokenText"), ("start", "pointAtLabel"), ("start", "clickLabel")], starts
    assert completes == [
        ("complete", "spokenText", "clicking save"),
        ("complete", "pointAtLabel", "save"),
        ("complete", "clickLabel", "save"),
    ], completes
    # Verify chunks aggregate to the value
    spoken_chunks = [e[2] for e in events if e[0] == "chunk" and e[1] == "spokenText"]
    assert "".join(spoken_chunks) == "clicking save"

    # Test with multi-char chunks (more realistic — model emits multi-char SSE tokens)
    events.clear()
    parser._reset()
    chunks = ['{"spo', 'kenText":"hel', 'lo world","poin', 'tAtLabel":"x","clickLabel":"x"}']
    for c in chunks:
        parser.feed(c)
    assert ("complete", "spokenText", "hello world") in events

    # Test with escape sequences
    events.clear()
    parser._reset()
    parser.feed('{"spokenText":"line1\\nline2","pointAtLabel":"","clickLabel":""}')
    assert ("complete", "spokenText", "line1\nline2") in events, events

    print("✓ unit tests pass")


# ----------------- CLI -----------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unit", action="store_true", help="run unit tests only, no serve")
    p.add_argument("--serve-url", default="http://127.0.0.1:8765/v1/chat/completions")
    p.add_argument("--model", default="tinygpt")
    p.add_argument("--prompt", default="click the save button")
    p.add_argument("--elements", action="append", default=[],
                     help="repeatable; each is one ELEMENT line")
    p.add_argument("--sys-prompt", type=Path,
                     default=Path("/Users/sarthak/Desktop/fleet/tinygpt/grammars/pace-system-prompt-v6-label.txt"))
    p.add_argument("--max-tokens", type=int, default=200)
    args = p.parse_args()

    if args.unit:
        _run_unit_tests()
        return

    sys_prompt = args.sys_prompt.read_text().strip()

    # Build user prompt from elements + intent
    parts = []
    if args.elements:
        parts.append("on-screen elements:")
        parts.extend(args.elements)
        parts.append("")
    parts.append(f"user said: {args.prompt}")
    user_prompt = "\n".join(parts)

    stream_from_serve(args.serve_url, user_prompt, args.model, sys_prompt,
                       max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
