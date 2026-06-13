#!/usr/bin/env python3
"""Two-stage architecture: rule-based ambiguity detector wraps any planner.

If the request is clearly ambiguous, this shim returns an intent=clarify
response directly. Otherwise it forwards to the upstream OpenAI-compatible
endpoint. The detection is a counting problem (visible elements, missing
slots, bare pronouns) — exactly what the 4B-Instruct's own intent
classifier fails to do reliably.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

UPSTREAM = ""
UPSTREAM_MODEL = ""

PRONOUNS = {"it", "that", "this", "them", "those", "these", "him", "her"}
REFERENT_VERBS = {
    "click", "press", "tap", "open", "send", "archive", "delete", "remove",
    "share", "forward", "reply", "pin", "duplicate", "rename", "close",
    "queue", "play", "join", "post", "download", "edit", "show", "hide",
}
TIME_VERBS_RE = re.compile(
    r"\b(remind\s+me|schedule|set\s+a\s+timer|wake\s+me|alarm|notify\s+me\s+at)\b", re.I)
TIME_PRESENT_RE = re.compile(
    r"\b(at|on|in|by|after|before|tomorrow|today|tonight|yesterday|"
    r"morning|afternoon|evening|night|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday|\d+\s*(?:am|pm|minutes?|hours?|days?))\b", re.I)
RECIPIENT_VERBS_RE = re.compile(
    r"\b(text|email|message|ping|dm|send.*to|draft.*to|tell|forward.*to)\b", re.I)
RECIPIENT_PRESENT_RE = re.compile(
    r"\b(to\s+\w+|@\w+|\w+@\w+|__resolve:\w+)\b", re.I)


def parse_request(user_content: str) -> tuple[str, list[tuple[str, str]]]:
    user_phrase = ""
    elements: list[tuple[str, str]] = []
    in_elements = False
    for line in user_content.splitlines():
        s = line.strip()
        if s.startswith("on-screen elements:"):
            in_elements = True
            continue
        if s.startswith("user said:"):
            user_phrase = s[len("user said:"):].strip()
            in_elements = False
            continue
        if in_elements and s.startswith("["):
            m = re.match(r"\[\d+\]\s+([^|]+)\|[^|]+\|([^|]+)\|", s)
            if m:
                elements.append((m.group(2).strip(), m.group(1).strip()))
    return user_phrase, elements


def tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", s.lower())
            if w not in {"the", "a", "an", "my", "to", "for", "on", "in", "of",
                         "and", "this", "that", "it", "is"}}


def detect_ambiguity(user_phrase: str, elements: list[tuple[str, str]]) -> tuple[str, str] | None:
    phrase = user_phrase.lower()
    user_toks = tokens(user_phrase)

    if RECIPIENT_VERBS_RE.search(phrase) and not RECIPIENT_PRESENT_RE.search(phrase):
        if not any(user_toks & tokens(label) for label, _ in elements):
            return ("who is the recipient?", "recipient")

    if TIME_VERBS_RE.search(phrase) and not TIME_PRESENT_RE.search(phrase):
        return ("what time?", "time")

    has_pronoun = bool(set(phrase.split()) & PRONOUNS)
    if has_pronoun:
        if len(elements) == 0 or len(elements) > 1:
            return ("which target do you mean?", "target")

    verbs_in_phrase = user_toks & REFERENT_VERBS
    if verbs_in_phrase:
        matches = [(label, role) for label, role in elements
                   if user_toks & tokens(label)]
        if len(matches) >= 2 and len({r for _, r in matches}) == 1:
            # Use the role as topic keyword (matches "which app", "which draft" patterns)
            role_word = matches[0][1].split('_')[0] if '_' in matches[0][1] else matches[0][1]
            first_two = " or ".join(m[0] for m in matches[:2])
            return (f"which {role_word} — {first_two}?", f"which {role_word}")
        if not matches and elements:
            return ("which target do you mean?", "target")
    return None


def clarify_response(question: str, topic: str) -> dict:
    payload = {"spokenText": question, "intent": "clarify",
               "payload": {"question": question, "topic": topic}}
    return {
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": json.dumps(payload)}}],
        "model": f"two-stage-{UPSTREAM_MODEL}", "object": "chat.completion",
        "_two_stage": {"intercepted": True, "rule": topic},
    }


def forward_upstream(body: dict) -> dict:
    payload = dict(body)
    payload["model"] = UPSTREAM_MODEL or payload.get("model", "")
    req = urllib.request.Request(
        UPSTREAM, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    raw = urllib.request.urlopen(req, timeout=120).read()
    out = json.loads(raw)
    out["_two_stage"] = {"intercepted": False}
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user_content = ""
        for m in body.get("messages", []):
            if m["role"] == "user" and isinstance(m["content"], str):
                user_content = m["content"]; break
        phrase, elements = parse_request(user_content)
        ambig = detect_ambiguity(phrase, elements)
        if ambig:
            self._send(200, clarify_response(*ambig))
            return
        try:
            self._send(200, forward_upstream(body))
        except Exception as e:
            self._send(500, {"error": {"message": str(e)}})

    def do_GET(self):
        self._send(200, {"data": [{"id": f"two-stage-{UPSTREAM_MODEL}"}]})

    def _send(self, status: int, payload: dict):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    global UPSTREAM, UPSTREAM_MODEL
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", required=True)
    p.add_argument("--model", default="")
    p.add_argument("--port", type=int, default=8769)
    args = p.parse_args()
    UPSTREAM, UPSTREAM_MODEL = args.upstream, args.model
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"two-stage shim on :{args.port} → {UPSTREAM}", file=sys.stderr)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
