#!/usr/bin/env python3
"""Best-of-N + cautious-critic shim over a local OpenAI-compatible model.

Test-time compute, not training: same 4B base, ~N× inference cost, the
critic biases toward asking/refusing/confirming when ANY of the N samples
emits a cautious intent. Goal: punch above the model's zero-shot weight
on the clarify/refusal dimensions where the universal failure lives.

Critic v1 (this file): cautious-intent priority.
  - clarify > confirm_destructive > out_of_scope > everything else
  - within a priority tier, pick the response with longest spokenText
    (rough proxy for "the model articulated a reason")
  - if no cautious intent in any sample, return the majority-action sample

Usage:
  python3 scripts/bon_shim.py --upstream http://127.0.0.1:1234/v1/chat/completions \\
                              --model qwen3-4b-instruct-2507 \\
                              --samples 8 --port 8768
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer

UPSTREAM = ""
UPSTREAM_MODEL = ""
SAMPLES = 8
SAMPLE_TEMPERATURE = 0.7  # need diversity for best-of-N to do anything

CAUTIOUS_PRIORITY = ["clarify", "confirm_destructive", "out_of_scope"]


def one_sample(body: dict, temperature: float, timeout: float = 120.0) -> str:
    payload = dict(body)
    payload["temperature"] = temperature
    payload["model"] = UPSTREAM_MODEL or payload.get("model", "")
    payload.pop("n", None)
    req = urllib.request.Request(
        UPSTREAM, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    return json.loads(raw)["choices"][0]["message"]["content"]


def parse_intent(text: str) -> tuple[str | None, dict | None]:
    text = text.strip()
    # tolerate markdown fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, None
    try:
        doc = json.loads(m.group(0))
    except Exception:
        return None, None
    if not isinstance(doc, dict):
        return None, None
    return doc.get("intent"), doc


def pick(samples: list[str]) -> str:
    """Cautious critic v1: any cautious intent wins; else majority action."""
    parsed = [(s, *parse_intent(s)) for s in samples]
    parsed = [(s, intent, doc) for s, intent, doc in parsed if intent and doc]
    if not parsed:
        return samples[0]  # all unparseable; defer to upstream's first

    # cautious-intent priority
    for tier in CAUTIOUS_PRIORITY:
        tier_hits = [(s, doc) for s, intent, doc in parsed if intent == tier]
        if tier_hits:
            # within a tier, prefer the response that articulated more
            tier_hits.sort(key=lambda sd: len(str(sd[1].get("spokenText", ""))),
                           reverse=True)
            return tier_hits[0][0]

    # no cautious intent in any sample — return majority intent's first sample
    intents = Counter(intent for _, intent, _ in parsed)
    top_intent, _ = intents.most_common(1)[0]
    for s, intent, _ in parsed:
        if intent == top_intent:
            return s
    return samples[0]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=SAMPLES) as pool:
            futures = [pool.submit(one_sample, body, SAMPLE_TEMPERATURE)
                       for _ in range(SAMPLES)]
            samples = []
            for f in futures:
                try:
                    samples.append(f.result())
                except Exception:
                    pass  # one bad upstream call is fine, others speak
        if not samples:
            self._send(500, {"error": {"message": "all upstream samples failed"}})
            return
        chosen = pick(samples)
        ms = (time.time() - t0) * 1000
        # diagnostics in headers, body stays OpenAI-shaped
        intents = Counter(parse_intent(s)[0] or "none" for s in samples)
        intent_summary = ",".join(f"{k}={v}" for k, v in intents.most_common())
        self._send(200, {
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": chosen}}],
            "model": f"bon-{SAMPLES}-{UPSTREAM_MODEL}",
            "object": "chat.completion",
            "_bon_diagnostics": {"samples": len(samples), "ms": round(ms),
                                 "intents": intent_summary},
        })

    def do_GET(self):
        self._send(200, {"data": [{"id": f"bon-{SAMPLES}-{UPSTREAM_MODEL}"}]})

    def _send(self, status: int, payload: dict):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    global UPSTREAM, UPSTREAM_MODEL, SAMPLES, SAMPLE_TEMPERATURE
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", required=True, help="OpenAI-compatible URL")
    p.add_argument("--model", default="", help="model id passed upstream")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--port", type=int, default=8768)
    args = p.parse_args()
    UPSTREAM, UPSTREAM_MODEL = args.upstream, args.model
    SAMPLES, SAMPLE_TEMPERATURE = args.samples, args.temperature
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"bon_shim on :{args.port} → {UPSTREAM} (N={SAMPLES}, T={SAMPLE_TEMPERATURE})",
          file=sys.stderr)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
