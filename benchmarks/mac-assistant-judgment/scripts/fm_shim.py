#!/usr/bin/env python3
"""OpenAI-compatible HTTP shim over fm_bridge (Apple Foundation Models).

Lets the v11 ship-gate runners (eval_pace_v2 / eval_pace_unhappy /
eval_bfcl) evaluate Apple's on-device model unchanged:

  swiftc -O scripts/fm_bridge.swift -o /tmp/fm_bridge
  python3 scripts/fm_shim.py --port 8766 &
  python3 scripts/eval_pace_v2.py --serve-url http://127.0.0.1:8766/v1/chat/completions ...

Single-threaded by design: one persistent bridge subprocess, requests
serialized (the FM session is per-request anyway). Logs per-call latency.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

BRIDGE = None
LATENCIES: list[float] = []


def bridge_call(system: str, user: str, timeout: float = 120.0) -> str:
    line = json.dumps({"system": system, "user": user}) + "\n"
    BRIDGE.stdin.write(line)
    BRIDGE.stdin.flush()
    t0 = time.time()
    reply = BRIDGE.stdout.readline()
    LATENCIES.append((time.time() - t0) * 1000)
    if not reply:
        raise RuntimeError("fm_bridge died")
    doc = json.loads(reply)
    if "error" in doc:
        raise RuntimeError(doc["error"])
    text = doc["text"].strip()
    # FM habitually wraps JSON in markdown fences; Pace's production client
    # uses typed @Generable output so fences are a bridge artifact, not a
    # model capability difference. Strip them for the gate's JSON parsers.
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = body.get("messages", [])
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n".join(
            m["content"] for m in messages
            if m["role"] == "user" and isinstance(m["content"], str)
        )
        try:
            text = bridge_call(system, user)
            status, payload = 200, {
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
                "model": "apple-foundation-models",
                "object": "chat.completion",
            }
        except Exception as e:
            status, payload = 500, {"error": {"message": str(e)}}
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # /v1/models readiness probe
        data = json.dumps({"data": [{"id": "apple-foundation-models"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    global BRIDGE
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--bridge", default="/tmp/fm_bridge")
    args = p.parse_args()

    BRIDGE = subprocess.Popen(
        [args.bridge], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"fm_shim on :{args.port} → {args.bridge}", file=sys.stderr)
    try:
        server.serve_forever()
    finally:
        if LATENCIES:
            lat = sorted(LATENCIES)
            print(f"calls={len(lat)} p50={lat[len(lat)//2]:.0f}ms", file=sys.stderr)
        BRIDGE.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
