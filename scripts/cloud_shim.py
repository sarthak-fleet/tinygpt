#!/usr/bin/env python3
"""OpenAI-compatible shim over the `claude -p` CLI — cloud-model baseline
for the gate runners. Same contract as fm_shim.py. Each request shells out
to headless Claude (bills the user's account; use for bounded eval runs).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

LATENCIES: list[float] = []


def claude_call(system: str, user: str, timeout: float = 120.0) -> str:
    prompt = (
        f"{system}\n\n"
        "Respond with ONLY the JSON object, no markdown fences, no other text.\n\n"
        f"{user}"
    )
    t0 = time.time()
    result = subprocess.run(
        ["claude", "-p", "--output-format", "text", prompt],
        capture_output=True, text=True, timeout=timeout,
    )
    LATENCIES.append((time.time() - t0) * 1000)
    if result.returncode != 0:
        raise RuntimeError(f"claude -p failed: {result.stderr[:200]}")
    text = result.stdout.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
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
            text = claude_call(system, user)
            status, payload = 200, {
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
                "model": "claude-cli",
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

    def do_GET(self):
        data = json.dumps({"data": [{"id": "claude-cli"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8767)
    args = p.parse_args()
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"cloud_shim (claude -p) on :{args.port}", file=sys.stderr)
    try:
        server.serve_forever()
    finally:
        if LATENCIES:
            lat = sorted(LATENCIES)
            print(f"calls={len(lat)} p50={lat[len(lat)//2]:.0f}ms", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
