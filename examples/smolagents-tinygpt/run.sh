#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TINYGPT_BIN="${TINYGPT_BIN:-$ROOT/native-mac/.build/arm64-apple-macosx/release/tinygpt}"
TINYGPT_MODEL="${TINYGPT_MODEL:-}"
PORT="${TINYGPT_PORT:-8080}"

if [[ -z "$TINYGPT_MODEL" ]]; then
  echo "Set TINYGPT_MODEL=/path/to/model.tinygpt" >&2
  exit 2
fi

if [[ ! -x "$TINYGPT_BIN" ]]; then
  echo "tinygpt binary not found at $TINYGPT_BIN; run swift build -c release --product tinygpt" >&2
  exit 2
fi

python3 -m pip install -q smolagents openai

"$TINYGPT_BIN" serve "$TINYGPT_MODEL" --host 127.0.0.1 --port "$PORT" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
sleep 2

python3 - <<'PY'
import os
from smolagents import CodeAgent, OpenAIServerModel, tool

port = os.environ.get("TINYGPT_PORT", "8080")
model = OpenAIServerModel(
    model_id="tinygpt",
    api_base=f"http://127.0.0.1:{port}/v1",
    api_key="not-needed",
)

@tool
def lookup_status(ticket_id: str) -> str:
    """Return the status for a support ticket."""
    return {"A-100": "ready", "B-200": "blocked"}.get(ticket_id, "unknown")

agent = CodeAgent(tools=[lookup_status], model=model)
print(agent.run("Check ticket A-100 and answer in one sentence."))
PY
