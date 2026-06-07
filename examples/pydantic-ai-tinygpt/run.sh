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

python3 -m pip install -q pydantic-ai openai

"$TINYGPT_BIN" serve "$TINYGPT_MODEL" --host 127.0.0.1 --port "$PORT" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
sleep 2

python3 - <<'PY'
import os
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

class TicketRoute(BaseModel):
    team: str
    priority: int
    summary: str

port = os.environ.get("TINYGPT_PORT", "8080")
model = OpenAIChatModel(
    "tinygpt",
    provider=OpenAIProvider(
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key="not-needed",
    ),
)
agent = Agent(model, output_type=TicketRoute)
result = agent.run_sync("Route this ticket: customer cannot log in after SSO migration.")
print(result.output)
PY
