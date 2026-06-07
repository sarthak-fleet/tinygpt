#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/.tinygpt/repo-specialist}"
PORT="${TINYGPT_PORT:-8080}"

cmd="${1:-help}"

case "$cmd" in
  corpus)
    repo="${2:-$PWD}"
    mkdir -p "$OUT_DIR"
    python3 - "$repo" "$OUT_DIR/corpus.jsonl" <<'PY'
import json
import pathlib
import sys

repo = pathlib.Path(sys.argv[1]).resolve()
out = pathlib.Path(sys.argv[2])
suffixes = {
    ".swift", ".py", ".ts", ".tsx", ".js", ".jsx", ".astro", ".md",
    ".rs", ".go", ".c", ".cc", ".cpp", ".h", ".hpp", ".json", ".yaml", ".yml"
}
skip_parts = {".git", ".build", "node_modules", "dist", "build", "__pycache__"}
rows = 0
with out.open("w", encoding="utf-8") as fh:
    for path in repo.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if any(part in skip_parts for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if not text.strip():
            continue
        rel = path.relative_to(repo).as_posix()
        fh.write(json.dumps({"path": rel, "text": text}, ensure_ascii=False) + "\n")
        rows += 1
print(f"wrote {rows} rows to {out}")
PY
    ;;
  continue)
    cat <<EOF
name: TinyGPT Local Specialist
version: 0.0.1
schema: v1
models:
  - name: TinyGPT Repo Specialist
    provider: openai
    model: tinygpt
    apiBase: http://127.0.0.1:${PORT}/v1
    apiKey: not-needed
    roles:
      - chat
      - edit
      - apply
    capabilities:
      - tool_use
EOF
    ;;
  aider)
    cat <<EOF
export OPENAI_API_BASE=http://127.0.0.1:${PORT}/v1
export OPENAI_API_KEY=not-needed
aider --model openai/tinygpt
EOF
    ;;
  help|*)
    cat <<'EOF'
usage:
  examples/repo-specialist-cli/run.sh corpus [repo]
  examples/repo-specialist-cli/run.sh continue
  examples/repo-specialist-cli/run.sh aider
EOF
    ;;
esac
