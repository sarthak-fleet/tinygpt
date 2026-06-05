#!/usr/bin/env bash
# scripts/nightly.sh — pick the next nightly training job and run it.
#
# Usage:
#   ./scripts/nightly.sh             # run next pending job
#   ./scripts/nightly.sh --list      # show queue + done status, exit
#   ./scripts/nightly.sh --dry       # show what would run, exit
#
# Picks the lowest-numbered scripts/nightly/N*.sh whose `.done` marker
# under ~/.cache/tinygpt/nightly/done/ doesn't exist, runs it under
# caffeinate -di, logs to ~/.cache/tinygpt/nightly/logs/. On success,
# touches the .done marker and posts a Mac notification. On failure,
# notifies with the error tail and leaves the .done marker absent so a
# rerun picks it up again.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS_DIR="$REPO_ROOT/scripts/nightly"
NIGHTLY_HOME="$HOME/.cache/tinygpt/nightly"
DONE_DIR="$NIGHTLY_HOME/done"
LOG_DIR="$NIGHTLY_HOME/logs"
mkdir -p "$DONE_DIR" "$LOG_DIR"

notify() {
    # Best-effort Mac notification. Silently succeeds if osascript is
    # missing (e.g., headless CI).
    osascript -e "display notification \"$2\" with title \"$1\"" 2>/dev/null || true
}

# Collect jobs in lex order (N01 < N02 < ...).
shopt -s nullglob
jobs=("$JOBS_DIR"/N*.sh)
shopt -u nullglob

if [[ ${#jobs[@]} -eq 0 ]]; then
    echo "no jobs under $JOBS_DIR" >&2
    notify "TinyGPT nightly" "queue is empty (no N*.sh under scripts/nightly)"
    exit 0
fi

# --list / --dry: report state and exit.
if [[ "${1:-}" == "--list" ]] || [[ "${1:-}" == "--dry" ]]; then
    for job in "${jobs[@]}"; do
        name="$(basename "$job" .sh)"
        if [[ -f "$DONE_DIR/$name.done" ]]; then
            echo "  done   $name"
        else
            echo "  queued $name"
        fi
    done
    exit 0
fi

# Pick first pending.
NEXT=""
for job in "${jobs[@]}"; do
    name="$(basename "$job" .sh)"
    if [[ ! -f "$DONE_DIR/$name.done" ]]; then
        NEXT="$job"
        NAME="$name"
        break
    fi
done

if [[ -z "$NEXT" ]]; then
    echo "queue empty — all ${#jobs[@]} jobs marked done."
    echo "to rerun a job, delete its .done marker under $DONE_DIR"
    notify "TinyGPT nightly" "queue complete — ${#jobs[@]} done"
    exit 0
fi

TS="$(date +%Y-%m-%d_%H%M)"
LOG="$LOG_DIR/$TS-$NAME.log"

# Make build artifacts addressable by both name + symlink for the dashboard.
LATEST_LOG="$LOG_DIR/latest.log"
ln -sf "$LOG" "$LATEST_LOG"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting $NAME"
echo "  log: $LOG"
echo "  cmd: bash $NEXT"
notify "TinyGPT nightly" "Starting $NAME"

START=$(date +%s)

# caffeinate -di:
#   -d  prevent display sleep (keeps GPU available)
#   -i  prevent idle sleep
# We deliberately don't use -m (disk) or -s (system) — those are too
# heavy and not needed.
if caffeinate -di bash "$NEXT" >"$LOG" 2>&1; then
    END=$(date +%s)
    DUR=$((END - START))
    H=$((DUR / 3600))
    M=$(((DUR % 3600) / 60))
    touch "$DONE_DIR/$NAME.done"
    # Pull a short summary from the tail — last 5 non-empty lines.
    SUMMARY="$(grep -v '^$' "$LOG" | tail -5 | tr '\n' ' ' | head -c 200)"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ $NAME complete in ${H}h${M}m"
    notify "TinyGPT nightly ✓" "$NAME complete in ${H}h${M}m · $SUMMARY"
else
    EXIT=$?
    END=$(date +%s)
    DUR=$((END - START))
    TAIL="$(grep -v '^$' "$LOG" | tail -5 | tr '\n' ' ' | head -c 200)"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ $NAME failed (exit $EXIT) after ${DUR}s"
    notify "TinyGPT nightly ✗" "$NAME FAILED ($EXIT) · $TAIL"
    exit "$EXIT"
fi
