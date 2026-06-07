#!/usr/bin/env bash
# scripts/duty-cycle-throttle.sh — throttle a long-running process via SIGSTOP/SIGCONT duty cycling.
#
# Usage:
#   ./scripts/duty-cycle-throttle.sh <PID> [duty=0.5] [period_secs=60]
#
# Throttles the target process to ~duty fraction of full load by toggling
# SIGCONT/SIGSTOP every (period * duty) / (period * (1-duty)) seconds.
#
# Preserves all in-memory state (Adam, weights, RNG, KV cache) — the
# process is frozen, not killed. Zero training-result impact.
#
# Stops automatically when the target PID disappears.
#
# To stop the throttler without affecting the target: kill the throttler
# script's PID (printed at start). The target will be left in whatever
# state it's currently in (probably STOPPED — send SIGCONT manually).

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <pid> [duty=0.5] [period_secs=60]" >&2
    exit 1
fi

PID=$1
DUTY=${2:-0.5}
PERIOD=${3:-60}

# Use python for floating-point math; bash's $(( )) is integer-only.
RUN_SECS=$(python3 -c "print(round($PERIOD * $DUTY, 1))")
PAUSE_SECS=$(python3 -c "print(round($PERIOD * (1 - $DUTY), 1))")

if ! kill -0 "$PID" 2>/dev/null; then
    echo "PID $PID not found" >&2
    exit 1
fi

echo "=== duty-cycle-throttle ==="
echo "  target PID:   $PID"
echo "  duty cycle:   $DUTY (${RUN_SECS}s on / ${PAUSE_SECS}s off per ${PERIOD}s)"
echo "  throttler PID: $$"
echo "  to stop:      kill $$"
echo "  to revert PID to full speed: kill $$ && kill -CONT $PID"
echo ""

CYCLE=0
while kill -0 "$PID" 2>/dev/null; do
    CYCLE=$((CYCLE + 1))
    kill -CONT "$PID" 2>/dev/null || break
    sleep "$RUN_SECS"
    kill -STOP "$PID" 2>/dev/null || break
    sleep "$PAUSE_SECS"
    # Log every 10 cycles
    if (( CYCLE % 10 == 0 )); then
        echo "[$(/bin/date '+%H:%M:%S')] cycle $CYCLE — target still alive"
    fi
done

echo "[$(/bin/date '+%H:%M:%S')] target PID $PID no longer running; exiting"
