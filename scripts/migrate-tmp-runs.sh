#!/usr/bin/env bash
# scripts/migrate-tmp-runs.sh — one-time helper to move surviving /tmp
# training artifacts into ~/.cache/tinygpt/runs/<name>/.
#
# Usage:
#   ./scripts/migrate-tmp-runs.sh          # interactive
#   ./scripts/migrate-tmp-runs.sh --yes    # move all without prompting

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="$HOME/.cache/tinygpt/runs"
AUTO_YES=false

if [[ "${1:-}" == "--yes" ]]; then
    AUTO_YES=true
fi

shopt -s nullglob
mapfile -t CKPTS < <(ls -1 /tmp/*.tinygpt 2>/dev/null | grep -v '\.step-' | grep -v '\.best\.' | grep -v '\.tmp$' || true)

if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "No canonical /tmp/*.tinygpt checkpoints found."
    exit 0
fi

echo "Found ${#CKPTS[@]} candidate run(s) under /tmp:"
for c in "${CKPTS[@]}"; do
    echo "  - $c"
done
echo ""

mkdir -p "$RUNS_ROOT"

for ckpt in "${CKPTS[@]}"; do
    base="$(basename "$ckpt" .tinygpt)"
    dest_dir="$RUNS_ROOT/$base"
    dest_ckpt="$dest_dir/$base.tinygpt"

    if [[ -f "$dest_ckpt" ]]; then
        echo "skip $base — already exists at $dest_ckpt"
        continue
    fi

    if ! $AUTO_YES; then
        read -r -p "Move $base and sidecars to $dest_dir? [y/N] " ans
        [[ "${ans,,}" == "y" || "${ans,,}" == "yes" ]] || continue
    fi

    mkdir -p "$dest_dir"
    mv "$ckpt" "$dest_ckpt"

    for sidecar in \
        "/tmp/$base.tinygpt.opt" \
        "/tmp/$base.best.tinygpt" \
        "/tmp/$base.best.meta.json" \
        "/tmp/$base.jsonl" \
        "/tmp/$base.history.jsonl" \
        /tmp/"$base".step-*.tinygpt \
        /tmp/"$base".step-*.tinygpt.opt
    do
        for f in $sidecar; do
            [[ -e "$f" ]] || continue
            mv "$f" "$dest_dir/"
        done
    done

    cat > "$dest_dir/README.md" <<EOF
# Migrated from /tmp

This run was moved by \`scripts/migrate-tmp-runs.sh\` on $(date -u +%Y-%m-%dT%H:%M:%SZ).

Original stem: \`$base\`
Canonical: \`$dest_ckpt\`
EOF

    echo "✓ migrated → $dest_dir"
done

echo ""
echo "Done. Inspect with: du -sh $RUNS_ROOT/*"
