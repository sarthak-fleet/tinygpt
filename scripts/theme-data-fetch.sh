#!/usr/bin/env bash
# scripts/theme-data-fetch.sh — fetch public theme palette data
# for the theme-completer specialist (path 1: from-scratch).
#
# Sources (ranked by quality):
#   1. Tailwind CSS    — npm @tailwindcss/colors
#   2. Material Design — Google's color spec JSON
#   3. Radix Colors    — npm @radix-ui/colors
#   4. Open Color      — github yeun/open-color
#   5. ColorHunt       — public 10K+ palette scrape
#
# Output: ~/.cache/tinygpt/datasets/themes/
#   ├── tailwind.json
#   ├── material.json
#   ├── radix.json
#   ├── open-color.json
#   ├── colorhunt-sample.json
#   └── themes-combined.jsonl  ← unified format

set -euo pipefail

OUT="$HOME/.cache/tinygpt/datasets/themes"
mkdir -p "$OUT"

echo "=== fetching theme palette data → $OUT ==="

# 1. Open Color — clean YAML, easy parse
echo "[1/5] Open Color (github yeun/open-color)..."
curl -s -L "https://raw.githubusercontent.com/yeun/open-color/master/open-color.json" \
    -o "$OUT/open-color.json"
echo "    $(wc -c < "$OUT/open-color.json") bytes"

# 2. Tailwind — pin to v3 colors which is the canonical reference
echo "[2/5] Tailwind CSS v3 colors..."
curl -s -L "https://raw.githubusercontent.com/tailwindlabs/tailwindcss/v3.4.1/src/public/colors.js" \
    -o "$OUT/tailwind-colors.js"
# (the .js exports the palette; we'll parse it in the build script)
echo "    $(wc -c < "$OUT/tailwind-colors.js") bytes"

# 3. Radix Colors — JSON export
echo "[3/5] Radix Colors..."
curl -s -L "https://raw.githubusercontent.com/radix-ui/colors/main/src/blue.ts" \
    -o "$OUT/radix-blue.ts"
# Radix is per-color files; we fetch the index
curl -s -L "https://api.github.com/repos/radix-ui/colors/contents/src" \
    -o "$OUT/radix-index.json"
echo "    radix index: $(wc -c < "$OUT/radix-index.json") bytes"

# 4. Material Design — colors from Material 2 spec
echo "[4/5] Material Design colors..."
curl -s -L "https://raw.githubusercontent.com/material-foundation/material-color-utilities/main/typescript/palettes/core_palette.ts" \
    -o "$OUT/material-core.ts" || echo "    (best-effort)"

# 5. ColorHunt sample — note: needs scraping (rate-limited)
echo "[5/5] ColorHunt sample — manual scrape needed"
echo "    skipping — set up colorhunt.co scraper separately"
echo "    suggested: 1000 palettes via https://colorhunt.co/palettes (page 1..100)"

echo ""
echo "=== fetched. To build unified JSONL, run:"
echo "    python3 scripts/theme-data-build.py"
echo ""
ls -la "$OUT"
