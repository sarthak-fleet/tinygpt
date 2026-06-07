#!/usr/bin/env python3
"""Build the theme-completer training corpus from fetched palette data.

Reads from ~/.cache/tinygpt/datasets/themes/{tailwind,open-color,radix-*}.{json,js,ts}
Writes ~/.cache/tinygpt/datasets/themes-train.jsonl

Each row = {"prompt": "...", "completion": "..."} — train a model to
predict the rest of a palette given 2-3 colors.

Hex colors are normalized: lowercase, always 6 digits, with leading #.
"""

import json
import re
import sys
from pathlib import Path
from itertools import combinations

CACHE = Path.home() / ".cache" / "tinygpt" / "datasets" / "themes"
OUT_JSONL = Path.home() / ".cache" / "tinygpt" / "datasets" / "themes-train.jsonl"

HEX_RE = re.compile(r"#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")


def normalize_hex(h: str) -> str:
    """Normalize hex like 'fff', 'FFFFFF', '#ffffff' → '#ffffff'."""
    h = h.lstrip("#").lower()
    if len(h) == 3:
        h = "".join(c + c for c in h)
    return "#" + h


def extract_hexes_from_text(text: str) -> list[str]:
    """Pull all hex codes from a blob of text/JSON/JS."""
    return [normalize_hex(m.group(1)) for m in HEX_RE.finditer(text)]


def load_palettes_from_open_color(path: Path) -> list[list[str]]:
    """Open Color: {'red': ['#fff5f5', '#ffe3e3', ...], 'pink': [...], ...}"""
    data = json.loads(path.read_text())
    palettes = []
    # Group all colors per hue as one "palette"
    for hue_name, shades in data.items():
        if isinstance(shades, list):
            palettes.append([normalize_hex(s) for s in shades])
    return palettes


def load_palettes_from_tailwind(path: Path) -> list[list[str]]:
    """Tailwind colors.js — JS object, parse loosely."""
    text = path.read_text()
    palettes = []
    # Each tailwind color hue is a block like:
    #   blue: { 50: '#eff6ff', 100: '#dbeafe', ... 900: '#1e3a8a' }
    # Find each block.
    for match in re.finditer(r"(\w+):\s*\{([^}]+)\}", text):
        block = match.group(2)
        hexes = extract_hexes_from_text(block)
        if len(hexes) >= 5:  # tailwind palettes have 10-11 shades
            palettes.append(hexes)
    return palettes


def load_palettes_from_blob(path: Path) -> list[list[str]]:
    """Generic fallback: extract all hexes, chunk into 5-color groups."""
    hexes = extract_hexes_from_text(path.read_text())
    if not hexes:
        return []
    # Conservative: just group sequentially
    out = []
    for i in range(0, len(hexes) - 4, 5):
        out.append(hexes[i : i + 5])
    return out


def build_training_pairs(palettes: list[list[str]]) -> list[dict]:
    """For each palette of N colors:
       - Pick K = 2 or 3 visible colors (input)
       - Output the remaining N-K colors

       For each palette of length N >= 5, generate multiple
       (input subset, output rest) pairs via combinations.
    """
    pairs = []
    for palette in palettes:
        n = len(palette)
        if n < 5:
            continue
        # Limit combinations to keep things small
        for k in (2, 3):
            for given_idxs in combinations(range(n), k):
                given = [palette[i] for i in given_idxs]
                rest = [palette[i] for i in range(n) if i not in given_idxs]
                if not rest:
                    continue
                prompt = "GIVEN: " + " ".join(given) + " → REST: "
                completion = " ".join(rest)
                pairs.append({"prompt": prompt, "completion": completion})
                # Cap pairs per palette to avoid combinatorial explosion
                if len(pairs) % 50 == 0 and len(pairs) > 5000:
                    return pairs
    return pairs


def main():
    if not CACHE.exists():
        print(f"error: {CACHE} not found. Run scripts/theme-data-fetch.sh first.", file=sys.stderr)
        sys.exit(1)

    all_palettes = []

    # Open Color
    oc = CACHE / "open-color.json"
    if oc.exists():
        p = load_palettes_from_open_color(oc)
        print(f"open-color: {len(p)} palettes")
        all_palettes.extend(p)

    # Tailwind
    tw = CACHE / "tailwind-colors.js"
    if tw.exists():
        p = load_palettes_from_tailwind(tw)
        print(f"tailwind:   {len(p)} palettes")
        all_palettes.extend(p)

    # Material
    mt = CACHE / "material-core.ts"
    if mt.exists():
        p = load_palettes_from_blob(mt)
        print(f"material:   {len(p)} palettes (blob extraction)")
        all_palettes.extend(p)

    if not all_palettes:
        print("error: no palettes parsed", file=sys.stderr)
        sys.exit(1)

    print(f"total palettes: {len(all_palettes)}")
    print(f"avg palette length: {sum(len(p) for p in all_palettes) / len(all_palettes):.1f}")

    pairs = build_training_pairs(all_palettes)
    print(f"training pairs: {len(pairs)}")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"wrote → {OUT_JSONL}")
    print()
    print("Sample rows:")
    for pair in pairs[:3]:
        print(f"  {pair['prompt']}{pair['completion'][:60]}...")


if __name__ == "__main__":
    main()
