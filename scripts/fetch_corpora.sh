#!/bin/bash
# Fetch a varied set of Project Gutenberg corpora to /tmp/tinygpt-corpora/.
# All public domain. Strip the standard Gutenberg headers/footers so the
# byte-level model sees clean text only.
#
# Usage:
#   ./scripts/fetch_corpora.sh           # default 11 books
#   OUT_DIR=./data/corpora ./scripts/fetch_corpora.sh
#
# After running, /tmp/tinygpt-corpora/ has ~34 MB of varied prose:
#
#   shakespeare-complete.txt   5.6 MB   Shakespeare (Complete)
#   bible-kjv.txt              4.4 MB   King James Bible
#   les-miserables.txt         3.3 MB   Hugo
#   war-and-peace.txt          3.3 MB   Tolstoy
#   monte-cristo.txt           2.8 MB   Dumas
#   don-quixote.txt            2.4 MB   Cervantes
#   anna-karenina.txt          2.0 MB   Tolstoy
#   bleak-house.txt            2.0 MB   Dickens
#   middlemarch.txt            1.8 MB   Eliot
#   ulysses.txt                1.6 MB   Joyce
#   moby-dick.txt              1.3 MB   Melville
#   crime-and-punishment.txt   1.2 MB   Dostoevsky
#   pride-prejudice.txt         752 KB  Austen
#   huck-finn.txt               591 KB  Twain
#   sherlock-holmes.txt         588 KB  Doyle
#   frankenstein.txt            429 KB  Shelley
#   heart-of-darkness.txt       217 KB  Conrad
#   alice.txt                   151 KB  Carroll
#   modest-proposal.txt          20 KB  Swift
#
# Combine into themed mega-corpora with `cat`:
#   cat shakespeare-complete.txt pride-prejudice.txt middlemarch.txt bleak-house.txt > english-classics.txt
#   cat war-and-peace.txt anna-karenina.txt crime-and-punishment.txt > russian-lit.txt
#   cat moby-dick.txt frankenstein.txt monte-cristo.txt les-miserables.txt > adventure.txt
#   cat *.txt > everything.txt   # ~34 MB unified literary corpus

set -uo pipefail

OUT_DIR="${OUT_DIR:-/tmp/tinygpt-corpora}"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

fetch_book() {
  local id=$1
  local out=$2
  local name=$3
  if [ -s "$out" ]; then
    echo "  ✓ $name: already present ($(wc -c < $out) bytes)"
    return
  fi
  # Try multiple URL conventions — Gutenberg's filenames are inconsistent.
  for url in \
    "https://www.gutenberg.org/files/$id/$id-0.txt" \
    "https://www.gutenberg.org/files/$id/$id.txt" \
    "https://www.gutenberg.org/cache/epub/$id/pg$id.txt"
  do
    curl -fsSL --max-time 60 "$url" 2>/dev/null \
      | awk 'BEGIN{p=0} /\*\*\* START OF/{p=1; next} /\*\*\* END OF/{p=0} p{print}' > "$out.tmp"
    if [ -s "$out.tmp" ]; then
      mv "$out.tmp" "$out"
      echo "  ✓ $name: $(wc -c < $out) bytes"
      return
    fi
  done
  rm -f "$out.tmp"
  echo "  ✗ $name: all sources failed"
}

# Each line: gutenberg-id  output-filename  display-name
# Curated for byte-level training: English prose, narrative + dialogue,
# varied register from Victorian formal to Twain's vernacular.
books=(
  # Top tier — biggest individual works
  "100   shakespeare-complete.txt   Shakespeare (Complete Works)"
  "10    bible-kjv.txt              King James Bible"
  "2600  war-and-peace.txt          War and Peace (Tolstoy)"
  "135   les-miserables.txt         Les Misérables (Hugo)"
  "1184  monte-cristo.txt           Count of Monte Cristo (Dumas)"
  "996   don-quixote.txt            Don Quixote (Cervantes)"
  "1399  anna-karenina.txt          Anna Karenina (Tolstoy)"
  "1023  bleak-house.txt            Bleak House (Dickens)"
  "145   middlemarch.txt            Middlemarch (Eliot)"
  "4300  ulysses.txt                Ulysses (Joyce)"
  "2554  crime-and-punishment.txt   Crime and Punishment (Dostoevsky)"
  "2701  moby-dick.txt              Moby Dick (Melville)"
  # Mid-tier
  "1342  pride-prejudice.txt        Pride and Prejudice (Austen)"
  "76    huck-finn.txt              Huckleberry Finn (Twain)"
  "1661  sherlock-holmes.txt        Adventures of Sherlock Holmes (Doyle)"
  "84    frankenstein.txt           Frankenstein (Shelley)"
  "219   heart-of-darkness.txt      Heart of Darkness (Conrad)"
  "11    alice.txt                  Alice in Wonderland (Carroll)"
  "1080  modest-proposal.txt        A Modest Proposal (Swift)"
)

echo "Fetching ${#books[@]} corpora to $OUT_DIR"
for line in "${books[@]}"; do
  read -r id out name <<< "$line"
  fetch_book "$id" "$out" "$name"
done

echo ""
echo "Sizes:"
ls -la *.txt 2>/dev/null | awk '{printf "  %10d  %s\n", $5, $9}' | sort -n

total=$(cat *.txt 2>/dev/null | wc -c)
echo ""
echo "Total: $total bytes ($(echo "scale=1; $total / 1048576" | bc) MB)"
