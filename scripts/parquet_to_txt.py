#!/usr/bin/env python3
"""parquet_to_txt.py — decode parquet shards to plain text (or JSONL).

Bridges the gap until `tinygpt download-dataset` learns parquet
natively. Most HuggingFace datasets ship as parquet shards; this
script unblocks the ones we already have on disk (FineWeb-Edu,
UltraFeedback, etc.).

Usage:
    parquet_to_txt.py <input> <output> [--field FIELD] [--jsonl] [--max-rows N]

<input> can be:
    - a single .parquet file
    - a directory (all .parquet files under it, sorted by filename)

Output mode:
    --jsonl    write one JSON record per line (default for SFT/DPO data
               where multiple fields matter)
    default    write one text document per row, separated by blank lines
               (right format for pretrain corpora like FineWeb-Edu)

--field FIELD selects which column to pull as text. Default: "text".
Fallback chain when FIELD is absent: text → content → instruction.

--max-rows N caps the total output (handy for sampling large corpora
or smoke-testing).

Examples:
    # FineWeb-Edu shard → pretrain text file
    parquet_to_txt.py \
        ~/.cache/tinygpt/datasets/HuggingFaceFW/fineweb-edu/data/CC-MAIN-2013-20/ \
        /tmp/fineweb-edu.txt --max-rows 200000

    # UltraFeedback → DPO JSONL (preserves prompt/chosen/rejected fields)
    parquet_to_txt.py \
        ~/.cache/tinygpt/datasets/HuggingFaceH4/ultrafeedback_binarized/data/ \
        ~/.cache/tinygpt/datasets/ultrafeedback.jsonl --jsonl
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:
    sys.exit("pyarrow not installed. Run: python3 -m pip install pyarrow")


def find_parquets(path: Path) -> list[Path]:
    """Return all .parquet files under path, sorted. Single-file input
    returns a one-element list."""
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.parquet"))
    sys.exit(f"no parquet at {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, help="parquet file or dir of parquets")
    p.add_argument("output", type=Path, help="output .txt or .jsonl")
    p.add_argument("--field", default="text",
                   help='text column to extract (default: "text"; falls back to content/instruction)')
    p.add_argument("--jsonl", action="store_true",
                   help="emit one JSON record per line instead of plain text")
    p.add_argument("--max-rows", type=int, default=None,
                   help="cap total rows written")
    args = p.parse_args()

    shards = find_parquets(args.input)
    if not shards:
        sys.exit(f"no .parquet shards under {args.input}")
    print(f"[{len(shards)} shard(s)] → {args.output}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    skipped = 0

    with args.output.open("w", encoding="utf-8") as out:
        for shard in shards:
            try:
                table = pq.read_table(shard)
            except Exception as exc:
                print(f"  ! could not read {shard.name}: {exc}", file=sys.stderr)
                continue

            cols = table.column_names
            # Pick the field. Use the user's --field if present; else fall
            # back. Fallback order matches the most common HF dataset shapes.
            field = (args.field if args.field in cols else
                     next((c for c in ("text", "content", "instruction") if c in cols), None))

            print(f"  {shard.name}: {table.num_rows} rows, columns={cols}, picking '{field}'",
                  file=sys.stderr)

            if args.jsonl:
                # Whole-row JSONL — preserve every column. Use Arrow's
                # iter_batches so we don't materialise the entire table
                # in memory for big shards.
                for batch in table.to_batches():
                    records = batch.to_pylist()
                    for rec in records:
                        if args.max_rows is not None and total >= args.max_rows:
                            break
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        total += 1
                    if args.max_rows is not None and total >= args.max_rows:
                        break
            else:
                # Plain text — one column, blank-line separator. Skip empty rows.
                if field is None:
                    print(f"  ! no text-ish column in {shard.name}; skipping",
                          file=sys.stderr)
                    continue
                col = table.column(field).to_pylist()
                for v in col:
                    if args.max_rows is not None and total >= args.max_rows:
                        break
                    if not v:
                        skipped += 1
                        continue
                    out.write(str(v))
                    out.write("\n\n")
                    total += 1

            if args.max_rows is not None and total >= args.max_rows:
                print(f"  ! hit --max-rows {args.max_rows}, stopping", file=sys.stderr)
                break

    size = args.output.stat().st_size
    print(f"wrote {total} rows ({size:,} bytes) to {args.output}"
          + (f"; skipped {skipped} empty" if skipped else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
