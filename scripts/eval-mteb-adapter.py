#!/usr/bin/env python3
"""Thin MTEB wrapper for tinygpt eval-mteb.

Runs a subset of BEIR/MTEB tasks and writes a compact JSON summary that
EvalMTEB.swift converts into E0 JSONL rows.

Usage (invoked by tinygpt eval-mteb, not directly):
  python3 scripts/eval-mteb-adapter.py --model BAAI/bge-small-en --hf \
      --tasks BEIR/scifact --limit 100 --results-json /tmp/out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--hf", action="store_true", help="Model is a HuggingFace hub id or local dir")
    args = parser.parse_args()

    try:
        import mteb  # type: ignore
    except ImportError:
        print("mteb not installed — run: pip install mteb", file=sys.stderr)
        return 1

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    tasks = []
    for name in task_names:
        try:
            tasks.append(mteb.get_task(name))
        except Exception as exc:  # noqa: BLE001
            print(f"warning: could not load task {name}: {exc}", file=sys.stderr)

    if not tasks:
        print("no valid MTEB tasks loaded", file=sys.stderr)
        return 1

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    if args.hf or not args.model.endswith((".tinygpt", ".tinygpt-embed")):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            print("sentence-transformers required for HF models: pip install sentence-transformers", file=sys.stderr)
            return 1
        model = SentenceTransformer(args.model)
    else:
        # Future: tinygpt embed CLI. For now, fail clearly.
        print(
            f"tinygpt embedder path {args.model} — embed CLI not shipped yet; use --hf-model for HF baselines",
            file=sys.stderr,
        )
        return 1

    rows: list[dict] = []
    for task in tasks:
        if args.limit > 0 and hasattr(task, "metadata"):
            try:
                task.metadata.eval_splits = getattr(task.metadata, "eval_splits", ["test"])
            except Exception:  # noqa: BLE001
                pass
        result = mteb.evaluate(
            model,
            tasks=[task],
            encode_kwargs={"batch_size": args.batch_size},
            overwrite_results=True,
            output_folder=str(work / task.metadata.name),
        )
        scores = _extract_scores(result, task.metadata.name)
        for metric, score, n in scores:
            rows.append({
                "task": f"mteb/{task.metadata.name}",
                "subtask": None,
                "metric": metric,
                "score": float(score),
                "n_examples": int(n),
            })

    out = {"model": args.model, "tasks": task_names, "rows": rows}
    Path(args.results_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} score rows → {args.results_json}")
    return 0


def _extract_scores(result, task_name: str) -> list[tuple[str, float, int]]:
    """Pull (metric, score, n) tuples from mteb result objects."""
    out: list[tuple[str, float, int]] = []
    if isinstance(result, list):
        for item in result:
            out.extend(_extract_scores(item, task_name))
        return out
    scores = getattr(result, "scores", None) or {}
    if isinstance(scores, dict):
        for split, metrics in scores.items():
            if not isinstance(metrics, dict):
                continue
            for metric, val in metrics.items():
                if isinstance(val, (int, float)):
                    out.append((str(metric), float(val), 0))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
