#!/usr/bin/env python3
"""Thin τ-bench launcher for Planner v7."""

from __future__ import annotations

import argparse
import shutil
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", default="tau-bench")
    parser.add_argument("--model", default="tinygpt")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765/v1")
    parser.add_argument("extra", nargs="*")
    args = parser.parse_args()

    exe = shutil.which(args.cmd)
    if not exe:
        raise SystemExit(
            f"tau-bench command '{args.cmd}' not found. Install tau-bench "
            "in your Python environment, then rerun this wrapper."
        )
    cmd = [
        exe,
        "--model", args.model,
        "--base-url", args.base_url,
        *args.extra,
    ]
    print("+", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
