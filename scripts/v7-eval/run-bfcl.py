#!/usr/bin/env python3
"""Thin BFCL launcher for Planner v7.

This intentionally does not install BFCL. It verifies that the harness command
is available and then forwards args so the run is explicit/reproducible.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", default="bfcl")
    parser.add_argument("--model", default="tinygpt")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765/v1")
    parser.add_argument("extra", nargs="*")
    args = parser.parse_args()

    exe = shutil.which(args.cmd)
    if not exe:
        raise SystemExit(
            f"BFCL command '{args.cmd}' not found. Install the BFCL harness "
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
