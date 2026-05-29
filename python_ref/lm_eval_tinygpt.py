#!/usr/bin/env python3
"""
lm_eval_tinygpt.py — drive lm-evaluation-harness against a tinygpt checkpoint.

Wraps the standard `lm-eval-harness` so any tinygpt model (from-scratch
.tinygpt file OR an HF model directory loaded through our HF path) can be
scored against HellaSwag, ARC-Easy, GSM8K, IFEval, MMLU-Pro, GPQA-Diamond,
MATH-500, HumanEval, … — anything the harness has a task definition for.

Pipeline:
  1. Spawn `tinygpt serve <model> --port <free port>` as a subprocess.
  2. Poll `GET /v1/models` until it answers 200 (server is ready).
  3. Run `lm-eval --model local-chat-completions
                  --model_args base_url=http://127.0.0.1:<port>/v1/chat/completions,model=tinygpt,…
                  --tasks <tasks> --output_path <out>`.
  4. SIGTERM the server on exit; print scores; write JSON results.

Pre-reqs (one-time):
  python -m venv .venv && source .venv/bin/activate
  pip install lm-eval==0.4.10
  # ^ pin 0.4.10 to dodge the stop-sequence bug in 0.4.11; alternatively
  # use whatever is latest at the time of running and check release notes.
  # See docs/lm_eval_integration.md "Known issues" for the bug spec.

Build prerequisite:
  The `tinygpt serve` subcommand requires the `case "serve":` to be wired
  in Sources/TinyGPT/TinyGPT.swift (see the TODO(serve-merge) comment in
  that file). Until that's merged, this script will fail with "unknown
  subcommand: serve". Workaround: pass --tinygpt-bin to a wrapper script
  or run the server manually first (see --skip-spawn flag).

Usage:
  python lm_eval_tinygpt.py /tmp/flagship-huge.tinygpt \\
      --tasks hellaswag,arc_easy --output-path bench/results/run1.json

  # Reuse an already-running server (skip the spawn):
  python lm_eval_tinygpt.py --skip-spawn --base-url http://127.0.0.1:9000/v1/chat/completions \\
      --tasks gsm8k
"""

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def pick_free_port() -> int:
    """Ask the kernel for a free port by binding to 0 and reading back."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_for_ready(base_url: str, timeout_s: float = 60.0) -> None:
    """Poll /v1/models until it answers 200 or we time out."""
    models_url = base_url.replace("/v1/chat/completions", "/v1/models").replace(
        "/v1/completions", "/v1/models"
    )
    deadline = time.monotonic() + timeout_s
    last_err = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(models_url, timeout=1.0) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionRefusedError, socket.timeout) as e:
            last_err = e
        time.sleep(0.5)
    raise RuntimeError(
        f"tinygpt serve did not become ready within {timeout_s}s at {models_url} "
        f"(last error: {last_err})"
    )


def _tinygpt_has_serve(binary: str) -> bool:
    """Quick probe: does this binary know the `serve` subcommand?

    Until `case "serve":` is wired into Sources/TinyGPT/TinyGPT.swift,
    the main `tinygpt` binary doesn't dispatch serve and we have to fall
    back to `tinygpt-serve-smoke`. This probe is what makes that
    fallback transparent.
    """
    try:
        result = subprocess.run([binary, "serve", "--help"],
                                 capture_output=True, timeout=5)
        # serve --help exits 2 (usage), but only if the case is wired.
        # When unwired, the main dispatch prints "unknown subcommand:
        # serve" to stderr — that's the signal we want to detect.
        stderr = result.stderr.decode("utf-8", errors="ignore")
        return "unknown subcommand" not in stderr
    except (subprocess.TimeoutExpired, OSError):
        return False


def find_tinygpt_binary(explicit: str | None) -> str:
    """Locate the `tinygpt` executable. Order:
       1. --tinygpt-bin flag if passed
       2. $TINYGPT_BIN env var
       3. ./native-mac/build/Release/tinygpt (and a few common derived-data paths)
       4. shutil.which("tinygpt")

    If the located binary doesn't have the `serve` subcommand wired,
    we swap to `tinygpt-serve-smoke` (the stand-in target — see
    docs/lm_eval_integration.md for why it exists).
    """
    binary: str | None = None
    if explicit:
        binary = explicit
    else:
        env = os.environ.get("TINYGPT_BIN")
        if env and Path(env).exists():
            binary = env
        else:
            root = Path(__file__).resolve().parent.parent
            candidates = [
                root / "native-mac" / "build" / "Release" / "tinygpt",
                root / "build" / "Release" / "tinygpt",
                Path("/tmp/tinygpt-smoke/Build/Products/Release/tinygpt"),
            ]
            for c in candidates:
                if c.exists():
                    binary = str(c)
                    break
            if binary is None:
                which = shutil.which("tinygpt")
                if which:
                    binary = which
    if binary is None:
        raise SystemExit(
            "couldn't find the tinygpt binary. Either build it first "
            "(see README) or pass --tinygpt-bin /path/to/tinygpt."
        )

    # If serve isn't wired into this binary, look for the smoke target alongside.
    if os.path.basename(binary) != "tinygpt-serve-smoke" and not _tinygpt_has_serve(binary):
        smoke = str(Path(binary).parent / "tinygpt-serve-smoke")
        if Path(smoke).exists():
            print(f"[lm_eval_tinygpt] '{binary}' lacks the serve subcommand — "
                  f"falling back to {smoke}", flush=True)
            binary = smoke
        else:
            raise SystemExit(
                f"'{binary} serve' isn't callable and {smoke} doesn't exist. "
                "Either wire case \"serve\": into Sources/TinyGPT/TinyGPT.swift, "
                "OR build the smoke target:\n"
                "  xcodebuild -scheme tinygpt-serve-smoke -derivedDataPath /tmp/tinygpt-smoke "
                "-configuration Release build"
            )
    return binary


def main():
    ap = argparse.ArgumentParser(
        description="Run lm-evaluation-harness against a tinygpt model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("model_path", nargs="?", help="path to .tinygpt file or HF model dir")
    ap.add_argument("--tasks", default="hellaswag,arc_easy",
                     help="comma-separated lm-eval tasks (default: hellaswag,arc_easy)")
    ap.add_argument("--output-path", default="bench/results",
                     help="directory to write lm-eval JSON results (default: bench/results)")
    ap.add_argument("--num-fewshot", type=int, default=None,
                     help="few-shot count override (task-dependent default otherwise)")
    ap.add_argument("--limit", type=int, default=None,
                     help="limit per task — useful for smoke runs (e.g. --limit 10)")
    ap.add_argument("--batch-size", default="1",
                     help="lm-eval batch size (default: 1; tinygpt serves one at a time)")
    ap.add_argument("--port", type=int, default=0,
                     help="serve port (default: auto-pick a free port)")
    ap.add_argument("--max-context", type=int, default=None,
                     help="cap context length below the model's native limit")
    ap.add_argument("--tinygpt-bin", default=None,
                     help="path to the `tinygpt` binary (auto-detect by default)")
    ap.add_argument("--skip-spawn", action="store_true",
                     help="don't spawn `tinygpt serve` — assume one is already running")
    ap.add_argument("--base-url", default=None,
                     help="explicit OpenAI base URL (default: auto from --port)")
    ap.add_argument("--lm-eval-extra", default="",
                     help="extra args appended verbatim to the lm-eval invocation")
    args = ap.parse_args()

    if not args.skip_spawn and not args.model_path:
        ap.error("model_path is required unless --skip-spawn is set")

    # Resolve port + base_url.
    port = args.port or pick_free_port()
    base_url = args.base_url or f"http://127.0.0.1:{port}/v1/chat/completions"

    server_proc = None
    server_log = None
    try:
        if not args.skip_spawn:
            binary = find_tinygpt_binary(args.tinygpt_bin)
            # The stand-in `tinygpt-serve-smoke` binary takes its first arg
            # directly as the model path (no "serve" keyword); the main
            # `tinygpt` CLI dispatches via `tinygpt serve <model>`. Detect
            # by basename so the wrapper works against either binary
            # transparently.
            is_smoke = os.path.basename(binary) == "tinygpt-serve-smoke"
            serve_cmd: list[str] = [binary]
            if not is_smoke:
                serve_cmd.append("serve")
            serve_cmd += [args.model_path, "--port", str(port), "--host", "127.0.0.1"]
            if args.max_context is not None:
                serve_cmd += ["--max-context", str(args.max_context)]
            print(f"[lm_eval_tinygpt] starting: {' '.join(serve_cmd)}", flush=True)
            # Capture server logs for post-mortem if it dies.
            server_log = open("/tmp/tinygpt-serve.log", "w")
            server_proc = subprocess.Popen(
                serve_cmd, stdout=server_log, stderr=subprocess.STDOUT
            )
            wait_for_ready(base_url, timeout_s=120.0)
            print(f"[lm_eval_tinygpt] server ready at {base_url}", flush=True)
        else:
            print(f"[lm_eval_tinygpt] reusing server at {base_url}", flush=True)
            wait_for_ready(base_url, timeout_s=10.0)

        # Build the lm-eval command.
        out_dir = Path(args.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        # The `local-chat-completions` model_args format is documented in
        # `lm-evaluation-harness/lm_eval/models/openai_completions.py`.
        # We pass tokenizer_backend=None so it doesn't try to install
        # tiktoken/transformers tokenizers we don't need.
        model_args = (
            f"base_url={base_url},"
            f"model=tinygpt,"
            f"tokenizer_backend=None,"
            f"tokenized_requests=False,"
            f"num_concurrent=1"
        )
        lm_eval_cmd = [
            "lm-eval",
            "--model", "local-chat-completions",
            "--model_args", model_args,
            "--tasks", args.tasks,
            "--batch_size", args.batch_size,
            "--output_path", str(out_dir),
        ]
        if args.num_fewshot is not None:
            lm_eval_cmd += ["--num_fewshot", str(args.num_fewshot)]
        if args.limit is not None:
            lm_eval_cmd += ["--limit", str(args.limit)]
        if args.lm_eval_extra:
            lm_eval_cmd += args.lm_eval_extra.split()

        print(f"[lm_eval_tinygpt] running: {' '.join(lm_eval_cmd)}", flush=True)
        rc = subprocess.call(lm_eval_cmd)
        if rc != 0:
            print(f"[lm_eval_tinygpt] lm-eval exited with code {rc}", file=sys.stderr)
            sys.exit(rc)

        # lm-eval writes one JSON per task plus a summary file in the
        # output dir; surface the headline numbers.
        summary_path = sorted(out_dir.glob("results_*.json"))[-1] if list(out_dir.glob("results_*.json")) else None
        if summary_path:
            with summary_path.open() as f:
                data = json.load(f)
            print("\n=== lm-eval results ===")
            print(f"file: {summary_path}")
            results = data.get("results", {})
            for task_name, metrics in results.items():
                # Print every metric the harness emitted for the task.
                print(f"  {task_name}:")
                for k, v in metrics.items():
                    if isinstance(v, float):
                        print(f"    {k}: {v:.4f}")
                    else:
                        print(f"    {k}: {v}")
        else:
            print("[lm_eval_tinygpt] no results_*.json found in output dir")

    finally:
        if server_proc is not None:
            print("[lm_eval_tinygpt] stopping tinygpt serve...", flush=True)
            try:
                server_proc.send_signal(signal.SIGTERM)
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        if server_log is not None:
            server_log.close()


if __name__ == "__main__":
    main()
