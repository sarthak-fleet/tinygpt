#!/usr/bin/env python3
"""eval_pace_v2 — side-by-side eval of FakePace (rule-based baseline) vs
a serve endpoint (trained model) against fm-fixtures-v2.

The whole point: produce a single number that says "this LoRA adds
N percentage points of capability above the rule-based ceiling."

Without this delta, every LoRA comparison is theatre (proven by
docs/learn/eval-methodology-2026-06-08.md).

Usage:

  # FakePace baseline only (no serve needed):
  python3 scripts/eval_pace_v2.py --skip-model

  # Both, with serve at default port:
  tinygpt serve <hf-dir> --lora <lora>.lora --grammar grammars/pace-fm-label-response.schema.json --port 8765
  python3 scripts/eval_pace_v2.py --serve-url http://127.0.0.1:8765/v1/chat/completions

  # Custom fixtures dir (e.g. v1 to see the false-positive comparison):
  python3 scripts/eval_pace_v2.py --fixtures-dir /Users/sarthak/Desktop/fleet/clickyLocal/evals/fm-fixtures
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

# Reuse rule-based responder + scorer from fake_pace.py
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fake_pace import fake_pace, parse_fixture, score_response


DEFAULT_FIXTURES = Path("/Users/sarthak/Desktop/fleet/clickyLocal/evals/fm-fixtures-v2")
DEFAULT_SYSP = Path("/Users/sarthak/Desktop/fleet/tinygpt/grammars/pace-system-prompt-v6-label.txt")
DEFAULT_SCHEMA = Path("/Users/sarthak/Desktop/fleet/tinygpt/grammars/pace-fm-label-response.schema.json")


def _format_user(fx: dict) -> str:
    """Mirror pace-eval-v6.py's user-message construction so the model
    sees the exact same prompt the production runtime feeds it."""
    parts: list[str] = []
    if fx["elements"]:
        parts.append("on-screen elements:")
        for el in fx["elements"]:
            parts.append(f"[{el['id']}] {el['role']}|{el['pos']}|{el['label']}|{el['text']}")
        parts.append("")
    parts.append(f"user said: {fx['user']}")
    return "\n".join(parts)


def query_serve(url: str, model_id: str, sys_prompt: str, schema: dict,
                  fx: dict, timeout: int = 180) -> str:
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": _format_user(fx)},
        ],
        "temperature": 0.0, "max_tokens": 250, "stream": False,
    }
    if not fx["free_text"]:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "PaceFMLabelResponse", "strict": True, "schema": schema},
        }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    r = urllib.request.urlopen(req, timeout=timeout).read()
    return json.loads(r)["choices"][0]["message"]["content"]


def run(fixtures_dir: Path, serve_url: str | None, model_id: str,
        sys_prompt_path: Path, schema_path: Path,
        verbose: bool = False) -> dict:
    sysp = sys_prompt_path.read_text().strip()
    schema = json.loads(schema_path.read_text())
    fxs = sorted(fixtures_dir.glob("*.txt"))
    print(f"=== eval_pace_v2 against {len(fxs)} fixtures in {fixtures_dir.name} ===")
    print()
    if serve_url:
        print(f"Serve URL: {serve_url}")
        print(f"Model ID:  {model_id}")
        print()

    # Header
    if serve_url:
        print(f"{'fixture':<32} | {'FakePace':<8} | {'Model':<8} | delta")
        print("-" * 70)
    else:
        print(f"{'fixture':<32} | {'FakePace':<8}")
        print("-" * 45)

    rows = []
    for fx_path in fxs:
        fx = parse_fixture(fx_path.read_text())
        # FakePace
        fp_resp = fake_pace(fx["user"], fx["elements"], free_text_mode=fx["free_text"])
        fp_passed, fp_reasons = score_response(
            fp_resp, fx["expects"], fx["elements"], fx["free_text"])

        # Model (optional)
        mdl_passed: bool | None = None
        mdl_reasons: list[str] = []
        mdl_resp = ""
        if serve_url:
            try:
                mdl_resp = query_serve(serve_url, model_id, sysp, schema, fx)
                mdl_passed, mdl_reasons = score_response(
                    mdl_resp, fx["expects"], fx["elements"], fx["free_text"])
            except Exception as e:
                mdl_reasons = [f"serve error: {type(e).__name__}: {str(e)[:120]}"]
                mdl_passed = False

        row = {
            "fixture": fx_path.stem,
            "fakepace_passed": fp_passed, "fakepace_response": fp_resp,
            "fakepace_reasons": fp_reasons,
            "model_passed": mdl_passed, "model_response": mdl_resp,
            "model_reasons": mdl_reasons,
        }
        rows.append(row)
        fp_mark = "PASS" if fp_passed else "FAIL"
        if serve_url:
            mdl_mark = "PASS" if mdl_passed else ("FAIL" if mdl_passed is not None else "skip")
            delta = "  +1" if (mdl_passed and not fp_passed) else \
                       ("  -1" if (fp_passed and not mdl_passed) else \
                       ("   =" if (mdl_passed == fp_passed) else "    "))
            print(f"{fx_path.stem:<32} | {fp_mark:<8} | {mdl_mark:<8} | {delta}")
        else:
            print(f"{fx_path.stem:<32} | {fp_mark:<8}")
        if verbose:
            if not fp_passed:
                print(f"    fakepace: {fp_resp[:140]}")
                for r in fp_reasons: print(f"      ✗ {r}")
            if serve_url and not mdl_passed:
                print(f"    model:    {mdl_resp[:140]}")
                for r in mdl_reasons: print(f"      ✗ {r}")

    print()
    n = len(rows)
    fp_pass = sum(1 for r in rows if r["fakepace_passed"])
    print(f"FakePace baseline:  {fp_pass}/{n}  ({100*fp_pass/n:.1f}%)")
    if serve_url:
        mdl_pass = sum(1 for r in rows if r["model_passed"])
        only_fp = sum(1 for r in rows if r["fakepace_passed"] and not r["model_passed"])
        only_mdl = sum(1 for r in rows if r["model_passed"] and not r["fakepace_passed"])
        both = sum(1 for r in rows if r["fakepace_passed"] and r["model_passed"])
        neither = sum(1 for r in rows if not r["fakepace_passed"] and not r["model_passed"])
        print(f"Model score:        {mdl_pass}/{n}  ({100*mdl_pass/n:.1f}%)")
        print()
        delta_pp = 100 * (mdl_pass - fp_pass) / n
        print(f"Δ Model − FakePace: {mdl_pass - fp_pass:+d} fixtures  ({delta_pp:+.1f} pp)")
        print()
        print(f"  Both passed:    {both}")
        print(f"  Neither passed: {neither}")
        print(f"  Only FakePace:  {only_fp}  (model regressed below rules)")
        print(f"  Only Model:     {only_mdl}  (real model contribution)")
        print()
        # Final verdict
        if delta_pp >= 30:
            verdict = "REAL MODEL CONTRIBUTION — ship-worthy delta"
        elif delta_pp >= 10:
            verdict = "marginal model contribution — investigate"
        elif delta_pp >= 0:
            verdict = "model ≈ rules — not worth shipping"
        else:
            verdict = "MODEL REGRESSES BELOW RULES — do not ship"
        print(f"Verdict: {verdict}")

    return {
        "n": n,
        "fakepace_pass": fp_pass,
        "model_pass": (sum(1 for r in rows if r["model_passed"]) if serve_url else None),
        "rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures-dir", type=Path, default=DEFAULT_FIXTURES)
    p.add_argument("--serve-url",
                     default="http://127.0.0.1:8765/v1/chat/completions",
                     help="OpenAI-compat endpoint (default tinygpt-serve port). "
                          "Skipped if --skip-model is set.")
    p.add_argument("--model", default="tinygpt", help="model ID for the request body")
    p.add_argument("--sys-prompt", type=Path, default=DEFAULT_SYSP)
    p.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    p.add_argument("--skip-model", action="store_true",
                     help="only run FakePace, skip the model query")
    p.add_argument("-v", "--verbose", action="store_true",
                     help="print response + failure reasons for each fixture")
    p.add_argument("--json", action="store_true",
                     help="emit results JSON to stdout (replaces table)")
    args = p.parse_args()

    serve_url = None if args.skip_model else args.serve_url
    result = run(args.fixtures_dir, serve_url, args.model, args.sys_prompt,
                   args.schema, verbose=args.verbose)
    if args.json:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
