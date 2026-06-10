#!/usr/bin/env python3
"""VLM A/B runner — UI-Venus vs Qwen3-VL (or any N models) on fm-vlm-fixtures.

Implements the eval half of docs/prds/vlm-ab-uivenus-vs-qwen3vl.md without
requiring any Swift port: models are queried over an OpenAI-compatible
endpoint (LM Studio by default). Reuses fake_pace_vlm's fixture parser,
rule baseline, and scorer so the "real model contribution" verdict is
computed the same way as eval_pace_v2.py does for the planner.

Fixtures: pace/evals/fm-vlm-fixtures-v1 (text-only: AX_TREE + OCR_TEXT).
If a fixture carries SCREENSHOT_PATH, the image is attached base64 so the
same runner covers the vision-grounding fixtures when they exist.

Usage:
  # baseline only
  python3 scripts/eval_pace_vlm_ab.py --fixtures-dir ../pace/evals/fm-vlm-fixtures-v1 --skip-model

  # A/B two models loaded in LM Studio
  python3 scripts/eval_pace_vlm_ab.py \
    --fixtures-dir ../pace/evals/fm-vlm-fixtures-v1 \
    --model ui-venus-1.5-2b --model qwen3-vl-2b-instruct \
    --serve-url http://127.0.0.1:1234/v1/chat/completions
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from fake_pace_vlm import parse_fixture, fake_pace_vlm, score_response

SYS_PROMPT = """you are pace's screen-understanding model on a mac. you receive the frontmost app name, the accessibility (AX) tree, OCR text, and sometimes a screenshot. answer the user's question about the screen.

respond with ONLY this JSON, no other text:
{"activity": "<what the user is doing, 2-5 words>", "app": "<frontmost app name>", "elements": ["<label (role)>", ...], "spoken": "<short spoken answer to the user, under 25 words>"}

rules:
- "app" is the app name exactly as given in APP_FRONTMOST.
- "elements" lists the interactive elements you can identify (from AX tree or the screenshot), max 10.
- "spoken" directly answers the user's question. for read-requests, quote the on-screen text. for click-requests, name the element label you would click.
- never mention element IDs or coordinates in "spoken"."""


def _screenshot_path(raw_text: str, fixtures_dir: Path) -> Path | None:
    m = re.search(r"^SCREENSHOT_PATH:\s*(.+)$", raw_text, re.MULTILINE)
    if not m:
        return None
    p = Path(m.group(1).strip())
    return p if p.is_absolute() else fixtures_dir / p


def _format_user(fx: dict) -> str:
    parts = [f"APP_FRONTMOST: {fx['app_frontmost'] or '(unknown)'}"]
    if fx["ax_tree"]:
        parts.append("AX_TREE:")
        for el in fx["ax_tree"]:
            parts.append(f"  [{el['id']}] {el['role']}|{el['pos']}|{el['label']}|{el['text']}")
    else:
        parts.append("AX_TREE: (empty — AX-blind app)")
    if fx["ocr_text"]:
        parts.append("OCR_TEXT:")
        parts.append(fx["ocr_text"])
    parts.append("")
    parts.append(f"user said: {fx['user']}")
    return "\n".join(parts)


def _extract_json(text: str) -> dict:
    """Strip <think> blocks, then take the first balanced {...}."""
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.DOTALL)
    start = text.find("{")
    if start == -1:
        return {}
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    doc = json.loads(text[start:i + 1])
                    return doc if isinstance(doc, dict) else {}
                except Exception:
                    return {}
    return {}


def query_model(url: str, model_id: str, fx: dict, screenshot: Path | None,
                timeout: int = 240) -> tuple[dict, float]:
    content: list | str
    text = _format_user(fx)
    if screenshot and screenshot.exists():
        b64 = base64.b64encode(screenshot.read_bytes()).decode()
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": text},
        ]
    else:
        content = text
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0.0, "max_tokens": 500, "stream": False,
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    t0 = time.time()
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    latency = (time.time() - t0) * 1000
    reply = json.loads(raw)["choices"][0]["message"]["content"]
    parsed = _extract_json(reply)
    # Tolerate near-miss key names and missing fields.
    return {
        "activity": str(parsed.get("activity", "")),
        "app": str(parsed.get("app", "")),
        "elements": parsed.get("elements", []) if isinstance(parsed.get("elements"), list) else [],
        "spoken": str(parsed.get("spoken", reply if not parsed else "")),
    }, latency


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures-dir", type=Path, required=True)
    p.add_argument("--serve-url", default="http://127.0.0.1:1234/v1/chat/completions")
    p.add_argument("--model", action="append", default=[],
                   help="model id as exposed by the endpoint; repeatable for A/B")
    p.add_argument("--skip-model", action="store_true", help="FakePaceVLM baseline only")
    p.add_argument("--json", dest="json_out", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if not args.skip_model and not args.model:
        p.error("pass --model at least once, or --skip-model")

    fx_paths = sorted(args.fixtures_dir.glob("*.txt"))
    models = [] if args.skip_model else args.model
    results: dict = {"fixtures": [], "totals": {}, "latency_ms": {}}
    passes = {"FakePace": 0, **{m: 0 for m in models}}
    latencies: dict[str, list[float]] = {m: [] for m in models}

    name_w = max((len(fp.stem) for fp in fx_paths), default=20)
    cols = ["FakePace"] + models
    print(f"=== vlm A/B on {len(fx_paths)} fixtures in {args.fixtures_dir.name} ===\n")
    print(f"{'fixture':<{name_w}} | " + " | ".join(f"{c[:18]:<18}" for c in cols))
    print("-" * (name_w + 3 + 21 * len(cols)))

    for fp in fx_paths:
        raw = fp.read_text()
        fx = parse_fixture(raw)
        shot = _screenshot_path(raw, args.fixtures_dir)
        row = {"fixture": fp.stem, "results": {}}

        fake = fake_pace_vlm(fx)
        ok, fails = score_response(fake, fx["expects"], fx)
        row["results"]["FakePace"] = {"pass": ok, "fails": fails}
        passes["FakePace"] += ok

        for m in models:
            try:
                resp, ms = query_model(args.serve_url, m, fx, shot)
                ok_m, fails_m = score_response(resp, fx["expects"], fx)
                latencies[m].append(ms)
            except Exception as e:  # endpoint/model error = fail, keep going
                resp, ok_m, fails_m, ms = {}, False, [f"query error: {e}"], 0.0
            row["results"][m] = {"pass": ok_m, "fails": fails_m, "latency_ms": ms,
                                 "response": resp}
            passes[m] += ok_m
            if args.verbose and fails_m:
                print(f"    {m}: {fails_m}  resp={json.dumps(resp)[:160]}")

        results["fixtures"].append(row)
        print(f"{fp.stem:<{name_w}} | " + " | ".join(
            f"{'PASS' if row['results'][c]['pass'] else 'FAIL':<18}" for c in cols))

    n = len(fx_paths)
    print()
    for c in cols:
        pct = 100.0 * passes[c] / n if n else 0.0
        lat = ""
        if c in latencies and latencies[c]:
            lat = f"  (median {sorted(latencies[c])[len(latencies[c]) // 2]:.0f}ms/call)"
        print(f"{c:<24} {passes[c]}/{n}  ({pct:.1f}%){lat}")
        results["totals"][c] = {"passed": passes[c], "n": n, "pct": pct}
        if c in latencies and latencies[c]:
            results["latency_ms"][c] = sorted(latencies[c])[len(latencies[c]) // 2]

    # PRD decision tree (≥5pp between exactly two models).
    if len(models) == 2:
        a, b = models
        delta = results["totals"][a]["pct"] - results["totals"][b]["pct"]
        print()
        if abs(delta) < 5.0:
            print(f"Verdict: TIE within 5pp ({delta:+.1f}pp) — per PRD, pick the Qwen3-family model.")
        else:
            winner = a if delta > 0 else b
            print(f"Verdict: {winner} wins by {abs(delta):.1f}pp.")
        results["delta_pp"] = delta

    if args.json_out:
        print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
