#!/usr/bin/env python3
"""tinygpt eval-bfcl — Berkeley Function Calling Leaderboard runner (#231).

Runs a model endpoint against BFCL v3 test files and reports per-category
pass-rate. Calls any OpenAI-compatible endpoint (tinygpt serve, LM Studio,
Ollama with OpenAI shim). Score = AST-match (function name + arg values
contained in ground-truth possible-value lists per arg).

For v10 validation: model must emit our v10 schema
    {"spokenText": "...", "intent": "action",
     "payload": {"name": "fn_name", "args": {...}}}
The runner extracts payload.name + payload.args and scores against BFCL ground truth.

Usage:
    python3 scripts/eval_bfcl.py \\
      --serve-url http://127.0.0.1:8765/v1/chat/completions \\
      --bfcl-dir ~/.cache/tinygpt/datasets/bfcl \\
      --categories simple,multiple,irrelevance \\
      --max-per-category 50 \\
      --out ~/.cache/tinygpt/evals/bfcl-v10-<timestamp>.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_BFCL_DIR = Path.home() / ".cache/tinygpt/datasets/bfcl"

# Files we score for v10. Multi-turn deliberately excluded (single-shot doctrine).
DEFAULT_CATEGORIES = [
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
    "irrelevance",
    "live_irrelevance",
]


# -------------------- Loaders --------------------


def _read_jsonl(p: Path) -> list[dict]:
    out: list[dict] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_category(bfcl_dir: Path, category: str) -> tuple[list[dict], dict[str, dict]]:
    """Returns (questions, answers_by_id). Irrelevance has no ground truth."""
    q_path = bfcl_dir / f"BFCL_v3_{category}.json"
    if not q_path.exists():
        raise FileNotFoundError(q_path)
    questions = _read_jsonl(q_path)

    a_path = bfcl_dir / "possible_answer" / f"BFCL_v3_{category}.json"
    answers_by_id: dict[str, dict] = {}
    if a_path.exists():
        for a in _read_jsonl(a_path):
            answers_by_id[a["id"]] = a

    return questions, answers_by_id


# -------------------- Prompting --------------------


SYSTEM_PROMPT_TEMPLATE = """you are a function-call router. you have access to the following functions:

{function_defs}

decide whether to call one (or several) of these functions to answer the user.

ALWAYS respond with valid JSON matching this schema:
{{"spokenText": "...", "intent": "action"|"answer", "payload": {{...}}}}

if a single function call answers the user: intent="action", payload={{"name": "fn_name", "args": {{...}}}}.
if NO function fits the user's request (irrelevance): intent="answer", payload={{"text": "i can't answer that with the tools available"}}.
if multiple function calls are needed in parallel: intent="action", payload={{"calls": [{{"name":...,"args":...}}, ...]}}.

spokenText is what would be read aloud as confirmation. keep it short."""


def render_function_defs(functions: list[dict]) -> str:
    """Render function defs in a compact YAML-ish form for the system prompt."""
    out: list[str] = []
    for fn in functions:
        out.append(f"- {fn['name']}({_args_summary(fn['parameters'])})")
        out.append(f"  description: {fn['description']}")
        out.append("  params:")
        props = fn["parameters"].get("properties", {})
        required = set(fn["parameters"].get("required", []))
        for name, spec in props.items():
            req = "required" if name in required else "optional"
            t = spec.get("type", "any")
            d = spec.get("description", "")
            out.append(f"    {name} ({t}, {req}): {d}")
    return "\n".join(out)


def _args_summary(parameters: dict) -> str:
    return ", ".join(parameters.get("properties", {}).keys())


def build_messages(question_turns: list[list[dict]], functions: list[dict]) -> list[dict]:
    """BFCL `question` is list-of-list-of-turns. We flatten to a single message list."""
    sys_msg = {
        "role": "system",
        "content": SYSTEM_PROMPT_TEMPLATE.format(function_defs=render_function_defs(functions)),
    }
    msgs = [sys_msg]
    for turn_group in question_turns:
        for turn in turn_group:
            msgs.append({"role": turn["role"], "content": turn["content"]})
    return msgs


# -------------------- Calling the model --------------------


def call_model(serve_url: str, model_name: str, messages: list[dict],
                max_tokens: int = 256, temperature: float = 0.0) -> tuple[str, float]:
    """Returns (content, elapsed_ms)."""
    body = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(serve_url, data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"},
                                   method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.loads(r.read())
    except Exception as e:
        return f"__ERROR__: {e}", (time.time() - t0) * 1000
    t = (time.time() - t0) * 1000
    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content, t


# -------------------- Parsing model output --------------------


def parse_v10(content: str) -> tuple[str, dict | None]:
    """Extract (intent, payload) from model output. Tolerate sloppy JSON."""
    # Strip code fences if present
    s = content.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        # Try to find first {...} block
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            return "__parse_error__", None
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            return "__parse_error__", None
    intent = d.get("intent", "")
    payload = d.get("payload", None)
    return intent, payload


def extract_calls(intent: str, payload: dict | None) -> list[dict]:
    """Return a list of {name, args} calls. Empty list if answer/irrelevance."""
    if intent != "action" or not payload:
        return []
    if "calls" in payload and isinstance(payload["calls"], list):
        return [c for c in payload["calls"] if isinstance(c, dict)]
    if "name" in payload:
        return [{"name": payload["name"], "args": payload.get("args", {})}]
    return []


# -------------------- Scoring --------------------


def score_call(predicted: dict, ground_truth: dict) -> bool:
    """A single predicted {name, args} vs a single ground truth {fn_name: {arg: [vals]}}.

    Returns True if function name matches AND every required arg's value is
    in the possible-value list. Optional args are allowed to be omitted.
    """
    if not predicted:
        return False
    pred_name = predicted.get("name", "")
    if pred_name not in ground_truth:
        return False
    gt_args = ground_truth[pred_name]
    pred_args = predicted.get("args", {})
    if not isinstance(pred_args, dict):
        return False
    for arg_name, possible_values in gt_args.items():
        if not isinstance(possible_values, list):
            return False
        # If arg present in prediction, it must match one of the possibles
        if arg_name in pred_args:
            if pred_args[arg_name] not in possible_values:
                # Allow string conversion match (BFCL uses strings widely)
                if str(pred_args[arg_name]) not in [str(v) for v in possible_values]:
                    return False
        else:
            # Missing arg — allowed only if "" is in possible_values (i.e. optional)
            if "" not in possible_values and None not in possible_values:
                return False
    return True


def score_question(predicted_calls: list[dict], ground_truth: list[dict]) -> bool:
    """All predicted calls and ground-truth calls must match, in any order."""
    if len(predicted_calls) != len(ground_truth):
        return False
    # Try to find a matching ground_truth call for each predicted call
    used = [False] * len(ground_truth)
    for pred in predicted_calls:
        matched = False
        for i, gt in enumerate(ground_truth):
            if used[i]:
                continue
            if score_call(pred, gt):
                used[i] = True
                matched = True
                break
        if not matched:
            return False
    return all(used)


# -------------------- Runner --------------------


def run_category(bfcl_dir: Path, category: str, serve_url: str, model_name: str,
                  max_per_category: int | None = None,
                  verbose: bool = False) -> dict:
    questions, answers = load_category(bfcl_dir, category)
    if max_per_category:
        questions = questions[:max_per_category]
    is_irrelevance = "irrelevance" in category

    results: list[dict] = []
    n_pass = 0
    n_fail = 0
    n_parse_error = 0
    latencies_ms: list[float] = []

    for q in questions:
        qid = q["id"]
        msgs = build_messages(q["question"], q["function"])
        content, elapsed_ms = call_model(serve_url, model_name, msgs)
        latencies_ms.append(elapsed_ms)
        intent, payload = parse_v10(content)
        pred_calls = extract_calls(intent, payload)
        if intent == "__parse_error__":
            n_parse_error += 1
            passed = False
        elif is_irrelevance:
            # Pass if model correctly refrained from calling a function
            passed = (intent != "action" or not pred_calls)
        else:
            gt = answers.get(qid, {}).get("ground_truth", [])
            passed = score_question(pred_calls, gt)
        if passed:
            n_pass += 1
        else:
            n_fail += 1
        if verbose and not passed:
            print(f"  FAIL {qid}: pred={pred_calls!r} gt={answers.get(qid,{}).get('ground_truth')}")
        results.append({
            "id": qid,
            "pass": passed,
            "intent": intent,
            "pred_calls": pred_calls,
            "latency_ms": elapsed_ms,
        })

    n = len(questions)
    return {
        "category": category,
        "n": n,
        "pass": n_pass,
        "fail": n_fail,
        "parse_errors": n_parse_error,
        "pass_rate": n_pass / n if n else 0.0,
        "median_latency_ms": sorted(latencies_ms)[len(latencies_ms)//2] if latencies_ms else 0,
        "details": results,
    }


# -------------------- CLI --------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--serve-url", default="http://127.0.0.1:8765/v1/chat/completions")
    p.add_argument("--model", default="tinygpt")
    p.add_argument("--bfcl-dir", type=Path, default=DEFAULT_BFCL_DIR)
    p.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES),
                     help="comma-separated; available: simple, multiple, parallel, parallel_multiple, live_simple, live_multiple, live_parallel, irrelevance, live_irrelevance")
    p.add_argument("--max-per-category", type=int, default=None,
                     help="cap N per category for quick runs")
    p.add_argument("--out", type=Path, default=None,
                     help="JSON output path; default ~/.cache/tinygpt/evals/bfcl-<timestamp>.json")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    overall = {
        "serve_url": args.serve_url,
        "model": args.model,
        "categories": {},
        "summary": {},
        "timestamp": int(time.time()),
    }
    total_pass = 0
    total_n = 0
    for cat in cats:
        print(f"--- {cat} ---")
        try:
            r = run_category(args.bfcl_dir, cat, args.serve_url, args.model,
                              max_per_category=args.max_per_category,
                              verbose=args.verbose)
        except FileNotFoundError as e:
            # A missing category is a config error, not a soft skip — silently
            # dropping one would let a ship-gate run (v11_pipeline.sh) report
            # success while omitting a dimension.
            print(f"  FATAL — {e}", file=sys.stderr)
            sys.exit(2)
        overall["categories"][cat] = r
        print(f"  {r['pass']}/{r['n']} ({r['pass_rate']*100:.1f}%) · parse_err {r['parse_errors']} · p50 lat {r['median_latency_ms']:.0f}ms")
        total_pass += r["pass"]
        total_n += r["n"]
    overall["summary"] = {
        "total_pass": total_pass,
        "total_n": total_n,
        "overall_pass_rate": total_pass / total_n if total_n else 0.0,
    }
    print()
    print(f"OVERALL: {total_pass}/{total_n} ({overall['summary']['overall_pass_rate']*100:.1f}%)")

    out = args.out or Path.home() / ".cache/tinygpt/evals" / f"bfcl-{overall['timestamp']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(overall, indent=2, default=str))
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
