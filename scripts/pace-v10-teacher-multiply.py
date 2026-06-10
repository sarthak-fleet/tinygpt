#!/usr/bin/env python3
"""Multiply Pace v10 hand-crafted seeds via Qwen3-14B teacher.

For each seed (a {prompt, response} pair in v10 schema), ask Qwen3-14B to
produce N variations: same intent + similar payload + paraphrased prompt.
Validates every output against the v10 schema before keeping.

Assumes LM Studio (or another OpenAI-compatible local server) is up at
http://127.0.0.1:1234/v1 with Qwen3-14B (or another 14B+ teacher) loaded.

DOES NOT call any cloud service — local-only per Pace doctrine.

Run AFTER v9 training finishes (memory pressure: 14B + 0.6B-DoRA = OOM).

Usage:
    python3 scripts/pace-v10-teacher-multiply.py \\
      --seeds-in ~/.cache/tinygpt/datasets/pace-v10-sft.jsonl \\
      --out ~/.cache/tinygpt/datasets/pace-v10-multiplied.jsonl \\
      --variations-per-seed 10 \\
      --teacher-url http://127.0.0.1:1234/v1/chat/completions \\
      --teacher-model qwen3-14b
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path


DEFAULT_SEEDS = Path.home() / ".cache/tinygpt/datasets/pace-v10-sft.jsonl"
DEFAULT_OUT = Path.home() / ".cache/tinygpt/datasets/pace-v10-multiplied.jsonl"
DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "grammars/v10-actions/registry.json"


# -------------------- v10 schema validation --------------------

VALID_INTENTS = {"action", "answer", "dictate", "edit"}


def load_registry(p: Path) -> dict[str, dict]:
    reg = json.loads(p.read_text())
    return {a["name"]: a["args"] for a in reg["actions"]}


def validate_v10(resp_str: str, action_registry: dict[str, dict]) -> tuple[bool, str]:
    """Strict v10 schema validation. Returns (ok, reason)."""
    try:
        d = json.loads(resp_str)
    except json.JSONDecodeError as e:
        return False, f"not json: {e}"
    if not isinstance(d, dict):
        return False, "not object"
    for k in ("spokenText", "intent", "payload"):
        if k not in d:
            return False, f"missing {k}"
    intent = d["intent"]
    if intent not in VALID_INTENTS:
        return False, f"bad intent: {intent}"
    payload = d["payload"]
    if not isinstance(payload, dict):
        return False, "payload not object"
    if intent == "action":
        if "name" not in payload:
            return False, "action payload missing name"
        if payload["name"] not in action_registry:
            return False, f"unknown action: {payload['name']}"
        # We could deeper-validate args shape against registry schema here;
        # skipped for now — runtime grammar will enforce at decode time.
    elif intent == "answer":
        if "text" not in payload:
            return False, "answer payload missing text"
    elif intent == "dictate":
        if "text" not in payload:
            return False, "dictate payload missing text"
    elif intent == "edit":
        for k in ("reference", "transform"):
            if k not in payload:
                return False, f"edit payload missing {k}"
    return True, ""


# -------------------- Teacher prompting --------------------


MULTIPLY_PROMPT_TEMPLATE = """you are a data augmentation helper for pace, a voice-first mac assistant. given one seed example below, produce {n} new variations.

each variation must:
- KEEP the same intent and the same action.name (if intent=action)
- PARAPHRASE the user-said voice command to something different in wording but same in meaning
- PARAPHRASE the spokenText so it sounds natural and matches the new prompt
- TWEAK the args/payload values to be plausible variations (e.g. different recipient names, different times, different element labels matching the on-screen list)
- KEEP the same on-screen-elements block VERBATIM (so the user can paraphrase the request against a fixed screen context)

output EXACTLY {n} variations as a JSON array. each element is an object with keys 'instruction' and 'response'. response is the v10 schema as a JSON STRING (escaped).

example output shape:
[
  {{"instruction": "on-screen elements:\\n[0] ...\\n\\nuser said: open mail please", "response": "{{\\"spokenText\\":\\"opening mail\\",\\"intent\\":\\"action\\",\\"payload\\":{{\\"name\\":\\"AX.press\\",\\"args\\":{{\\"target\\":\\"Mail\\"}}}}}}"}},
  ...
]

DO NOT add commentary. JSON only.

--- SEED ---
instruction: {instruction}
response: {response}
--- END SEED ---

now emit the {n} variations:"""


def call_teacher(teacher_url: str, model: str, prompt: str,
                  max_tokens: int = 2000, temperature: float = 0.7,
                  disable_thinking: bool = True) -> str:
    # /no_think disables Qwen3's chain-of-thought reasoning tokens. Cuts
    # latency 3-5× per call. Lossless for data-augmentation use cases
    # where we don't need the model to "think out loud."
    user_content = ("/no_think\n" + prompt) if disable_thinking else prompt
    body = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(teacher_url, data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"},
                                   method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
    choice = resp["choices"][0]
    if choice.get("finish_reason") == "length":
        print(f"  WARNING: teacher output truncated at max_tokens={max_tokens}; "
              "JSON is likely incomplete and will fail to parse", flush=True)
    return choice["message"]["content"]


def parse_teacher_output(content: str) -> list[dict]:
    """Extract JSON array of {instruction, response} from teacher response."""
    s = content.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # Try to find first [...] block if there's preamble
    if not s.startswith("["):
        m = re.search(r"\[.*\]", s, re.DOTALL)
        if not m:
            return []
        s = m.group(0)
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        return []
    if not isinstance(d, list):
        return []
    return [item for item in d if isinstance(item, dict) and "instruction" in item and "response" in item]


# -------------------- Main loop --------------------


def load_jsonl(p: Path) -> list[dict]:
    out = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds-in", type=Path, default=DEFAULT_SEEDS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p.add_argument("--variations-per-seed", type=int, default=10)
    p.add_argument("--teacher-url", default="http://127.0.0.1:1234/v1/chat/completions")
    p.add_argument("--teacher-model", default="qwen3-14b")
    p.add_argument("--max-seeds", type=int, default=None,
                     help="cap number of seeds to multiply (for quick runs)")
    p.add_argument("--only-new", action="store_true",
                     help="multiply only the v10 new seeds (intent in dictate/edit/non-answer-action), not the v9-converted rows")
    p.add_argument("--retries", type=int, default=1,
                     help="how many times to retry a seed if the teacher returns invalid output (default 1; bad seeds shouldn't burn time)")
    p.add_argument("--cooldown-seconds", type=float, default=0.0,
                     help="sleep N seconds between seeds to let the GPU cool. 0 = full throttle. 10 = noticeable cooling. 30 = aggressive cool.")
    args = p.parse_args()

    if not args.seeds_in.exists():
        raise SystemExit(f"missing seeds: {args.seeds_in}")
    if not args.registry.exists():
        raise SystemExit(f"missing registry: {args.registry}")

    seeds = load_jsonl(args.seeds_in)
    registry = load_registry(args.registry)

    if args.only_new:
        # Skip rows that look like the v9-converted ones (intent=answer with empty text,
        # or AX.press from clickLabel — both are 1-shot per v9).
        # Heuristic: if instruction contains "user said:" and response intent is answer with empty text → v9 carryover
        kept = []
        for s in seeds:
            try:
                r = json.loads(s["response"])
                if r.get("intent") == "answer" and r.get("payload", {}).get("text", "") == "":
                    continue  # likely v9-carryover
                kept.append(s)
            except: kept.append(s)
        seeds = kept
        print(f"only-new: kept {len(seeds)} seeds")
    if args.max_seeds:
        seeds = seeds[:args.max_seeds]

    print(f"multiplying {len(seeds)} seeds × {args.variations_per_seed} variations each", flush=True)
    print(f"teacher: {args.teacher_url} ({args.teacher_model})", flush=True)
    print(f"target: ~{len(seeds) * args.variations_per_seed} rows", flush=True)

    # Smoke-test teacher latency before unleashing on all seeds. If the
    # smoke call takes >30s, abort — the script will not finish in
    # reasonable time and the user should diagnose teacher config first.
    print("smoke-testing teacher latency (1 real call with /no_think)...", flush=True)
    smoke_start = time.time()
    try:
        smoke_prompt = "/no_think\nreply with the single character: 1"
        body = {"model": args.teacher_model,
                "messages": [{"role": "user", "content": smoke_prompt}],
                "max_tokens": 50, "temperature": 0.0}
        req = urllib.request.Request(args.teacher_url, data=json.dumps(body).encode(),
                                      headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            r.read()
        smoke_ms = (time.time() - smoke_start) * 1000
        if smoke_ms > 30_000:
            # One call produces ALL variations for a seed, so total wall is
            # roughly smoke latency × number of seeds.
            print(f"  ERROR: teacher latency {smoke_ms:.0f}ms exceeds 30s. Try a smaller teacher.", flush=True)
            print(f"  expected total wall: ~{smoke_ms * len(seeds) / 1000 / 60:.0f} min — aborting", flush=True)
            sys.exit(1)
        print(f"teacher latency: {smoke_ms:.0f}ms — proceeding", flush=True)
    except Exception as e:
        timed_out = isinstance(e, TimeoutError) or isinstance(getattr(e, "reason", None), TimeoutError)
        if timed_out:
            print("TEACHER TOO SLOW: teacher responded too slowly (>60s)", flush=True)
            print("hint: use a smaller/faster teacher model, or check GPU load", flush=True)
        else:
            print(f"TEACHER UNREACHABLE: {e}", flush=True)
            print("hint: start LM Studio + load a 4B+ model, or change --teacher-url/--teacher-model", flush=True)
        return

    all_multiplied: list[dict] = []
    n_invalid = 0

    for i, seed in enumerate(seeds, 1):
        prompt = MULTIPLY_PROMPT_TEMPLATE.format(
            n=args.variations_per_seed,
            instruction=json.dumps(seed["instruction"]),
            response=json.dumps(seed["response"]),
        )
        t0 = time.time()
        variations: list[dict] = []
        for attempt in range(args.retries + 1):
            try:
                # Scale max_tokens with variation count: each variation repeats
                # the full instruction text, so 10 variations easily exceed a
                # fixed 2000-token budget and truncate mid-JSON.
                content = call_teacher(args.teacher_url, args.teacher_model, prompt,
                                       max_tokens=max(2000, 350 * args.variations_per_seed))
            except Exception as e:
                print(f"  seed {i}: teacher call failed ({e}); skipping")
                break
            cand = parse_teacher_output(content)
            valid = []
            for c in cand:
                ok, _reason = validate_v10(c["response"], registry)
                if ok:
                    valid.append(c)
                else:
                    n_invalid += 1
            if valid:
                variations = valid
                break
        elapsed = time.time() - t0
        all_multiplied.extend(variations)
        print(f"  seed {i}/{len(seeds)}: +{len(variations)} variations ({elapsed:.1f}s, total {len(all_multiplied)})",
              flush=True)
        # Incremental save every 10 seeds — if the run is killed, we don't
        # lose all work like the 2026-06-09 hour-long waste.
        if i % 10 == 0:
            partial_out = args.out.with_suffix(".partial.jsonl")
            write_jsonl(partial_out, seeds + all_multiplied)
            print(f"    (saved partial: {partial_out})", flush=True)
        # Cooldown: let the GPU breathe between seeds. Doesn't cap peak
        # temp per call, but cuts sustained thermal load by adding idle
        # time. 10-30s is a good range; 0 = no cooldown.
        if args.cooldown_seconds > 0 and i < len(seeds):
            time.sleep(args.cooldown_seconds)

    print(f"\ninvalid (rejected) variations: {n_invalid}")
    # Merge: keep original seeds + multiplied
    merged = seeds + all_multiplied
    write_jsonl(args.out, merged)
    print(f"wrote {len(merged)} rows → {args.out}")
    print(f"  originals: {len(seeds)}")
    print(f"  multiplied: {len(all_multiplied)}")


if __name__ == "__main__":
    main()
