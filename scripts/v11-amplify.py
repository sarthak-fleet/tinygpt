#!/usr/bin/env python3 -u
"""v11-amplify.py — thinking-teacher + N=4 + judge amplifier for v11 seeds.

Fixes the four mistakes from the v10 multiplier waste:
  1. Thinking ENABLED (was disabled, killed reasoning about scope)
  2. Scope-spec system prompt (teacher told exactly what's in/out of scope)
  3. Single-seed expansion (one seed per call, no corpus drift)
  4. Critic-judge filter (keep only top-N rated rows)

Pipeline:
  pace-v11-seed.jsonl  (60 rows in)
       │
       │ for each seed, generate N=4 variations with thinking-teacher
       ▼
  ~240 candidates
       │
       │ critic scores each 1-5
       ▼
  keep score >= 4
       ▼
  pace-v11-amplified.jsonl  (~150-180 rows out, target)

Usage:
  # boot LM Studio with Qwen3-14B-MLX-4bit first, then:
  python3 -u scripts/v11-amplify.py \\
      --seeds ~/.cache/tinygpt/datasets/pace-v11-seed.jsonl \\
      --out   ~/.cache/tinygpt/datasets/pace-v11-amplified.jsonl

Expected wall: ~60-100 minutes on Qwen3-14B + thinking with N=4.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path


# ---- the scope spec is the key correction from v10 -------------------------
PACE_SCOPE_SPEC = """\
Pace is a Mac voice assistant. Pace can ONLY do these 12 actions:
  AX.press, AX.setValue, AX.scroll, App.launch, App.activate,
  Mail.draft, Cal.event, Reminders.add, Notes.create, Shortcut.run,
  Window.snap, Clipboard.read.

The intent enum is:
  - "action"               do one of the 12 actions
  - "answer"               answer a question about what's on screen
  - "out_of_scope"         user wants something Pace cannot do
  - "clarify"              user's request is ambiguous, ask back
  - "confirm_destructive"  user wants something irreversible, get confirmation

Pace voice style: lowercase, casual, warm, written for the ear.
One or two sentences. No emojis, no markdown, no bullets.
Spell out small numbers. Never say "simply" or "just"."""


# ---- teacher + critic prompt templates -------------------------------------
GENERATOR_SYSTEM = f"""You generate one training row for a Pace planner.

{PACE_SCOPE_SPEC}

You will receive ONE seed row showing the desired intent class with a specific
user prompt and assistant response. Generate ONE NEW training row that:
- maps to the SAME intent class as the seed
- has a fresh and DIFFERENT user prompt phrasing
- uses Pace's voice in the spokenText
- has a valid {{spokenText, intent, payload}} shape

Output EXACTLY one JSON object, no prose, no markdown fences:
{{"instruction": "user said: <new prompt>", "response": "<json string of spokenText/intent/payload>"}}

Do not copy the seed's prompt. Diversify the phrasing significantly.
"""

CRITIC_SYSTEM = f"""You rate Pace planner training rows 1-5.

{PACE_SCOPE_SPEC}

Score the row 1-5 on the combination of:
- Schema correctness: response parses as JSON, intent equals the expected class.
- Pace voice: lowercase, casual, no emojis/markdown, written for the ear.
- Intent match: the user's prompt actually maps to the claimed intent class.
- Diversity from a generic refusal/clarify: not just "I can't" 100 times.

Return EXACTLY one line: SCORE: <1-5>
No prose, no explanation.

5 = perfect example. 4 = good with minor issue. 3 = wrong class or weak. 2 = clearly wrong. 1 = unusable.
"""


# ---- API helpers (thinking ENABLED) ----------------------------------------
def call_lm(teacher_url: str, model: str, system: str, user: str,
            max_tokens: int = 800, temperature: float = 0.7,
            timeout: int = 240) -> str:
    """Call LM Studio's OpenAI-compatible /v1/chat/completions.
    Thinking ENABLED — no /no_think marker. Qwen3-14B will burn 200-500
    reasoning tokens per call but produce much higher-quality output."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        teacher_url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    return resp["choices"][0]["message"]["content"]


def _strip_think_tags(s: str) -> str:
    """Qwen3 thinking returns <think>...</think> blocks at the top.
    Strip them so JSON extraction sees the actual output."""
    s = re.sub(r"<think>.*?</think>\s*", "", s, flags=re.DOTALL)
    return s.strip()


def parse_one_row(content: str) -> dict | None:
    """Extract the single training row JSON from teacher output."""
    s = _strip_think_tags(content)
    # strip code fences
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()
    # find first balanced {...}
    if not s.startswith("{"):
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            return None
        s = m.group(0)
    # tolerate trailing text after the JSON
    depth = 0
    end = -1
    for i, ch in enumerate(s):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    try:
        d = json.loads(s[:end])
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    if "instruction" not in d or "response" not in d:
        return None
    # response is supposed to be a JSON string — validate
    try:
        inner = json.loads(d["response"]) if isinstance(d["response"], str) else d["response"]
    except json.JSONDecodeError:
        return None
    if not isinstance(inner, dict) or "intent" not in inner:
        return None
    # normalize: response always serialized as JSON string
    d["response"] = json.dumps(inner, ensure_ascii=False)
    return d


def parse_score(content: str) -> int | None:
    s = _strip_think_tags(content)
    m = re.search(r"SCORE\s*:\s*([1-5])", s)
    if not m:
        return None
    return int(m.group(1))


# ---- main amplification loop -----------------------------------------------
def amplify(seeds_path: Path, out_path: Path, teacher_url: str, model: str,
            n_per_seed: int, min_score: int, gen_max_tokens: int,
            critic_max_tokens: int, gen_temp: float, partial_save_every: int,
            cooldown_seconds: float, smoke_only: bool) -> None:
    print(f"=== v11-amplify start {time.strftime('%H:%M:%S')} ===")
    print(f"  seeds:   {seeds_path}")
    print(f"  out:     {out_path}")
    print(f"  teacher: {teacher_url} ({model})")
    print(f"  N per seed: {n_per_seed}  min-score: {min_score}")
    print()

    # smoke test: confirm teacher reachable, thinking works
    print("  smoke test (one call with thinking ON)...", flush=True)
    t0 = time.time()
    try:
        out = call_lm(teacher_url, model, GENERATOR_SYSTEM,
                      "Generate one row for intent=out_of_scope. Seed:\n"
                      'INSTRUCTION: user said: hi\nRESPONSE: {"spokenText":"hi","intent":"answer","payload":{}}',
                      max_tokens=600, temperature=0.7, timeout=120)
        dt = time.time() - t0
        first = out[:80].replace("\n", " ")
        print(f"  smoke ok in {dt:.1f}s: {first!r}")
    except Exception as e:
        print(f"  SMOKE FAILED: {e}", file=sys.stderr)
        sys.exit(2)
    if smoke_only:
        print("--smoke-only set; exiting after smoke.")
        return

    seeds = [json.loads(l) for l in seeds_path.read_text().splitlines() if l.strip()]
    print(f"  loaded {len(seeds)} seeds")
    print()

    kept: list[dict] = []
    rejected_count = 0
    parse_fail_count = 0
    t_start = time.time()

    for i, seed in enumerate(seeds, 1):
        seed_intent = "(unknown)"
        try:
            seed_intent = json.loads(seed["response"])["intent"]
        except Exception:
            pass

        seed_text = (
            f"INSTRUCTION: {seed['instruction']}\n"
            f"RESPONSE: {seed['response']}\n"
            f"EXPECTED INTENT CLASS: {seed_intent}"
        )

        gen_kept_for_seed = 0
        for k in range(n_per_seed):
            try:
                gen_out = call_lm(
                    teacher_url, model,
                    GENERATOR_SYSTEM, seed_text,
                    max_tokens=gen_max_tokens, temperature=gen_temp)
            except Exception as e:
                print(f"  seed {i} gen {k+1}: API fail ({e})", flush=True)
                continue

            row = parse_one_row(gen_out)
            if row is None:
                parse_fail_count += 1
                continue

            # critic
            critic_user = (
                f"USER PROMPT: {row['instruction']}\n"
                f"ASSISTANT RESPONSE: {row['response']}\n"
                f"EXPECTED INTENT: {seed_intent}"
            )
            try:
                critic_out = call_lm(
                    teacher_url, model,
                    CRITIC_SYSTEM, critic_user,
                    max_tokens=critic_max_tokens, temperature=0.2)
            except Exception:
                continue
            score = parse_score(critic_out)
            if score is None:
                continue
            row["_meta"] = {
                "source_seed_idx": i - 1,
                "intent_class": seed_intent,
                "critic_score": score,
            }
            if score >= min_score:
                kept.append(row)
                gen_kept_for_seed += 1
            else:
                rejected_count += 1

        elapsed = time.time() - t_start
        rate = i / elapsed if elapsed > 0 else 0
        eta = (len(seeds) - i) / rate if rate > 0 else float("inf")
        print(f"  seed {i}/{len(seeds)} ({seed_intent}): "
              f"kept {gen_kept_for_seed}/{n_per_seed} "
              f"· total kept {len(kept)} · rej {rejected_count} · parse-fail {parse_fail_count} "
              f"· eta {eta/60:.1f}min",
              flush=True)

        if i % partial_save_every == 0:
            partial = out_path.with_suffix(".partial.jsonl")
            partial.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n")
            print(f"    (partial saved: {partial})", flush=True)

        if cooldown_seconds > 0 and i < len(seeds):
            time.sleep(cooldown_seconds)

    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n")
    pass_rate = (len(kept) / (len(seeds) * n_per_seed) * 100) if seeds else 0
    print()
    print(f"=== v11-amplify DONE ===")
    print(f"  in:  {len(seeds)} seeds × {n_per_seed} = {len(seeds)*n_per_seed} candidates")
    print(f"  kept: {len(kept)} ({pass_rate:.1f}% judge pass-rate)")
    print(f"  rejected (low score): {rejected_count}")
    print(f"  parse failures:       {parse_fail_count}")
    print(f"  wall: {(time.time()-t_start)/60:.1f}min")
    print(f"  → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=Path, required=True)
    p.add_argument("--out",   type=Path, required=True)
    p.add_argument("--teacher-url", default="http://127.0.0.1:1234/v1/chat/completions")
    p.add_argument("--model", default="qwen/qwen3-14b")
    p.add_argument("--n-per-seed", type=int, default=4)
    p.add_argument("--min-score",  type=int, default=4,
                   help="Keep candidates with critic score >= this (default 4)")
    p.add_argument("--gen-max-tokens",    type=int, default=800)
    p.add_argument("--critic-max-tokens", type=int, default=300)
    p.add_argument("--gen-temp",          type=float, default=0.7)
    p.add_argument("--partial-save-every", type=int, default=10)
    p.add_argument("--cooldown-seconds",   type=float, default=0,
                   help="Sleep N seconds between seeds for thermal headroom (e.g. 5-15)")
    p.add_argument("--smoke-only", action="store_true",
                   help="Run only the smoke test and exit (latency/reachability check)")
    args = p.parse_args()

    amplify(
        seeds_path=args.seeds, out_path=args.out,
        teacher_url=args.teacher_url, model=args.model,
        n_per_seed=args.n_per_seed, min_score=args.min_score,
        gen_max_tokens=args.gen_max_tokens,
        critic_max_tokens=args.critic_max_tokens,
        gen_temp=args.gen_temp,
        partial_save_every=args.partial_save_every,
        cooldown_seconds=args.cooldown_seconds,
        smoke_only=args.smoke_only,
    )


if __name__ == "__main__":
    main()
