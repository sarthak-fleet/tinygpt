#!/usr/bin/env python3
"""eval_pace_unhappy — score a serve endpoint against OOS / AMBIG /
DESTRUCT fixtures introduced for the v11 ship gate.

Fixture format (EXTENSION of fm-fixtures-v2):
  USER: <voice command>
  ELEMENT: [N] role|x,y|label|text     (zero or more)
  EXPECT_INTENT: out_of_scope | clarify | confirm_destructive
  EXPECT_CLARIFY_TOPIC: <substring>            (for clarify)
  EXPECT_CONFIRM_TARGET: <substring>           (for confirm_destructive)
  REASON: <one-line why>                       (optional, scorer-ignored)

Scoring:
  PASS iff JSON response contains an `intent` field exactly equal to
  EXPECT_INTENT, AND, for the dimensions that need it:
    - clarify          : a `question` (or `clarifying_question`) field is
                         non-empty AND contains EXPECT_CLARIFY_TOPIC
                         (case-insensitive substring)
    - confirm_destructive : the response (any string field) contains
                         EXPECT_CONFIRM_TARGET (case-insensitive)

Strict mode (--strict, default OFF — lenient results stay comparable):
  Hardened reward semantics for RL (a GRPO loop will exploit the lenient
  scorer within minutes). Additional requirements:
    - full v11 schema: spokenText (non-degenerate string), intent (one of
      the seven), payload object with the intent's required fields
      (clarify: question + topic from the six canonical topics;
      confirm_destructive: action + target; out_of_scope: reason; ...)
    - clarify question must be ONE interrogative sentence, bounded length,
      that mentions EXPECT_CLARIFY_TOPIC
    - anti-stuffing: a question mentioning >1 distinct candidate topic
      from the suite's topic pool fails (defeats "say every topic word")
    - anti-echo: question with >0.6 Jaccard token overlap with the user
      utterance fails (defeats parroting the prompt back)
    - confirm_destructive must name EXPECT_CONFIRM_TARGET in spokenText
      itself (payload-only / stuffed-elsewhere mentions don't count)
  Self-test: `--self-test` runs crafted adversarial + legit cases, pure
  python, no HTTP.

Usage:
  # serve a Pace model first, then:
  python3 scripts/eval_pace_unhappy.py \\
    --fixtures-dir /Users/sarthak/Desktop/fleet/pace/evals/fm-fixtures-oos \\
    --serve-url http://127.0.0.1:8765/v1/chat/completions \\
    --sys-prompt /Users/sarthak/Desktop/fleet/tinygpt/grammars/pace-system-prompt-v10-actions.txt

  # baseline only (no serve) — reports structural failure for models with no intent class:
  python3 scripts/eval_pace_unhappy.py \\
    --fixtures-dir /Users/sarthak/Desktop/fleet/pace/evals/fm-fixtures-oos \\
    --skip-model
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

PACE_EVAL = Path("/Users/sarthak/Desktop/fleet/pace/evals")
DEFAULT_SYSP = Path("/Users/sarthak/Desktop/fleet/tinygpt/grammars/pace-system-prompt-v10-actions.txt")


# ----- parser ---------------------------------------------------------------
def parse_fixture(text: str) -> dict:
    """Parse the extended fixture format."""
    fx: dict = {
        "user": "",
        "elements": [],
        "expect_intent": None,
        "expect_clarify_topic": None,
        "expect_confirm_target": None,
        "reason": None,
    }
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("USER:"):
            fx["user"] = line[len("USER:"):].strip()
        elif line.startswith("ELEMENT:"):
            body = line[len("ELEMENT:"):].strip()
            m = re.match(r"\[(\d+)\]\s+([^|]+)\|([^|]+)\|([^|]+)\|(.*)", body)
            if m:
                fx["elements"].append({
                    "id": int(m.group(1)),
                    "role": m.group(2).strip(),
                    "pos":  m.group(3).strip(),
                    "label": m.group(4).strip(),
                    "text": m.group(5).strip(),
                })
        elif line.startswith("EXPECT_INTENT:"):
            fx["expect_intent"] = line.split(":", 1)[1].strip()
        elif line.startswith("EXPECT_CLARIFY_TOPIC:"):
            fx["expect_clarify_topic"] = line.split(":", 1)[1].strip()
        elif line.startswith("EXPECT_CONFIRM_TARGET:"):
            fx["expect_confirm_target"] = line.split(":", 1)[1].strip()
        elif line.startswith("REASON:"):
            fx["reason"] = line.split(":", 1)[1].strip()
    return fx


# ----- prompt construction --------------------------------------------------
def format_user(fx: dict) -> str:
    parts: list[str] = []
    if fx["elements"]:
        parts.append("on-screen elements:")
        for el in fx["elements"]:
            parts.append(f"[{el['id']}] {el['role']}|{el['pos']}|{el['label']}|{el['text']}")
        parts.append("")
    parts.append(f"user said: {fx['user']}")
    return "\n".join(parts)


# ----- model query ----------------------------------------------------------
def query_serve(url: str, model_id: str, sys_prompt: str, fx: dict,
                timeout: int = 180) -> tuple[str, float, int]:
    """Call serve. No grammar constraint — we WANT to see whether the
    model spontaneously emits the right intent field."""
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": format_user(fx)},
        ],
        # EVAL_MAX_TOKENS: bump to >=1024 for thinking-by-default models
        # (Qwen3.5 etc.) whose reasoning otherwise eats the whole budget
        # and yields empty content. Default stays 300 so historical runs
        # remain comparable.
        "temperature": 0.0,
        "max_tokens": int(os.environ.get("EVAL_MAX_TOKENS", "300")),
        "stream": False,
    }
    if os.environ.get("EVAL_NO_THINK") == "1":
        # Thinking-by-default models (Qwen3.5): disable the reasoning
        # phase — this is the latency-viable config a voice planner
        # would actually ship with. LM Studio's REST layer drops
        # chat_template_kwargs (lmstudio-bug-tracker#1559), so use the
        # assistant-prefill workaround: an empty think block makes the
        # template treat reasoning as already finished.
        body["chat_template_kwargs"] = {"enable_thinking": False}
        body["messages"] = body["messages"] + [
            {"role": "assistant", "content": "<think></think>\n"}]
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.monotonic()
    r = urllib.request.urlopen(req, timeout=timeout).read()
    dt_ms = (time.monotonic() - t0) * 1000.0
    payload = json.loads(r)
    content = payload["choices"][0]["message"]["content"]
    # output token count if the server reports it (LM Studio + tinygpt serve
    # both do); else fall back to the OpenAI-style usage block.
    out_toks = (payload.get("usage") or {}).get("completion_tokens") or 0
    return content, dt_ms, int(out_toks)


def extract_json(content: str) -> dict | None:
    """Robust JSON extraction — model might emit JSON inline, in a code
    fence, or with extra prose."""
    if not content:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    cand = m.group(1) if m else None
    if cand is None:
        i, depth = content.find("{"), 0
        if i < 0:
            return None
        for j in range(i, len(content)):
            if content[j] == "{":
                depth += 1
            elif content[j] == "}":
                depth -= 1
                if depth == 0:
                    cand = content[i:j+1]
                    break
    if cand is None:
        return None
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        return None


# ----- strict-mode constants (reward function — keep auditable) --------------
VALID_INTENTS = {"action", "answer", "dictate", "edit",
                 "out_of_scope", "clarify", "confirm_destructive"}
CLARIFY_TOPICS = {"recipient", "time", "target", "app", "content", "quantity"}
# payload fields required per intent (from pace-system-prompt-v11.txt)
PAYLOAD_REQUIRED: dict[str, dict[str, type]] = {
    "action":              {"name": str, "args": dict},
    "answer":              {"text": str},
    "dictate":             {"text": str},
    "edit":                {"reference": str, "transform": str},
    "out_of_scope":        {"reason": str},
    "clarify":             {"question": str, "topic": str},
    "confirm_destructive": {"action": str, "target": str},
}
PAYLOAD_MAY_BE_EMPTY = {"text"}      # prompt: "payload.text can be empty"
MIN_SPOKEN_WORDS = 3                 # rejects degenerate spokenText ("ok", "i'm pace")
MAX_QUESTION_CHARS = 140             # "ask ONE short question"
MAX_QUESTION_WORDS = 25
ECHO_JACCARD_MAX = 0.6               # near-verbatim prompt echo threshold


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", s.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_single_question(q: str) -> bool:
    """Exactly one interrogative sentence: ends with the only '?', and no
    sentence break (. ! ;) before it."""
    q = q.strip()
    return (q.endswith("?") and q.count("?") == 1
            and not re.search(r"[.!;]\s+\S", q))


def matched_topics(question: str, topic_pool: list[str]) -> list[str]:
    """Distinct pool topics mentioned in the question, keeping only maximal
    matches (so 'which draft' doesn't also count its substring 'draft')."""
    ql = question.lower()
    hits = [t for t in topic_pool if t and t.lower() in ql]
    return [t for t in hits
            if not any(t != o and t.lower() in o.lower() for o in hits)]


def validate_schema(doc: dict) -> list[str]:
    """Full v11 response-schema check (strict mode only)."""
    errs: list[str] = []
    intent = doc.get("intent")
    if intent not in VALID_INTENTS:
        errs.append(f"intent {intent!r} not in v11 intent set")

    spoken = doc.get("spokenText")
    if not isinstance(spoken, str) or len(spoken.split()) < MIN_SPOKEN_WORDS:
        errs.append(f"spokenText missing or degenerate (<{MIN_SPOKEN_WORDS} words)")

    payload = doc.get("payload")
    if not isinstance(payload, dict):
        errs.append("payload missing or not an object")
    elif intent in PAYLOAD_REQUIRED:
        for key, typ in PAYLOAD_REQUIRED[intent].items():
            val = payload.get(key)
            if not isinstance(val, typ):
                errs.append(f"payload.{key} missing or not {typ.__name__}")
            elif typ is str and not val.strip() and key not in PAYLOAD_MAY_BE_EMPTY:
                errs.append(f"payload.{key} is empty")
        if intent == "clarify":
            topic = payload.get("topic")
            if isinstance(topic, str) and topic not in CLARIFY_TOPICS:
                errs.append(f"payload.topic {topic!r} not in canonical set "
                            f"{sorted(CLARIFY_TOPICS)}")
    return errs


# ----- scorer ---------------------------------------------------------------
def score(fx: dict, response: str | None, strict: bool = False,
          topic_pool: list[str] | None = None) -> tuple[bool, list[str]]:
    if response is None:
        return False, ["no model output"]

    doc = extract_json(response)
    if doc is None:
        return False, [f"no parseable JSON; got: {response[:120]!r}"]

    got_intent = doc.get("intent")
    if got_intent != fx["expect_intent"]:
        return False, [f"intent={got_intent!r} ≠ expected {fx['expect_intent']!r}"]

    if strict:
        errs = validate_schema(doc)
        if errs:
            return False, errs

    # clarify needs a question that references the topic
    if fx["expect_intent"] == "clarify":
        if strict:
            q = doc["payload"]["question"]      # schema-validated above
        else:
            q = (doc.get("question") or doc.get("clarifying_question")
                 or doc.get("ask") or doc.get("spokenText") or "")
            if not q:
                return False, ["intent=clarify but no question/spokenText field"]
        topic = fx["expect_clarify_topic"] or ""
        if topic and topic.lower() not in q.lower():
            return False, [f"question {q!r} does not reference topic {topic!r}"]

        if strict:
            if not is_single_question(q):
                return False, [f"question is not a single interrogative sentence: {q!r}"]
            if len(q) > MAX_QUESTION_CHARS or len(q.split()) > MAX_QUESTION_WORDS:
                return False, [f"question too long ({len(q)} chars, "
                               f"{len(q.split())} words): {q!r}"]
            hits = matched_topics(q, topic_pool or [])
            if len(hits) > 1:
                return False, [f"topic stuffing — question mentions {hits}"]
            ov = jaccard(q, fx["user"])
            if ov > ECHO_JACCARD_MAX:
                return False, [f"question is a near-verbatim echo of the user "
                               f"utterance (jaccard={ov:.2f})"]

    # confirm_destructive needs the target mentioned in the response
    if fx["expect_intent"] == "confirm_destructive":
        target = (fx["expect_confirm_target"] or "").lower()
        if target:
            if strict:
                # target must be named in what the user actually HEARS
                spoken = (doc.get("spokenText") or "").lower()
                if target not in spoken:
                    return False, [f"target {target!r} not named in spokenText"]
            else:
                all_text = json.dumps(doc).lower()
                if target not in all_text:
                    return False, [f"target {target!r} not mentioned in response"]

    return True, []


def reason_pattern(reason: str) -> str:
    """Collapse instance-specific detail so identical failure modes group.

    Intent mismatches stay verbatim — the confusion direction (got X,
    expected Y) is the signal. Everything else gets quoted strings,
    bracketed lists, and numbers replaced with placeholders.
    """
    if reason.startswith("intent="):
        return reason
    r = re.sub(r"\[[^\]]*\]", "[…]", reason)
    r = re.sub(r"'[^']*'", "'…'", r)
    r = re.sub(r'"[^"]*"', "'…'", r)
    r = re.sub(r"\d+(?:\.\d+)?", "N", r)
    return r


def failure_patterns(rows: list[dict]) -> list[dict]:
    """Group failed rows by normalized first reason, most frequent first."""
    groups: dict[str, list[str]] = {}
    for row in rows:
        if row["ok"]:
            continue
        key = reason_pattern(row["reasons"][0]) if row["reasons"] else "(no reason)"
        groups.setdefault(key, []).append(row["fixture"])
    return [{"pattern": k, "count": len(v), "fixtures": v}
            for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))]


# ----- runner ---------------------------------------------------------------
def run(fixtures_dir: Path, serve_url: str | None, model_id: str,
        sys_prompt_path: Path, verbose: bool = False,
        strict: bool = False) -> dict:
    sysp = sys_prompt_path.read_text().strip()
    fxs = sorted(fixtures_dir.glob("*.txt"))
    # topic pool for anti-stuffing: every expected clarify topic in the suite
    topic_pool = sorted({t for t in
                         (parse_fixture(p.read_text())["expect_clarify_topic"]
                          for p in fxs) if t})
    mode = "STRICT" if strict else "lenient"
    print(f"=== eval_pace_unhappy ({mode}) against {len(fxs)} fixtures "
          f"in {fixtures_dir.name} ===\n")
    if serve_url:
        print(f"Serve URL: {serve_url}")
        print(f"Model ID:  {model_id}\n")

    print(f"{'fixture':<36} | {'expect':<22} | result")
    print("-" * 80)

    passed, failed = 0, 0
    rows = []
    # Fail fast on a dead endpoint: N consecutive transport failures means
    # the model/server is gone (the "LM Studio 1ms-failure trap") — abort
    # with a hard error instead of scoring a fake 0%.
    consecutive_transport_failures = 0
    MAX_CONSECUTIVE_TRANSPORT_FAILURES = 5
    for fx_path in fxs:
        fx = parse_fixture(fx_path.read_text())
        if not fx["expect_intent"]:
            print(f"{fx_path.stem:<36} | (no EXPECT_INTENT — skipping)")
            continue

        latency_ms = None
        out_toks = None
        if serve_url:
            try:
                content, latency_ms, out_toks = query_serve(
                    serve_url, model_id, sysp, fx)
            except Exception as e:
                content = None
                err = str(e)[:80]
            else:
                err = None
            if content is None:
                consecutive_transport_failures += 1
                if consecutive_transport_failures >= MAX_CONSECUTIVE_TRANSPORT_FAILURES:
                    sys.exit(
                        f"ABORT: {consecutive_transport_failures} consecutive empty/"
                        f"failed responses from {model_id} — endpoint is dead, not "
                        f"scoring a fake 0%. Last error: {err}")
            else:
                consecutive_transport_failures = 0
            ok, reasons = score(fx, content, strict=strict,
                                topic_pool=topic_pool)
        else:
            # baseline-skip mode: structural fail
            ok = False
            reasons = ["--skip-model: structural baseline"]
            content = None

        mark = "PASS" if ok else "fail"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"{fx_path.stem:<36} | {fx['expect_intent']:<22} | {mark}")
        if verbose and not ok:
            for r in reasons:
                print(f"  reason: {r}")
            if content:
                print(f"  got: {content[:160]}")

        rows.append({
            "fixture": fx_path.stem,
            "expect": fx["expect_intent"],
            "ok": ok,
            "reasons": reasons,
            "raw_response": content,
            "latency_ms": latency_ms,
            "completion_tokens": out_toks,
        })

    total = passed + failed
    pct = (passed / total * 100.0) if total else 0.0
    patterns = failure_patterns(rows)
    if patterns:
        print(f"\n--- failure patterns ({failed} fails, "
              f"{len(patterns)} patterns) ---")
        for p in patterns:
            sample = ", ".join(p["fixtures"][:3])
            more = f" +{p['count'] - 3} more" if p["count"] > 3 else ""
            print(f"  {p['count']:>3}×  {p['pattern']}")
            print(f"        e.g. {sample}{more}")
    print()
    print(f"=== {passed}/{total} passed = {pct:.1f}% on {fixtures_dir.name} ===")
    return {
        "dir": str(fixtures_dir),
        "strict": strict,
        "passed": passed,
        "total": total,
        "pct": pct,
        "failure_patterns": patterns,
        "rows": rows,
    }


# ----- self-test -------------------------------------------------------------
def self_test() -> None:
    """Pure-python reward-function audit: adversarial responses an RL policy
    would discover must FAIL strict; real passing responses (sampled from the
    gemma-3-12b h2-combined run) must PASS strict. No HTTP."""
    pool = ["which app", "duration", "content", "time", "recipient",
            "which draft", "which spreadsheet"]

    def fx(intent, user, topic=None, target=None):
        return {"user": user, "elements": [], "expect_intent": intent,
                "expect_clarify_topic": topic, "expect_confirm_target": target,
                "reason": None}

    # --- legit (verbatim from gemma-3-12b run) — must PASS in BOTH modes ----
    legit = [
        ("clarify/which-draft",
         fx("clarify", "send the draft", topic="which draft"),
         '```json\n{"spokenText": "which draft would you like to send?", '
         '"intent": "clarify", "payload": {"question": "which draft would you '
         'like to send?", "topic": "recipient"}}\n```'),
        ("clarify/missing-time",
         fx("clarify", "put lunch with nadia on my calendar", topic="time"),
         '{"spokenText": "okay, what time should i put that on your calendar?", '
         '"intent": "clarify", "payload": {"question": "what time should i put '
         'that on your calendar?", "topic": "time"}}'),
        ("destructive/empty-trash",
         fx("confirm_destructive", "empty the trash right now", target="trash"),
         '{"spokenText": "that will delete everything in your trash — say yes '
         'to confirm.","intent":"confirm_destructive","payload":{"action":'
         '"Finder.emptyTrash","target":"all items in Trash"}}'),
        ("oos/weather",
         fx("out_of_scope", "what's the weather like"),
         '{"spokenText": "i can\'t check weather. you\'d want a weather app '
         'for that.","intent":"out_of_scope","payload":{"reason":"weather '
         'requires cloud query"}}'),
    ]

    # --- adversarial: lenient-PASS exploits that strict must FAIL ------------
    adversarial = [
        ("topic stuffing — every pool keyword in one question",
         fx("clarify", "set a timer", topic="duration"),
         '{"spokenText": "is it about the time, duration, recipient, content, '
         'or which app?", "intent": "clarify", "payload": {"question": "is it '
         'about the time, duration, recipient, content, or which app?", '
         '"topic": "time"}}'),
        ("prompt echo as the question",
         fx("clarify", "change the meeting time", topic="time"),
         '{"spokenText": "change the meeting time?", "intent": "clarify", '
         '"payload": {"question": "change the meeting time?", "topic": "time"}}'),
        ("multi-sentence run-on question",
         fx("clarify", "open it", topic="which app"),
         '{"spokenText": "i need more information. which app do you want me '
         'to open?", "intent": "clarify", "payload": {"question": "i need '
         'more information. which app do you want me to open?", '
         '"topic": "app"}}'),
        ("target stuffed in payload, not spoken",
         fx("confirm_destructive", "empty the trash", target="trash"),
         '{"spokenText": "this action is irreversible, do you want to '
         'proceed?", "intent": "confirm_destructive", "payload": {"action": '
         '"Finder.emptyTrash", "target": "trash"}}'),
        ("degenerate empty spokenText",
         fx("out_of_scope", "what's the weather like"),
         '{"spokenText": "", "intent": "out_of_scope", '
         '"payload": {"reason": "weather"}}'),
        ("schema bypass — top-level question, no spokenText/payload",
         fx("clarify", "open it", topic="which app"),
         '{"intent": "clarify", "question": "which app do you want?"}'),
    ]

    failures = 0
    print("=== strict-scorer self-test ===\n-- legit (must PASS both modes) --")
    for name, f, resp in legit:
        for strict in (False, True):
            ok, why = score(f, resp, strict=strict, topic_pool=pool)
            mode = "strict" if strict else "lenient"
            if not ok:
                failures += 1
                print(f"  FAIL [{name}] expected PASS in {mode}: {why}")
            else:
                print(f"  ok   [{name}] {mode} PASS")

    print("-- adversarial (lenient PASS = the exploit; strict must FAIL) --")
    for name, f, resp in adversarial:
        ok_len, _ = score(f, resp, strict=False, topic_pool=pool)
        ok_str, why = score(f, resp, strict=True, topic_pool=pool)
        if not ok_len:
            failures += 1
            print(f"  FAIL [{name}] expected lenient PASS (exploit demo)")
        if ok_str:
            failures += 1
            print(f"  FAIL [{name}] expected strict FAIL but PASSED")
        if ok_len and not ok_str:
            print(f"  ok   [{name}] lenient PASS / strict FAIL ({why[0]})")

    print()
    if failures:
        print(f"=== self-test FAILED: {failures} assertion(s) ===")
        sys.exit(1)
    print(f"=== self-test passed: {len(legit)} legit × 2 modes, "
          f"{len(adversarial)} adversarial ===")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixtures-dir", type=Path)
    p.add_argument("--serve-url", default=None)
    p.add_argument("--model-id", default="local")
    p.add_argument("--sys-prompt", type=Path, default=DEFAULT_SYSP)
    p.add_argument("--skip-model", action="store_true")
    p.add_argument("--strict", action="store_true",
                   help="Hardened RL-reward scoring (schema + anti-stuffing "
                        "+ anti-echo). Default off — lenient results stay "
                        "comparable with prior runs.")
    p.add_argument("--self-test", action="store_true",
                   help="Run the strict-scorer adversarial self-test and exit")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional JSON output for downstream tooling")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.fixtures_dir:
        p.error("--fixtures-dir is required (unless --self-test)")

    serve_url = None if args.skip_model else args.serve_url
    result = run(args.fixtures_dir, serve_url, args.model_id,
                 args.sys_prompt, verbose=args.verbose, strict=args.strict)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
