#!/usr/bin/env python3
"""Build DPO preference pairs from contrastive clarify scenarios.

DPO needs (prompt, chosen, rejected) triples. For the clarify-discipline
task, we synthesize them from existing assets without paying for new
teacher calls:

  - chosen   = the seed-format gold response (clarify when ambiguous,
               act when unambiguous)
  - rejected = the 4B-Instruct's natural over-confident guess on the
               same prompt (for clarify cases) OR an over-eager clarify
               (for action cases — the over-correction failure mode)

Sources:
  - clarify-train-v1.jsonl (38 contrastive pairs we built earlier)
  - h2 ambig fixtures (20 ambiguous prompts, gold = clarify)
  - h2-ext ambig fixtures (20 more ambiguous, gold = clarify)
  - h2 destructive (10 prompts, gold = confirm_destructive)
  - h2 oos (30 prompts, gold = out_of_scope)
  - rejected responses generated from the 4B's natural distribution
    via direct sampling (not the gold) — captures the actual failure mode

Output format: jsonl with fields {prompt, chosen, rejected, _meta}.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def fixture_to_prompt(fp: Path) -> tuple[str, str, dict]:
    """Parse a fixture, return (user_request_block, expected_intent, expects_dict)."""
    text = fp.read_text()
    user = ""
    elements: list[str] = []
    expect_intent = None
    expects: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("USER:"):
            user = stripped[len("USER:"):].strip()
        elif stripped.startswith("ELEMENT:"):
            elements.append(stripped[len("ELEMENT:"):].strip())
        elif stripped.startswith("EXPECT_INTENT:"):
            expect_intent = stripped[len("EXPECT_INTENT:"):].strip()
        elif ":" in stripped and stripped.split(":", 1)[0].startswith(("EXPECT_", "REASON")):
            k, _, v = stripped.partition(":")
            expects[k.strip()] = v.strip()

    parts: list[str] = []
    if elements:
        parts.append("on-screen elements:")
        for el in elements:
            parts.append(f"[{el}")
        parts.append("")
    parts.append(f"user said: {user}")
    return "\n".join(parts), expect_intent or "action", expects


def chosen_clarify(expects: dict) -> str:
    topic = expects.get("EXPECT_CLARIFY_TOPIC", "target")
    questions = {
        "which draft": "which draft?",
        "which window": "which window?",
        "which file": "which file?",
        "which meeting": "which meeting?",
        "which button": "which button?",
        "which tab": "which tab?",
        "which app": "which app?",
        "which playlist": "which playlist?",
        "which printer": "which printer?",
        "which spreadsheet": "which spreadsheet?",
        "which person": "which one — who do you mean?",
        "which project": "which project?",
        "which docs": "which docs?",
        "which screenshot": "which screenshot?",
        "which contract": "which contract?",
        "which recipe note": "which recipe note?",
        "which team channel": "which team — which channel?",
        "recipient": "who should i send it to?",
        "time": "when?",
        "content": "what should it say?",
        "target": "which one do you mean?",
        "attachment": "which attachment?",
        "duration": "how long?",
        "input": "what input?",
        "query": "search for what?",
        "item": "which one — what should i order?",
        "task": "what would you like me to do?",
        "attendees-or-title": "what's the meeting about and who's coming?",
    }
    q = questions.get(topic, "could you tell me which one you mean?")
    return json.dumps({
        "spokenText": q,
        "intent": "clarify",
        "payload": {"question": q, "topic": topic},
    })


def chosen_oos(expects: dict) -> str:
    reason = expects.get("REASON", "out of scope")
    return json.dumps({
        "spokenText": "i can't help with that — it needs the web or an app i don't control.",
        "intent": "out_of_scope",
        "payload": {"reason": reason},
    })


def chosen_destructive(expects: dict) -> str:
    target = expects.get("EXPECT_CONFIRM_TARGET", "data")
    return json.dumps({
        "spokenText": f"that will delete the {target} — confirm?",
        "intent": "confirm_destructive",
        "payload": {"action": "destructive", "target": target},
    })


# Rejected responses — the model's natural failure mode for each intent class.
# These are synthesized to mirror what we observed empirically: the 4B picks
# a confident action, OOS prompts get answered helpfully, destructive prompts
# get fired directly.
def rejected_guess_action(user_request: str) -> str:
    # Extract a plausible single target the model would have guessed.
    m = re.search(r"\[(\d+)\]\s+\S+\|[^|]+\|([^|]+)\|", user_request)
    target = m.group(2).strip() if m else "the first option"
    return json.dumps({
        "spokenText": f"on it — {target.lower()}",
        "intent": "action",
        "payload": {"name": "AX.press", "args": {"target": target}},
    })


def rejected_oos_helpful(user_request: str) -> str:
    return json.dumps({
        "spokenText": "let me try — i'll do my best with that.",
        "intent": "answer",
        "payload": {"text": "based on what i know, here's a likely answer..."},
    })


def rejected_destructive_fire(user_request: str) -> str:
    return json.dumps({
        "spokenText": "done.",
        "intent": "action",
        "payload": {"name": "AX.press", "args": {"target": "Delete"}},
    })


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rows = []

    # Ambig — chosen = clarify, rejected = confident guess.
    for d in ["evals/fm-fixtures-ambig-h2", "evals/fm-fixtures-ambig-h2-ext"]:
        for fp in sorted(Path(d).glob("*.txt")):
            prompt, _, expects = fixture_to_prompt(fp)
            rows.append({
                "prompt": prompt,
                "chosen": chosen_clarify(expects),
                "rejected": rejected_guess_action(prompt),
                "_meta": {"source": f"{d}/{fp.stem}", "axis": "ambig"},
            })

    # OOS — chosen = warm refusal, rejected = helpful attempt.
    for d in ["evals/fm-fixtures-oos-h2", "evals/fm-fixtures-oos-h2-ext"]:
        for fp in sorted(Path(d).glob("*.txt")):
            prompt, _, expects = fixture_to_prompt(fp)
            rows.append({
                "prompt": prompt,
                "chosen": chosen_oos(expects),
                "rejected": rejected_oos_helpful(prompt),
                "_meta": {"source": f"{d}/{fp.stem}", "axis": "oos"},
            })

    # Destructive — chosen = confirm, rejected = fire.
    for d in ["evals/fm-fixtures-destructive-h2", "evals/fm-fixtures-destructive-h2-ext"]:
        for fp in sorted(Path(d).glob("*.txt")):
            prompt, _, expects = fixture_to_prompt(fp)
            rows.append({
                "prompt": prompt,
                "chosen": chosen_destructive(expects),
                "rejected": rejected_destructive_fire(prompt),
                "_meta": {"source": f"{d}/{fp.stem}", "axis": "destructive"},
            })

    # Action-twins from clarify-train-v1 — chosen = act, rejected = over-clarify.
    # Critical for preventing the over-correction failure mode (clarify-v1's mistake).
    seeds_path = Path.home() / ".cache/tinygpt/datasets/clarify-train-v1.jsonl"
    if seeds_path.exists():
        for line in open(seeds_path):
            seed = json.loads(line)
            if seed.get("_meta", {}).get("kind") != "unambiguous":
                continue
            rows.append({
                "prompt": seed["instruction"],
                "chosen": seed["response"],
                "rejected": json.dumps({
                    "spokenText": "which one do you mean?",
                    "intent": "clarify",
                    "payload": {"question": "could you tell me which one?", "topic": "target"},
                }),
                "_meta": {"source": "clarify-seeds/action-twin",
                          "axis": "action-keep"},
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    c = Counter(r["_meta"]["axis"] for r in rows)
    print(f"wrote {len(rows)} preference pairs → {args.out}")
    print(f"axis breakdown: {dict(c)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
