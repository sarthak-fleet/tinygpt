"""Eval-rigor probes (task #48):
  IRRELEVANCE  — model is shown tools but NONE fit; correct = emit NO call.
                 Measures over-triggering (false-positive tool calls).
  FORMAT-SENS  — paraphrase each prompt a few ways; report the AST-accuracy spread
                 (lightweight templated proxy; real BFCL-v4 uses LLM paraphrase).
Backends: BACKEND=frontier | local (MODEL=path). Usage: eval_probes.py [n_irrel] [n_fmt]
"""
import sys, json, os
sys.argv_backup = sys.argv
N_IRREL = int(sys.argv[1]) if len(sys.argv) > 1 else 50
N_FMT = int(sys.argv[2]) if len(sys.argv) > 2 else 15
sys.argv = ["bfcl_harness"]
import bfcl_harness as h
sys.argv = sys.argv_backup

# ---------- IRRELEVANCE: correct = no call emitted ----------
irr = [json.loads(l) for l in open(f"{h.BFCL}/BFCL_v4_live_irrelevance.json")][:N_IRREL]
abst = n = 0
for r in irr:
    n += 1
    system, user = h.build_prompt(r["function"], r["question"])
    out = h.gen(system, user)
    if not h.extract_calls(out):       # abstained = correct
        abst += 1
print(f"IRRELEVANCE (abstain-when-no-tool-fits): {abst}/{n} = {100*abst/max(n,1):.0f}%  "
      f"(higher=better; low => over-triggers / hallucinates calls)")

# ---------- FORMAT-SENSITIVITY: templated paraphrases, report spread ----------
PARAS = [
    lambda q: q,
    lambda q: "Please help me with this request. " + q,
    lambda q: q + "\n\nThanks!",
    lambda q: "I need the following done: " + q,
]
cat = "multiple"
data = [json.loads(l) for l in open(f"{h.BFCL}/BFCL_v4_{cat}.json")]
ans = {json.loads(l)["id"]: json.loads(l)["ground_truth"]
       for l in open(f"{h.BFCL}/possible_answer/BFCL_v4_{cat}.json")}
data = data[:N_FMT]
accs = []
for vi, para in enumerate(PARAS):
    ok = m = 0
    for r in data:
        gt = ans.get(r["id"])
        if not gt: continue
        m += 1
        q_turns = [[{"role": "user", "content": para(t[0]["content"])}] for t in r["question"]]
        system, user = h.build_prompt(r["function"], q_turns)
        if h.ast_match(h.extract_calls(h.gen(system, user)), gt): ok += 1
    accs.append(100 * ok / max(m, 1))
print(f"\nFORMAT-SENSITIVITY on {cat} (n={N_FMT}, {len(PARAS)} paraphrases):")
print(f"  per-variant acc: {[round(a) for a in accs]}  spread={max(accs)-min(accs):.0f}pp  "
      f"(big spread => brittle to phrasing; lightweight templated proxy)")
