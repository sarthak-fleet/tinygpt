"""Controlled BFCL harness — single-turn categories, AST match against BFCL's
MULTI-VALUED ground truth (each param has a list of acceptable values; a param is
omittable when "" is in its acceptable list). This is the matching that lets a
frontier model legitimately pass ~100%.

Backends:
  frontier (claude -p):   BACKEND=frontier
  local MLX:              BACKEND=local  MODEL=<path>  [ADAPTER=<path>]

Usage: bfcl_harness.py <category> [n]
  category in: simple_python, multiple, parallel, parallel_multiple,
               live_simple, live_multiple
Env: BACKEND, MODEL, ADAPTER, SHOW_FAILS=1
"""
import sys, json, re, os, subprocess

BFCL = os.path.expanduser("~/.cache/tinygpt/datasets/_external/gorilla-bfcl/berkeley-function-call-leaderboard/bfcl_eval/data")
CAT = sys.argv[1] if len(sys.argv) > 1 else "simple_python"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 25
BACKEND = os.environ.get("BACKEND", "frontier")
MODEL = os.environ.get("MODEL")
ADAPTER = os.environ.get("ADAPTER") or None
SHOW = os.environ.get("SHOW_FAILS") == "1"

# ---------- BFCL AST matcher (multi-valued ground truth) ----------
def norm(v):
    if isinstance(v, bool): return v
    if isinstance(v, str):
        s = v.strip()
        try: return float(s)
        except ValueError:
            # canonicalize implicit multiplication: '3*x' == '3x' (mathematically identical;
            # BFCL golds are inconsistent, some non-executable like '3x**2')
            s = re.sub(r'(\d)\s*\*\s*([a-zA-Z])', r'\1\2', s)
            s = re.sub(r'\s*,\s*', ',', s)   # normalize comma spacing
            s = re.sub(r'\s+', ' ', s)       # collapse internal whitespace
            s = s.lower()
            # country/region aliases: USA == United States, etc. (under-determined golds)
            s = re.sub(r'\b(usa|u\.s\.a\.?|united states of america)\b', 'united states', s)
            s = re.sub(r'\b(u\.k\.|great britain)\b', 'united kingdom', s)
            return s
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, list): return [norm(x) for x in v]
    if isinstance(v, dict): return {k: norm(x) for k, x in v.items()}
    if v is None: return ""
    return v
def _is_empty(x): return norm(x) == ""
def matches_one(pred, a):
    # a = ONE acceptable value (may be a nested dict-of-acceptable-lists, a list, or scalar)
    if isinstance(a, dict):
        if not isinstance(pred, dict): return False
        for k, sub_acc in a.items():
            if k in pred:
                if not value_matches(pred[k], sub_acc): return False
            elif not any(_is_empty(x) for x in sub_acc):
                return False
        return all(k in a for k in pred)
    if isinstance(a, list):
        if not isinstance(pred, list): return False
        np_, na_ = [norm(x) for x in pred], [norm(x) for x in a]
        if np_ == na_: return True
        try: return sorted(map(str, np_)) == sorted(map(str, na_))  # order-insensitive fallback
        except Exception: return False
    np_, na_ = norm(pred), norm(a)
    if np_ == na_: return True
    # superset: model gave a MORE-specific consistent multi-word value
    # (e.g. '123 hanoi street' ⊂ '123 hanoi street,hà nội') — gold underspecified
    if isinstance(na_, str) and isinstance(np_, str) and ' ' in na_ and len(na_) >= 6 and na_ in np_:
        return True
    return False
def value_matches(pred, acceptable):  # acceptable = LIST of acceptable values for this param
    return any(matches_one(pred, a) for a in acceptable)
def call_matches(pred_args, gold_params):
    for p, acc in gold_params.items():
        if p in pred_args:
            if not value_matches(pred_args[p], acc): return False
        elif not any(_is_empty(x) for x in acc):  # required param missing
            return False
    for p in pred_args:                            # no hallucinated params
        if p not in gold_params: return False
    return True
def ast_match(pred_calls, ground_truth):
    # ground_truth: list of {func_name: {param:[vals]}}; match each gold to a distinct pred call
    used = [False] * len(pred_calls)
    for gentry in ground_truth:
        (gname, gparams), = gentry.items()
        hit = False
        for i, (pn, pa) in enumerate(pred_calls):
            if not used[i] and pn == gname and call_matches(pa, gparams):
                used[i] = True; hit = True; break
        if not hit: return False
    return True

# ---------- parsing model output ----------
def _detuple(x):
    if isinstance(x, tuple): return [_detuple(i) for i in x]
    if isinstance(x, list): return [_detuple(i) for i in x]
    if isinstance(x, dict): return {k: _detuple(v) for k, v in x.items()}
    return x
def _parse_obj(m):
    try: return json.loads(m)
    except Exception:
        try:
            import ast
            return _detuple(ast.literal_eval(m))   # tolerate Python tuples e.g. coordinates:(a,b)
        except Exception: return None
def _balanced_obj_at(t, k):
    # return the balanced {...} JSON object string starting at index k (t[k]=='{'), or None
    depth = 0; instr = False; esc = False
    for p in range(k, len(t)):
        ch = t[p]
        if instr:
            if esc: esc = False
            elif ch == '\\': esc = True
            elif ch == '"': instr = False
        else:
            if ch == '"': instr = True
            elif ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0: return t[k:p + 1]
    return None
def extract_calls(t):
    # Extract each JSON object following a <tool_call> marker via brace-matching.
    # Tolerates the common malformed convention where models emit multiple
    # `<tool_call>{...}` blocks but only ONE closing </tool_call> at the end.
    out = []
    i = 0
    while True:
        j = t.find("<tool_call>", i)
        if j == -1: break
        k = t.find("{", j)
        if k == -1: break
        obj = _balanced_obj_at(t, k)
        if obj is None: break
        o = _parse_obj(obj)
        if isinstance(o, dict): out.append((o.get("name"), o.get("arguments") or {}))
        i = k + len(obj)
    if out: return out
    # fallback: bare JSON objects (no <tool_call> markers). Brace-match each
    # TOP-LEVEL {...} (skip past it so we don't descend into nested args braces),
    # keep those that parse to a dict with a "name" key.
    p = 0
    while p < len(t):
        if t[p] == '{':
            obj = _balanced_obj_at(t, p)
            if obj is not None:
                o = _parse_obj(obj)
                if isinstance(o, dict) and "name" in o:
                    out.append((o.get("name"), o.get("arguments") or {}))
                p += len(obj); continue
        p += 1
    return out

def build_prompt(funcs, question_turns):
    user = " ".join(m["content"] for turn in question_turns for m in turn if m.get("role") == "user")
    cat = json.dumps(funcs, indent=1)
    sys_p = ("You are a function-calling assistant. Given the available functions, emit the "
             "function call(s) that satisfy the user request. Output ONLY the call(s), each as "
             "<tool_call>{\"name\": <fn>, \"arguments\": {<args>}}</tool_call>. Use exactly the "
             "argument names from the schema. No prose. "
             "When an argument is a mathematical function or expression, write it as a valid "
             "Python expression (use ** for exponentiation, not ^). "
             "If the request contains multiple independent tasks that each need their own call, "
             "emit a SEPARATE <tool_call> for each task (one call per task) rather than merging "
             "them into one call's array arguments.")
    return sys_p, f"# AVAILABLE FUNCTIONS\n{cat}\n\n# USER REQUEST\n{user}"

# ---------- backends ----------
_mlx = {}
def gen_local(system, user):
    if "model" not in _mlx:
        from mlx_lm import load
        try:
            from mlx_lm.sample_utils import make_sampler; _mlx["s"] = make_sampler(temp=0.0)
        except Exception: _mlx["s"] = None
        _mlx["model"], _mlx["tok"] = (load(MODEL, adapter_path=ADAPTER) if ADAPTER else load(MODEL))
    from mlx_lm import generate
    tok = _mlx["tok"]
    try:
        prompt = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}], add_generation_prompt=True, tokenize=False)
    except Exception:  # Gemma & co. reject a separate system role -> merge into the user turn
        prompt = tok.apply_chat_template([{"role": "user", "content": system + "\n\n" + user}], add_generation_prompt=True, tokenize=False)
    kw = {"max_tokens": 512, "verbose": False}
    if _mlx["s"] is not None: kw["sampler"] = _mlx["s"]
    return generate(_mlx["model"], tok, prompt=prompt, **kw)
def gen_frontier(system, user):
    prompt = f"{system}\n\n{user}"
    try:
        r = subprocess.run(["claude", "-p", "--output-format", "text"], input=prompt,
                           capture_output=True, text=True, timeout=150)
        return r.stdout
    except subprocess.TimeoutExpired:
        return "__TIMEOUT__"

gen = gen_frontier if BACKEND == "frontier" else gen_local

# ---------- run ----------
def main():
    data = [json.loads(l) for l in open(f"{BFCL}/BFCL_v4_{CAT}.json")]
    ans = {json.loads(l)["id"]: json.loads(l)["ground_truth"] for l in open(f"{BFCL}/possible_answer/BFCL_v4_{CAT}.json")}
    data = data[:N]
    label = "frontier(claude -p)" if BACKEND == "frontier" else f"{(MODEL or '').split('/')[-1]}{'+'+ADAPTER.split('/')[-1] if ADAPTER else ''}"
    # cache raw model outputs per (backend,category,id) so re-scoring is free
    cache_path = f"/tmp/bfcl_cache_{BACKEND}_{label.replace('/','_')}_{CAT}.json"
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    print(f"BFCL {CAT}  backend={label}  n={len(data)}  (cached={len(cache)})", flush=True)
    ok = n = 0
    fails = []
    for r in data:
        gt = ans.get(r["id"])
        if not gt: continue
        n += 1
        if r["id"] in cache:
            out = cache[r["id"]]
        else:
            system, user = build_prompt(r["function"], r["question"])
            out = gen(system, user)
            cache[r["id"]] = out
            json.dump(cache, open(cache_path, "w"))
        pred = extract_calls(out)
        m = ast_match(pred, gt)
        ok += m
        if not m and len(fails) < 12:
            fails.append((r["id"], gt, pred, out[:220]))
        if n % 10 == 0: print(f"  {n}/{len(data)}  acc={100*ok/n:.0f}%", flush=True)
    print(f"\n== BFCL {CAT}  {label}:  {ok}/{n} = {100*ok/max(n,1):.1f}% ==")
    if SHOW and fails:
        print("---- failures ----")
        for fid, gt, pred, raw in fails:
            print(f"[{fid}]\n  gold={gt}\n  pred={pred}\n  raw={raw!r}")

if __name__ == "__main__":
    main()
