"""Multi-turn / agentic tool-calling eval — SKELETON (see docs/prds/multi-turn-agentic-eval.md).

Stateful BFCL multi_turn_base: the model's calls EXECUTE against backend instances
and results feed back across turns; scoring is end-to-end (final state / executed
path vs gold), not per-call AST. Frontier-gated.

STATUS: scaffolding. The TODO(executor) blocks are the real work — faithfully
executing calls against BFCL's involved_classes (vendor them from the gorilla repo).

Backends: BACKEND=frontier | local (MODEL=path). Usage: bfcl_multiturn_eval.py [n]
"""
import sys, json, os
sys.argv_backup = sys.argv
N = int(sys.argv[1]) if len(sys.argv) > 1 else 25
sys.argv = ["bfcl_ast_eval"]
import bfcl_ast_eval as h          # reuse: extract_calls, gen (local/frontier), BFCL path
sys.argv = sys.argv_backup

DATA = f"{h.BFCL}/BFCL_v4_multi_turn_base.json"

def instantiate_backends(example):
    """TODO(executor): build the stateful backend(s) named in example['involved_classes']
    from example['initial_config']. Vendor BFCL's gorilla `multi_turn` backend classes
    (GorillaFileSystem, etc.) rather than reimplement. Returns a name->instance map +
    a dispatch(call)->result function."""
    raise NotImplementedError("vendor BFCL involved_classes executor")

def execute(call, backends):
    """TODO(executor): route a parsed call {name, arguments} to the right backend method,
    return its result (to feed back into the transcript). Handle unknown/invalid calls."""
    raise NotImplementedError

def final_state_matches(backends, gold_path):
    """TODO(executor): compare the post-run backend state (and/or executed-call path) to
    the gold `path`. BFCL's multi-turn checker does state + response comparison."""
    raise NotImplementedError

def render_turn(example, history):
    """Render system(tools) + conversation-so-far (user turns + assistant calls + tool
    results) into a prompt. Reuse h.build_prompt's tool catalog + the running transcript."""
    # TODO: assemble messages from example['function'] + history; for now, single-turn shape.
    raise NotImplementedError

def run_example(example):
    backends = instantiate_backends(example)
    history = []
    for turn in example["question"]:           # each turn is a list of user msgs
        prompt = render_turn(example, history + turn)
        out = h.gen(*prompt)                    # prompt = (system, user)
        for call in h.extract_calls(out):
            result = execute(call, backends)
            history.append({"call": call, "result": result})
    return final_state_matches(backends, example.get("path"))

def main():
    data = [json.loads(l) for l in open(DATA)][:N]
    print(f"multi-turn eval (SKELETON)  backend={os.environ.get('BACKEND','frontier')}  n={len(data)}")
    print("Executor not yet implemented — see TODO(executor) blocks + the PRD.")
    # ok = sum(run_example(e) for e in data); print(f"task-completion: {ok}/{len(data)}")

if __name__ == "__main__":
    main()
