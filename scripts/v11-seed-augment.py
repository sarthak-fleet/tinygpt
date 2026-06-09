#!/usr/bin/env python3
"""v11-seed-augment.py — additional hand-curated training seeds.

Produces ~30 NEW (prompt, response) pairs that are NOT in the held-out
fm-fixtures eval suite. Used as additional amplifier prompts so v11
training data exhibits more phrasing diversity than the 60 fixture-based
seeds alone provide.

The output of this script is APPENDED to pace-v11-seed.jsonl. The
existing fixture-based 60 rows stay; this script adds ~30 more.

KEY INVARIANT: prompts here must NOT overlap with any prompt in
pace/evals/fm-fixtures-{oos,ambig,destructive}/. Verified post-build
via a substring check.

Output: ~/.cache/tinygpt/datasets/pace-v11-seed.jsonl  (extended in place)
"""
import json
from pathlib import Path

OUT_FILE = Path.home() / ".cache/tinygpt/datasets/pace-v11-seed.jsonl"
PACE_EVAL = Path("/Users/sarthak/Desktop/fleet/pace/evals")


# ---- 30 augmented seeds, hand-written, distinct from fixture prompts -------

OOS_AUGMENTED = [
    # narrow knowledge that requires no Mac action
    ("how old is keanu reeves",                                    "i can't look up facts. spotlight or safari can.",          "knowledge query, no Mac action"),
    ("what's the capital of australia",                            "i can't answer geography. try a web search.",               "knowledge query, no Mac action"),
    ("what's the square root of 144",                              "i can't compute. spotlight does math.",                     "math query, no Mac action"),
    ("when did world war two end",                                 "i can't recall historical dates. safari can.",              "knowledge query, no Mac action"),
    # external service requests
    ("send a venmo to dad for 50 bucks",                            "i can't send payments. open venmo for that.",              "payment service not integrated"),
    ("call an uber to the airport",                                 "i can't book rides. uber app handles that.",                "ride service not integrated"),
    ("order doordash for tonight",                                  "i can't order food. doordash app handles that.",            "delivery service not integrated"),
    ("check my facebook messages",                                  "i can't access facebook. open the app or messenger.",       "social platform not integrated"),
    # non-mac control
    ("turn on the air conditioner",                                 "i can't control hvac.",                                     "no smart-home integration"),
    ("lock my front door",                                          "i can't control locks.",                                    "no smart-home integration"),
    ("start my car",                                                "i can't talk to your car.",                                 "vehicle integration not supported"),
    # streaming / continuous
    ("watch for new emails from sarah",                             "i can't monitor in the background.",                        "no event monitor"),
    ("alert me if the price drops",                                 "i can't watch prices over time.",                           "no continuous monitor"),
]

AMBIG_AUGMENTED = [
    # missing target
    ("rename this",                                                "what do you want to call it?",                              "what to rename"),
    ("save it as",                                                 "save it as what?",                                          "the new filename"),
    ("move this to the folder",                                    "which folder do you want to move it to?",                   "destination folder"),
    ("change the color",                                           "what color do you want?",                                   "the new color"),
    # pronoun reference
    ("show me the same one",                                       "the same as what?",                                          "the reference item"),
    ("do that again",                                              "what should i repeat?",                                      "the previous action"),
    # missing field
    ("set the deadline",                                           "what's the deadline?",                                       "the date"),
    ("add a tag",                                                  "what tag should i add?",                                     "the tag name"),
    ("group by",                                                   "group by what — date, sender, or size?",                    "the grouping key"),
    # ambiguous app
    ("open the music app",                                         "do you mean apple music or spotify?",                       "which music app"),
    ("shoot a message off",                                        "via messages, mail, or slack?",                              "which app"),
]

DESTRUCT_AUGMENTED = [
    ("delete this entire folder including subfolders",             ("delete this folder and all subfolders", "Finder.deleteFolder")),
    ("permanently delete all photos from today",                   ("permanently delete today's photos", "Photos.permanentDelete")),
    ("remove all my bookmarks",                                    ("remove every bookmark", "Browser.deleteAllBookmarks")),
    ("clear my entire download history",                           ("clear all download history", "Browser.clearDownloadHistory")),
    ("delete all my notes in this folder",                         ("delete every note in this folder", "Notes.deleteFolder")),
    ("close every window in every app",                             ("close all windows across all apps", "Window.closeAll")),
    ("erase the contents of this file",                            ("erase this file's contents", "Doc.erase")),
    ("unmount all my external drives",                             ("unmount all external drives", "Finder.unmountAll")),
    ("revoke this app's permissions",                              ("revoke permissions for this app", "Privacy.revokeAll")),
]


# ---- builders -------------------------------------------------------------

def build_oos(user: str, spoken: str, reason: str) -> dict:
    return {
        "instruction": f"user said: {user}",
        "response": json.dumps({
            "spokenText": spoken,
            "intent": "out_of_scope",
            "payload": {"reason": reason},
        }, ensure_ascii=False),
        "_meta": {"source": "augmented", "intent_class": "out_of_scope"},
    }


def build_ambig(user: str, spoken: str, topic: str) -> dict:
    return {
        "instruction": f"user said: {user}",
        "response": json.dumps({
            "spokenText": spoken,
            "intent": "clarify",
            "payload": {"question": spoken, "topic": topic},
        }, ensure_ascii=False),
        "_meta": {"source": "augmented", "intent_class": "clarify"},
    }


def build_destruct(user: str, target_desc: str, action_name: str) -> dict:
    spoken = f"that will {target_desc} — say yes to confirm."
    return {
        "instruction": f"user said: {user}",
        "response": json.dumps({
            "spokenText": spoken,
            "intent": "confirm_destructive",
            "payload": {"action": action_name, "target": target_desc},
        }, ensure_ascii=False),
        "_meta": {"source": "augmented", "intent_class": "confirm_destructive"},
    }


# ---- main -----------------------------------------------------------------

def main():
    if not OUT_FILE.exists():
        print(f"ERROR: {OUT_FILE} missing; run build-v11-seed-jsonl.py first")
        return

    # Load existing 60 fixture-based seeds
    existing = [json.loads(l) for l in OUT_FILE.read_text().splitlines() if l.strip()]
    existing_prompts = {row["instruction"].split("user said: ", 1)[-1].strip().lower()
                        for row in existing}

    # Build new rows
    new_rows: list[dict] = []
    for user, spoken, reason in OOS_AUGMENTED:
        new_rows.append(build_oos(user, spoken, reason))
    for user, spoken, topic in AMBIG_AUGMENTED:
        new_rows.append(build_ambig(user, spoken, topic))
    for user, (target_desc, action_name) in DESTRUCT_AUGMENTED:
        new_rows.append(build_destruct(user, target_desc, action_name))

    # Invariant check: no augmented prompt appears in fixture-based seeds
    collisions = []
    for row in new_rows:
        prompt = row["instruction"].split("user said: ", 1)[-1].strip().lower()
        if prompt in existing_prompts:
            collisions.append(prompt)
    if collisions:
        print("ERROR: augmented prompts collide with fixture-based seeds:")
        for p in collisions:
            print(f"  {p}")
        return

    # Invariant check: no augmented prompt matches any fixture USER: line
    fixture_prompts: set[str] = set()
    for d in ["fm-fixtures-oos", "fm-fixtures-ambig", "fm-fixtures-destructive"]:
        for fp in (PACE_EVAL / d).glob("*.txt"):
            for line in fp.read_text().splitlines():
                if line.startswith("USER:"):
                    fixture_prompts.add(line[len("USER:"):].strip().lower())
    collisions = []
    for row in new_rows:
        prompt = row["instruction"].split("user said: ", 1)[-1].strip().lower()
        if prompt in fixture_prompts:
            collisions.append(prompt)
    if collisions:
        print("ERROR: augmented prompts collide with held-out eval fixtures:")
        for p in collisions:
            print(f"  {p}")
        return

    # Append + write back
    combined = existing + new_rows
    OUT_FILE.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in combined) + "\n")
    print(f"wrote {len(combined)} rows to {OUT_FILE}")
    print(f"  existing (fixture-based): {len(existing)}")
    print(f"  augmented (hand):         {len(new_rows)}")
    by_class = {}
    for r in combined:
        c = r["_meta"]["intent_class"]
        by_class[c] = by_class.get(c, 0) + 1
    print(f"  totals by class:")
    for c, n in sorted(by_class.items()):
        print(f"    {c:24s} {n}")


if __name__ == "__main__":
    main()
