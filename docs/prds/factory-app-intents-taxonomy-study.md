---
name: App Intents taxonomy study — leverage Apple's verb design for v7
status: shipped-2026-06-07
owner: unassigned (parallel-agent task — research-only, no code)
created: 2026-06-08
priority: P1 — feeds factory-planner-v7-tools-in-prompt.md taxonomy lock-in
parallel-safe: yes (research output only)
supports: factory-planner-v7-tools-in-prompt.md (verb taxonomy section)
---

# PRD — Apple App Intents taxonomy study

## Ship note — 2026-06-07

Research output landed at `docs/learn/app-intents-comparison.md`.
The recommendation is to revise v7's ten verbs before SFT lock-in:
`query`, `perform`, `set`, `compose`, `clarify`, `say`,
`query_memory`, `wait`, `open`, `schedule`. The study also recommends
adding a shared `EntityRef` schema and folding top-level `search` into
`query(mode=search)`.

## Why this PRD exists

The v7 planner PRD proposes a 10-verb taxonomy (`read`, `act`,
`compose`, `ask`, `say`, `recall`, `wait`, `navigate`, `schedule`,
`search`). Per `[[feedback_leverage_first]]`: before designing our
own taxonomy, we should read Apple's App Intents framework — they
already designed one for Siri/Shortcuts at scale.

This is a research task: study Apple's design, compare to our
proposed taxonomy, recommend changes (additions, removals,
renamings) before v7 SFT data construction begins.

## Goal

Produce `docs/learn/app-intents-comparison.md` (~150-250 lines) that:

1. Summarizes Apple App Intents' verb / action taxonomy
2. Lists our proposed v7 taxonomy
3. Maps each v7 verb to App Intents equivalent (if any)
4. Flags gaps: verbs we have that App Intents doesn't, and vice versa
5. Recommends taxonomy adjustments before v7 SFT lock-in
6. Captures any naming conventions worth borrowing

## Scope — sources to read

**Required**:
- Apple App Intents framework documentation:
  - <https://developer.apple.com/documentation/appintents>
  - Specifically: `AppIntent`, `EntityQuery`, `EntityPropertyQuery`,
    `AppEnum`, `IntentParameter`, `ActionEnum`
- WWDC sessions on App Intents (2022 intro, 2023/2024 updates if released)
- The Shortcuts app's published action library (`shortcuts://` URLs
  document a lot of structure)

**Strongly recommended**:
- Apple's "Design for App Intents" HIG section
- AppleScript dictionary structure (older but related taxonomy)
- Voice Shortcuts examples in Apple's sample code

**Comparative — for context, not authority**:
- OpenAI function calling specification
- Anthropic tool use specification
- MCP tool definition specification
- Google Assistant actions taxonomy (if accessible)

## Specific questions to answer

For each, the dossier should give a clear answer with citations:

1. **Verb count**: how many distinct verbs does App Intents have?
   How does Apple group them?

2. **Verb naming convention**: are App Intents verbs noun-form
   ("CreateNote") or verb-form ("Create"), or both? How are args
   named?

3. **Entity model**: App Intents uses Entities (Notes, Photos,
   Reminders) as first-class. Do we need an equivalent for our
   taxonomy, or do "target" arg values suffice?

4. **Parameterization**: how does App Intents handle the case where
   the same verb applies to multiple entity types (e.g., "Open" works
   for File, URL, App, Document)? Does it have one verb or many?

5. **Disambiguation**: when a user request is ambiguous, App Intents
   has structured clarification. Does our `ask` verb match this
   pattern?

6. **Composition**: App Intents supports compound intents (multi-step
   shortcuts). Our v7 model emits one verb call; the orchestrator
   chains them. Is this consistent with Apple's design, or does Apple
   have a better composition model worth borrowing?

7. **Voice mapping**: App Intents has specific patterns for mapping
   natural language to actions (Siri's NL → intent mapping). What
   patterns are worth borrowing for our planner's training data
   construction?

## Scope — recommended output

Sections:

1. **App Intents taxonomy summary** (50-80 lines)
2. **Our proposed v7 taxonomy** (recap from v7 PRD, 30-40 lines)
3. **Mapping table**: v7 verb → App Intents equivalent (+ confidence)
4. **Gaps**: verbs in App Intents we don't have, verbs we have that
   don't map to App Intents
5. **Recommendations**:
   - Verbs to rename for alignment
   - Verbs to add (drawn from App Intents)
   - Verbs to drop (if redundant)
   - Schema conventions to borrow (arg naming, optional patterns)
6. **Action items for the v7 PRD**: specific edits to make to
   `factory-planner-v7-tools-in-prompt.md` before SFT begins

## Scope — out

- Building an App Intents adapter for Pace (separate; out of scope here)
- Implementing the taxonomy changes (that's v7 SFT work)
- Comparing to other voice assistant taxonomies in depth (Alexa,
  Google Assistant) — keep focus on Apple since we're Mac-native

## Acceptance

1. `docs/learn/app-intents-comparison.md` exists
2. All 7 questions answered with concrete references
3. Mapping table covers all 10 proposed v7 verbs
4. Recommendations are specific (not "consider X" but "rename `recall`
   to `query_memory` to match App Intents' pattern X" — with citation)
5. Action items are immediately actionable on v7 PRD
6. Maintainer can review and decide taxonomy lock-in in ~30 minutes

## Estimated effort

**1 day** of research + writing. App Intents docs are well-organized;
the comparison is mechanical once both taxonomies are laid out.

## Why this is leverage (per the principle)

Apple has shipped Siri + Shortcuts + Spotlight at hundreds of millions
of users. Their verb taxonomy survived the user-test fire that ours
hasn't. Adopting their proven naming + structure where it fits saves
us from re-litigating decisions Apple already made well — and where
we *do* diverge, the divergence is intentional, not accidental.

## Won't conflict with other elves

- No code touched, only docs
- New file in `docs/learn/`
- v7 PRD gets edited (small, targeted) AFTER this study lands
