# App Intents comparison for Planner v7

Status: research landed 2026-06-07  
Supports: `docs/prds/factory-planner-v7-tools-in-prompt.md`

## Sources

- Apple App Intents overview:
  <https://developer.apple.com/documentation/appintents/app-intents>
- `AppIntent` protocol:
  <https://developer.apple.com/documentation/appintents/appintent>
- `AppEntity` protocol:
  <https://developer.apple.com/documentation/appintents/appentity>
- `EntityQuery` and entity query overview:
  <https://developer.apple.com/documentation/appintents/entityquery>,
  <https://developer.apple.com/documentation/appintents/entity-queries>
- `AppEnum` protocol:
  <https://developer.apple.com/documentation/appintents/appenum>
- `AppShortcutsProvider`, `AppShortcut`, and App Shortcuts overview:
  <https://developer.apple.com/documentation/appintents/appshortcutsprovider>,
  <https://developer.apple.com/documentation/appintents/appshortcut>,
  <https://developer.apple.com/documentation/appintents/app-shortcuts>
- Parameter clarification and confirmation APIs:
  <https://developer.apple.com/documentation/appintents/intentparameter/requestvalue(_:)-592nd>,
  <https://developer.apple.com/documentation/appintents/appintent/requestconfirmation()>
- WWDC22 sessions:
  <https://developer.apple.com/videos/play/wwdc2022/10170/>,
  <https://developer.apple.com/videos/play/wwdc2022/10169/>

## Executive summary

Apple App Intents is not a small global verb taxonomy. It is closer to a
typed action framework:

- An `AppIntent` is one app-specific action with parameters and a `perform()`
  method.
- Apple adds specialized protocols for common action families, such as
  `OpenIntent`, `DeleteIntent`, `SetValueIntent`, `ShowInAppSearchResultsIntent`,
  `PlayVideoIntent`, `CameraCaptureIntent`, and media / live activity /
  widget-specific protocols.
- `AppEntity` makes nouns first-class. Entities have stable IDs,
  display representations, optional URLs, and query providers.
- `EntityQuery` is the important retrieval primitive. It resolves entity IDs,
  suggests entities, and supports natural-language or property-based lookup.
- `AppEnum` is the fixed-value argument primitive.
- `IntentParameter` handles missing values, disambiguation, and dynamic
  options.
- `AppShortcutsProvider` exposes preconfigured phrases and shortcut tiles so
  Siri, Spotlight, and Shortcuts can discover the actions.

The main lesson for v7: do not encode apps as tool names and do not treat
targets as loose strings. Use a small verb set, but make entities and enums
first-class in the schema.

## App Intents taxonomy summary

Apple's action model has three layers.

First, there is the generic action layer. `AppIntent` is the base capability:
it exposes app functionality to Siri, Shortcuts, Spotlight, widgets, and other
system experiences. Each intent has a human-readable `title`, a description,
parameters, a parameter summary, and a `perform()` method.

Second, Apple has specialized action families. The public docs list protocols
such as `OpenIntent`, `DeleteIntent`, `SetValueIntent`,
`ShowInAppSearchResultsIntent`, `SnippetIntent`, `URLRepresentableIntent`,
`AudioPlaybackIntent`, `CameraCaptureIntent`, `PlayVideoIntent`,
`LiveActivityStartingIntent`, and `WidgetConfigurationIntent`. This is the
closest thing to a verb taxonomy, but it is an extensible protocol hierarchy,
not a fixed command vocabulary.

Third, Apple has assistant schemas for Siri and Apple Intelligence. The docs
tell developers not to adopt `AssistantIntent` directly; instead they use the
`AssistantIntent(schema:)` macro. That is a strong hint that Apple wants common
domains to map into system-owned schemas rather than every app inventing names.

Apple's noun model is as important as its action model. `AppEntity` exposes
app-specific concepts like notes, trails, files, documents, photos, or settings.
Entities have persistent IDs and display representations. Queries retrieve
entities by ID, provide suggestions, and support spoken or typed resolution.

Parameterization is typed. Use `@Parameter` for inputs, `AppEnum` for static
choice sets, `AppEntity` for app data, and query providers for dynamic options.
Apple also has APIs for missing values, user choice, and confirmation.

Discovery is explicit. `AppShortcut` phrases, titles, descriptions, display
representations, and shortcut metadata train the system's matching behavior and
make shortcuts available without user setup.

## Proposed v7 taxonomy recap

The current v7 PRD proposes ten verbs:

| Current verb | Intended role |
|---|---|
| `read` | Pull information from a source |
| `act` | Execute an action against a target |
| `compose` | Draft text or structured content |
| `ask` | Ask the user for clarification |
| `say` | Speak or show a response |
| `recall` | Query long-term memory |
| `wait` | Wait for a condition |
| `navigate` | Open an app, URL, file, or view |
| `schedule` | Create a time-bound action |
| `search` | Search across a scope |

This is directionally right: bounded verbs, tools in the prompt, and apps as
values instead of retrain-only tool names. The weak spot is that several verbs
mix action type with source type. Apple avoids that by separating action,
entity, query, and enum schemas.

## Mapping table

| v7 verb | App Intents equivalent | Confidence | Notes |
|---|---|---:|---|
| `read` | `EntityQuery`, `AppEntity`, result-returning `AppIntent` | High | Apple treats retrieval as entity resolution/query, not as a generic `read` verb. |
| `act` | `AppIntent`, plus specialized protocols | Medium | Too broad. Apple uses domain-specific intents or protocols like `SetValueIntent` / `DeleteIntent`. |
| `compose` | App-specific `AppIntent` | Medium | Apple has no generic compose protocol; apps expose actions such as create/export/send. |
| `ask` | `requestValue`, `requestChoice`, `requestConfirmation` | High | Rename to clarify intent: missing value, disambiguation, or confirmation. |
| `say` | `IntentDialog`, returned `IntentResult`, snippets | Medium | Apple treats response as result/dialog, not an independent verb. Still useful for Pace voice UX. |
| `recall` | No direct App Intents equivalent; closest is entity query over indexed app data | Low | Memory is our runtime feature. Keep it, but model it as a query source. |
| `wait` | `ProgressReportingIntent`, long-running intent modes | Medium | Apple has progress/foreground continuation patterns, not a generic wait verb. |
| `navigate` | `OpenIntent`, `OpenURLIntent`, `URLRepresentableIntent` | High | Rename to `open`. Apple naming is clearer and narrower. |
| `schedule` | App-specific calendar/reminder intents | Medium | No generic scheduling protocol in App Intents docs; still useful for agent planning. |
| `search` | `EntityStringQuery`, `EntityPropertyQuery`, `ShowInAppSearchResultsIntent` | High | Fold into `query` unless the output is explicitly opening search UI. |

## Answers to the PRD questions

### 1. Verb count

There is no closed App Intents verb count. Apple exposes a generic
`AppIntent` protocol and a growing set of specialized intent protocols. Treat
the public specialized protocols as examples of durable action families, not as
the full taxonomy.

For v7, keep the vocabulary small, but borrow Apple's clearest specialized
families: open, set value, delete / perform, search results, media, and
foreground continuation.

### 2. Verb naming convention

Apple uses intent type names that are usually verb + noun or specialized verb
protocols: `OpenIntent`, `SetValueIntent`, `DeleteIntent`,
`ShowInAppSearchResultsIntent`, `PlayVideoIntent`. The human-facing title is
described as a short title using a verb and noun in title case.

For v7, prefer verb names that are concrete English imperatives:
`query`, `open`, `set`, `perform`, `compose`, `clarify`, `say`, `wait`,
`schedule`, `query_memory`.

### 3. Entity model

Yes, v7 needs an entity model. Apple makes entities first-class because the
hard part is often resolving "the thing" the user means. A string `target`
field is too weak for held-out tools.

Use:

```json
{
  "target": {
    "type": "app|window|element|file|url|calendar_event|memory|custom",
    "id": "stable-id-if-known",
    "label": "human-visible label",
    "query": "fallback natural-language selector"
  }
}
```

This mirrors `AppEntity` plus `EntityQuery` without requiring Apple framework
types in the model output.

### 4. Same verb across many entity types

Apple generally keeps the action stable and varies the parameter/entity type.
`OpenIntent` opens associated items; URL-representable intents/entities provide
universal links; entity queries resolve app-specific objects.

For v7, do not create `open_file`, `open_url`, `open_app`, and
`open_document`. Use `open(target: EntityRef)` and let the target type carry
the distinction.

### 5. Disambiguation

The current `ask` verb matches the broad idea, but the name is too vague.
Apple distinguishes missing values, choices, and confirmation. Rename `ask` to
`clarify` and add `kind`.

```json
{
  "verb": "clarify",
  "args": {
    "kind": "missing_value|choice|confirmation",
    "question": "Which calendar?",
    "choices": [{"id": "work", "label": "Work"}]
  }
}
```

### 6. Composition

Shortcuts composes multiple actions into workflows. App Intents expose single
actions that Shortcuts can chain. The v7 plan is consistent: the model emits
one structured call and the orchestrator chains calls.

Do not train v7 to emit long multi-step plans by default. Train it to emit one
call plus an optional `continuation_hint` when it expects another step.

### 7. Voice mapping

Borrow App Shortcuts' phrase discipline:

- Provide multiple natural phrases per tool schema.
- Include the app name / domain in some phrases but not all phrases.
- Use parameterized phrases where a visible or spoken entity fills a slot.
- Include negative phrases for common false matches.
- Keep display names and schema descriptions short and user-facing.

Training data should include shortcut-style phrase templates, not just
developer-ish commands.

## Recommended taxonomy edits

Lock v7 to this revised ten-verb set before SFT:

| Revised verb | Replaces | Why |
|---|---|---|
| `query` | `read`, most of `search` | Aligns with `EntityQuery`; covers get/list/find/summarize. |
| `perform` | most of `act` | Generic app/domain action when no specialized verb fits. |
| `set` | part of `act` | Borrow `SetValueIntent`; value changes deserve a stable verb. |
| `compose` | `compose` | Keep. Content generation is model-native. |
| `clarify` | `ask` | Matches value/choice/confirmation resolution. |
| `say` | `say` | Keep for voice/agent UX, but treat as response, not tool side effect. |
| `query_memory` | `recall` | Makes memory a query source and avoids vague nostalgia language. |
| `wait` | `wait` | Keep for event/time coupling. |
| `open` | `navigate` | Borrow `OpenIntent`; clearer than navigate. |
| `schedule` | `schedule` | Keep as agent-level planning verb. |

Drop top-level `search`. Use:

```json
{"verb": "query", "args": {"source": "files", "mode": "search", "query": "invoice"}}
```

Use `open` only when the desired side effect is to show a URL, app, document,
screen, or search-results surface.

## Schema conventions to borrow

- Every verb schema should have a short title and description, not just a JSON
  field list.
- Arguments should be typed as entity refs, enums, strings, booleans, numbers,
  or structured filters. Avoid `Any`.
- Static choices should be enums. Dynamic choices should be entity queries.
- Every entity ref should include `type`, optional `id`, optional `label`, and
  optional `query`.
- Clarification should be explicit: missing value, disambiguation, or
  confirmation.
- Confirmation is not the same as clarification. Use it for destructive,
  irreversible, or high-cost actions.
- Include prompt-visible examples with natural user language and expected JSON.
- Keep tool names stable; vary app/domain through parameters.

## Action items for the v7 PRD

1. Replace the proposed taxonomy table with the revised ten-verb set above.
2. Add an `EntityRef` schema shared by all verbs.
3. Replace `ask` examples with `clarify(kind=...)` examples.
4. Fold `search` examples into `query(mode=search)`.
5. Rename `navigate` examples to `open(target=...)`.
6. Split `act` examples into `perform(...)` and `set(...)`.
7. Add shortcut-style phrase templates to Stage B data generation.
8. Add held-out evals where the verb is known but the entity type is new.
9. Add a destructive-action eval that requires `clarify(kind=confirmation)`.
10. Keep v7 implementation gated until the ANE arc and owner authorization land.
