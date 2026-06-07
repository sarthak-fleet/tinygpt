# GitHub Corpus Parity Decision

Date: 2026-06-06

Decision: keep `native-mac/Sources/TinyGPTData/GitHubCorpus.swift` for now.

The Swift and Python paths are not schema-equivalent yet, so deleting the
legacy Swift extractor would drop training data behavior rather than just
removing duplicate code.

## Compared Paths

- Swift legacy path: `native-mac/Sources/TinyGPTData/GitHubCorpus.swift`
- Python data-prep path: `scripts/data-prep/github_reader.py`

## Divergence

The Swift path emits SFT records:

```json
{"instruction":"...","response":"...","metadata":{...}}
```

The Python shim currently emits router records:

```json
{"query":"...","tool":"...","metadata":{...}}
```

That schema difference is not cosmetic. The Swift `issues-prs` extractor
links closed issues to PRs, fetches the PR body and diff, then writes the
issue as `instruction` and the PR/diff bundle as `response`. The Python
`issues-prs` reader currently writes the issue text as `query` and routes it
to `github_issue_triage`; it does not preserve the issue-to-PR diff response.

The same mismatch exists for reviews and commits: the Swift path produces
SFT-style response records, while the Python path produces tool-routing rows.

## Outcome

Do not delete `GitHubCorpus.swift` in this PRD pass. Keep the deprecated
dispatch path until the Python shim grows an SFT-compatible mode, such as:

```text
scripts/data-prep/prep_data.py --source github --format sft
```

At that point parity should be re-run on small, medium, and large repos with
row-count, schema, and sample-record checks before deleting the Swift path.
