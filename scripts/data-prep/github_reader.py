from __future__ import annotations

import os
from typing import Any, Iterable


API_ROOT = "https://api.github.com"


def read_github(repo: str, kinds: list[str], limit: int | None = None) -> Iterable[dict[str, Any]]:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps
        raise RuntimeError("GitHub ingestion requires requests; install scripts/data-prep dependencies") from exc

    session = requests.Session()
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    session.headers["Accept"] = "application/vnd.github+json"

    emitted = 0
    for kind in kinds:
        for row in _read_kind(session, repo, kind):
            yield row
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def _read_kind(session: Any, repo: str, kind: str) -> Iterable[dict[str, Any]]:
    if kind == "commits":
        for commit in _paginate(session, f"{API_ROOT}/repos/{repo}/commits", {"per_page": 100}):
            message = ((commit.get("commit") or {}).get("message") or "").strip()
            sha = commit.get("sha")
            if message and sha:
                yield {
                    "query": f"Summarize commit {sha[:12]} in {repo}.",
                    "tool": "github_commit_summary",
                    "metadata": {"source": "github", "kind": kind, "repo": repo, "sha": sha, "message": message},
                }
    elif kind == "reviews":
        for pull in _paginate(session, f"{API_ROOT}/repos/{repo}/pulls", {"state": "all", "per_page": 100}):
            number = pull.get("number")
            if number is None:
                continue
            for review in _paginate(session, f"{API_ROOT}/repos/{repo}/pulls/{number}/reviews", {"per_page": 100}):
                body = (review.get("body") or "").strip()
                if body:
                    yield {
                        "query": body,
                        "tool": "github_review_response",
                        "metadata": {"source": "github", "kind": kind, "repo": repo, "pull": number},
                    }
    elif kind == "issues-prs":
        for issue in _paginate(session, f"{API_ROOT}/repos/{repo}/issues", {"state": "closed", "per_page": 100}):
            if "pull_request" in issue:
                continue
            title = (issue.get("title") or "").strip()
            body = (issue.get("body") or "").strip()
            if title or body:
                yield {
                    "query": "\n\n".join(part for part in (title, body) if part),
                    "tool": "github_issue_triage",
                    "metadata": {"source": "github", "kind": kind, "repo": repo, "issue": issue.get("number")},
                }
    else:
        raise ValueError(f"unknown GitHub kind: {kind}")


def _paginate(session: Any, url: str, params: dict[str, Any]) -> Iterable[dict[str, Any]]:
    while url:
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            return
        for item in payload:
            if isinstance(item, dict):
                yield item
        url = response.links.get("next", {}).get("url")
        params = {}
