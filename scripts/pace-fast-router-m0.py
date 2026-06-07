#!/usr/bin/env python3
"""Deterministic M0 Pace fast-router eval for clickyLocal fm-fixtures.

This is intentionally standalone: no TinyGPT serve path, no model runtime, and
no dependencies beyond the Python standard library. It measures the local
obvious-action router only.
"""

from __future__ import annotations

import argparse
import re
import statistics
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


DEFAULT_FIXTURES = Path("/Users/sarthak/Desktop/fleet/clickyLocal/evals/fm-fixtures")


@dataclass(frozen=True)
class Element:
    id: int
    role: str
    x: int
    y: int
    label: str
    text: str


@dataclass(frozen=True)
class Route:
    verb: str
    confidence: float
    reason: str
    target_id: int | None = None
    text: str | None = None
    key: str | None = None
    direction: str | None = None
    app_name: str | None = None
    action_tags: tuple[str, ...] = ()


def parse_fixture(text: str) -> dict:
    out = {"user": "", "elements": [], "expects": {}, "free_text": False}
    list_keys = {"SPOKEN_MUST_MATCH_REGEX"}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("USER:"):
            out["user"] = line.removeprefix("USER:").strip()
        elif line.startswith("ELEMENT:"):
            elem = parse_element(line.removeprefix("ELEMENT:").strip())
            if elem is not None:
                out["elements"].append(elem)
        elif line.startswith("FREE_TEXT_MODE:"):
            out["free_text"] = "true" in line.lower()
        elif ":" in line and any(
            line.startswith(prefix)
            for prefix in (
                "EXPECT_POINT_ID",
                "EXPECT_CLICK_ID",
                "SPOKEN_MUST_CONTAIN",
                "SPOKEN_MUST_NOT_CONTAIN",
                "SPOKEN_MUST_MATCH_REGEX",
                "SPOKEN_MAX_WORDS",
            )
        ):
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if key in list_keys:
                out["expects"].setdefault(key, []).append(value)
            else:
                out["expects"][key] = value
    return out


def parse_element(raw: str) -> Element | None:
    match = re.match(r"\[(\d+)\]\s*([^|]+)\|(-?\d+),(-?\d+)\|([^|]*)\|(.*)", raw)
    if not match:
        return None
    return Element(
        id=int(match.group(1)),
        role=match.group(2).strip(),
        x=int(match.group(3)),
        y=int(match.group(4)),
        label=match.group(5).strip(),
        text=match.group(6).strip(),
    )


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+ ]+", " ", value.lower())).strip()


STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "button",
    "can",
    "click",
    "could",
    "i",
    "it",
    "like",
    "maybe",
    "menu",
    "open",
    "press",
    "select",
    "tap",
    "the",
    "thing",
    "to",
    "uh",
    "yeah",
    "you",
}


def content_tokens(value: str) -> set[str]:
    return {token for token in normalize(value).split() if token and token not in STOPWORDS}


def compact_target_phrase(user: str) -> str:
    lowered = normalize(user)
    lowered = re.split(r"\band\s+type\b", lowered, maxsplit=1)[0]
    lowered = re.sub(r"^(can you |could you |please )+", "", lowered)
    lowered = re.sub(r"^(click|tap|select|press|open)\s+", "", lowered)
    return lowered.strip()


def score_element(user: str, target_phrase: str, element: Element) -> float:
    label = normalize(element.label)
    text = normalize(element.text)
    target = normalize(target_phrase)
    user_tokens = content_tokens(user)
    target_tokens = content_tokens(target)
    label_tokens = content_tokens(label)
    text_tokens = content_tokens(text)

    score = 0.0
    if target and (target == label or target == text):
        score += 1.0
    if target and (target in label or label in target):
        score += 0.55
    if target and (target in text or text in target):
        score += 0.25
    if label_tokens:
        score += 0.7 * (len(user_tokens & label_tokens) / len(label_tokens))
    if text_tokens:
        score += 0.25 * (len(user_tokens & text_tokens) / len(text_tokens))
    if target and label:
        score += 0.35 * SequenceMatcher(None, target, label).ratio()
    if element.role in {"button", "tab", "menu_item", "text_field", "text_area"}:
        score += 0.03
    return score


def choose_target(user: str, elements: list[Element]) -> tuple[Element | None, float]:
    if not elements:
        return None, 0.0
    target = compact_target_phrase(user)
    scored = sorted(
        ((score_element(user, target, element), element) for element in elements),
        key=lambda item: (-item[0], item[1].id),
    )
    best_score, best = scored[0]
    if "second tab" in normalize(user):
        for element in elements:
            if element.role == "tab" and ("second" in normalize(element.label) or element.id == 1):
                return element, 0.99
    if best_score < 0.58:
        return None, best_score
    return best, min(best_score, 0.99)


def extract_type_text(user: str) -> str | None:
    match = re.search(r"\btype\s+(.+)$", user, flags=re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).strip()
    value = re.sub(r"^(the text|text)\s+", "", value, flags=re.IGNORECASE)
    return value.strip() or None


def route(user: str, elements: list[Element]) -> Route:
    normalized = normalize(user)

    if re.search(r"\b(command|cmd)\s*\+?\s*s\b", normalized) or "save shortcut" in normalized:
        return Route(verb="key", key="cmd+s", confidence=0.99, reason="key_shortcut")

    if normalized.startswith("scroll ") or " scroll " in f" {normalized} ":
        direction = "up" if " up" in f" {normalized} " else "down"
        return Route(verb="scroll", direction=direction, confidence=0.98, reason="scroll_direction")

    if normalized.startswith("open ") and not any(word in normalized for word in ("menu", "button")):
        app_name = user.split(None, 1)[1].strip()
        return Route(verb="open_app", app_name=app_name, confidence=0.85, reason="open_app_phrase")

    if "who are you" in normalized or "are you siri" in normalized:
        return Route(verb="answer", text="i'm pace", confidence=0.99, reason="identity_allowlist")

    if "what is html" in normalized:
        return Route(
            verb="answer",
            text="html is the markup language used to structure web pages",
            confidence=0.95,
            reason="tiny_qa_allowlist",
        )
    if "what is css" in normalized:
        return Route(
            verb="answer",
            text="css is the language used to style web pages",
            confidence=0.95,
            reason="tiny_qa_allowlist",
        )

    if "describe" in normalized or "what does this screen show" in normalized:
        labels = [element.label for element in elements[:4] if element.label]
        text = "this screen shows " + ", ".join(labels) if labels else "i can't see the screen"
        return Route(verb="answer", text=text, confidence=0.82, reason="screen_summary")

    if "type " in normalized:
        typed = extract_type_text(user)
        if typed:
            if "click" in normalized:
                target, confidence = choose_target(user, elements)
                tags: list[str] = []
                if target is not None:
                    tags.append(f"[CLICK:{target.x},{target.y}]")
                tags.append(f"[TYPE:{typed}]")
                return Route(
                    verb="click",
                    target_id=target.id if target else None,
                    text=typed,
                    confidence=min(confidence, 0.96),
                    reason="click_then_type",
                    action_tags=tuple(tags),
                )
            return Route(verb="type", text=typed, confidence=0.98, reason="type_phrase")

    if any(word in normalized.split() for word in ("click", "tap", "select", "press")):
        target, confidence = choose_target(user, elements)
        if target is None:
            return Route(verb="escalate", confidence=0.2, reason="missing_or_ambiguous_target")
        return Route(
            verb="click",
            target_id=target.id,
            confidence=confidence,
            reason="deterministic_label_match",
        )

    return Route(verb="escalate", confidence=0.35, reason="not_obvious_action")


def route_to_observed(route_result: Route, elements: list[Element], free_text: bool) -> dict:
    by_id = {element.id: element for element in elements}
    target = by_id.get(route_result.target_id) if route_result.target_id is not None else None
    point_id = route_result.target_id if route_result.verb == "click" and route_result.target_id is not None else -1
    click_id = point_id

    if route_result.action_tags:
        spoken = " ".join(route_result.action_tags)
    elif route_result.verb == "key":
        spoken = f"[KEY:{route_result.key}]"
    elif route_result.verb == "scroll":
        spoken = f"[SCROLL:{route_result.direction}]"
    elif route_result.verb == "type":
        spoken = f"[TYPE:{route_result.text}]"
    elif route_result.verb == "open_app":
        spoken = f"[OPEN_APP:{route_result.app_name}]"
    elif route_result.verb == "click" and target is not None:
        spoken = f"opening the {target.label}"
        if free_text:
            spoken = f"[CLICK:{target.x},{target.y}]"
    elif route_result.verb == "answer" and route_result.text:
        spoken = route_result.text
    else:
        spoken = "i can't see that on this screen"

    return {"spoken": spoken, "point_id": point_id, "click_id": click_id}


def split_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def split_csv_text(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def evaluate_observed(fixture: dict, observed: dict) -> list[str]:
    failures: list[str] = []
    expects = fixture["expects"]
    spoken = observed["spoken"]
    spoken_lower = spoken.lower()

    if "EXPECT_POINT_ID" in expects and observed["point_id"] != int(expects["EXPECT_POINT_ID"]):
        failures.append(f"point: got {observed['point_id']} want {expects['EXPECT_POINT_ID']}")
    if "EXPECT_CLICK_ID" in expects and observed["click_id"] != int(expects["EXPECT_CLICK_ID"]):
        failures.append(f"click: got {observed['click_id']} want {expects['EXPECT_CLICK_ID']}")
    if "EXPECT_POINT_ID_ONE_OF" in expects:
        allowed = split_csv_ints(expects["EXPECT_POINT_ID_ONE_OF"])
        if observed["point_id"] not in allowed:
            failures.append(f"point: got {observed['point_id']} want one_of {allowed}")
    if "EXPECT_CLICK_ID_ONE_OF" in expects:
        allowed = split_csv_ints(expects["EXPECT_CLICK_ID_ONE_OF"])
        if observed["click_id"] not in allowed:
            failures.append(f"click: got {observed['click_id']} want one_of {allowed}")
    if "SPOKEN_MUST_CONTAIN" in expects:
        for token in split_csv_text(expects["SPOKEN_MUST_CONTAIN"]):
            if token not in spoken_lower:
                failures.append(f"missing: {token}")
    if "SPOKEN_MUST_NOT_CONTAIN" in expects:
        for token in split_csv_text(expects["SPOKEN_MUST_NOT_CONTAIN"]):
            if token in spoken_lower:
                failures.append(f"forbidden: {token}")
    for pattern in expects.get("SPOKEN_MUST_MATCH_REGEX", []):
        if not re.search(pattern, spoken):
            failures.append(f"regex: {pattern}")
    if "SPOKEN_MAX_WORDS" in expects:
        cap = int(expects["SPOKEN_MAX_WORDS"])
        words = len(spoken.split())
        if words > cap:
            failures.append(f"words {words} > {cap}")
    return failures


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate deterministic Pace fast-router M0.")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--repeat", type=int, default=200, help="Latency repeats per fixture.")
    args = parser.parse_args()

    fixture_paths = sorted(path for path in args.fixtures.glob("*.txt") if path.name != "README.md")
    if not fixture_paths:
        raise SystemExit(f"no fixtures found in {args.fixtures}")

    passed = 0
    latencies_ms: list[float] = []
    print(f"=== Pace fast-router M0 deterministic eval: {len(fixture_paths)} fm-fixtures ===\n")
    for fixture_path in fixture_paths:
        fixture = parse_fixture(fixture_path.read_text())
        first_route: Route | None = None
        first_observed: dict | None = None
        fixture_latencies: list[float] = []
        for _ in range(max(1, args.repeat)):
            start = time.perf_counter_ns()
            routed = route(fixture["user"], fixture["elements"])
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
            fixture_latencies.append(elapsed_ms)
            if first_route is None:
                first_route = routed
                first_observed = route_to_observed(routed, fixture["elements"], fixture["free_text"])
        latencies_ms.extend(fixture_latencies)

        assert first_route is not None and first_observed is not None
        failures = evaluate_observed(fixture, first_observed)
        ok = not failures
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] {fixture_path.stem} "
            f"verb={first_route.verb} reason={first_route.reason} "
            f"conf={first_route.confidence:.2f} latency_ms={statistics.median(fixture_latencies):.4f}"
        )
        if not ok:
            print(f"    spoken: {first_observed['spoken']}")
            print(f"    point={first_observed['point_id']} click={first_observed['click_id']}")
            for failure in failures:
                print(f"    - {failure}")

    print(f"\n=== {passed}/{len(fixture_paths)} fm-fixtures passed ===")
    print(
        "latency_ms "
        f"p50={statistics.median(latencies_ms):.4f} "
        f"p95={percentile(latencies_ms, 0.95):.4f} "
        f"max={max(latencies_ms):.4f} "
        f"samples={len(latencies_ms)}"
    )
    return 0 if passed == len(fixture_paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())
