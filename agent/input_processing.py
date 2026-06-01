"""Convert benchmark records into agent-ready inputs.

This module owns the boundary between released query JSON and the runtime
agent. It handles both single-turn and multi-turn records, plus the sample
metadata used by prompts and sample-database lookup.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Tuple


_TEST_DATA_INDEX_CACHE: Dict[Tuple[str, int, int], Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]] = {}
_TEST_DATA_INDEX_CACHE_LOCK = Lock()


def strip_id_prefix_value(sample_id: object) -> str:
    text = str(sample_id or "").strip()
    return text[3:] if text.startswith("id_") else text


def load_test_data_index(test_data_path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    stat = test_data_path.stat()
    key = (str(test_data_path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    with _TEST_DATA_INDEX_CACHE_LOCK:
        cached = _TEST_DATA_INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    samples = json.loads(test_data_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError(f"Test data must be a JSON list: {test_data_path}")

    by_id: Dict[str, Dict[str, Any]] = {}
    by_base_id: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        sample_id = strip_id_prefix_value(sample.get("id", ""))
        if sample_id:
            by_id[sample_id] = sample
        base_query_id = strip_id_prefix_value(sample.get("base_query_id", ""))
        if base_query_id:
            by_base_id[base_query_id] = sample

    loaded = (by_id, by_base_id)
    with _TEST_DATA_INDEX_CACHE_LOCK:
        stale_keys = [old_key for old_key in _TEST_DATA_INDEX_CACHE if old_key[0] == key[0] and old_key != key]
        for old_key in stale_keys:
            _TEST_DATA_INDEX_CACHE.pop(old_key, None)
        _TEST_DATA_INDEX_CACHE[key] = loaded
    return loaded


def planner_meta_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    meta = sample.get("meta_info") or {}
    if not isinstance(meta, dict):
        return {}
    base_meta = meta.get("base_query_meta")
    if isinstance(base_meta, dict):
        merged = dict(base_meta)
        merged["multiturn_meta"] = meta
        return merged
    return meta


def load_sample_meta(sample_id: object, test_data_path: Path | None) -> Dict[str, Any]:
    if sample_id is None or test_data_path is None or not test_data_path.exists():
        return {}
    normalized_id = strip_id_prefix_value(sample_id)
    by_id, by_base_id = load_test_data_index(test_data_path)
    sample = by_id.get(normalized_id) or by_base_id.get(normalized_id)
    if isinstance(sample, dict):
        return planner_meta_from_sample(sample)
    return {}


def is_multiturn_sample(sample: Dict[str, Any]) -> bool:
    turns = sample.get("turns")
    return isinstance(turns, list) and any(isinstance(turn, dict) for turn in turns)


def turn_utterance(sample: Dict[str, Any], turn: Dict[str, Any], turn_index: int) -> str:
    utterance = str(turn.get("utterance") or "").strip()
    if utterance:
        return utterance
    if turn_index == 0:
        return str(sample.get("base_query") or sample.get("query") or "").strip()
    return ""


def build_multiturn_turn_query(sample: Dict[str, Any], turn_index: int, language: str) -> str:
    turns = [turn for turn in sample.get("turns", []) if isinstance(turn, dict)]
    visible_turns = turns[: turn_index + 1]

    lines = [
        "This is a multi-turn travel-planning interaction from the same user. Plan only from user-visible messages so far, the visible profile, and tool evidence.",
        "The newest feedback has priority, while earlier still-valid hard constraints and preferences must be preserved; by default, return a complete updated itinerary, not only a diff.",
        "Return `<clarification>` only when the latest request lacks a necessary anchor, requires the user to choose between conflicting hard constraints, or conflicts with hard observable-profile facts.",
        "Return `<no_solution>` only when the user explicitly authorized impossibility judgment and tool evidence proves active hard constraints cannot be jointly satisfied.",
        "Entity existence, opening hours, prices, distances, and routes are tool-verification responsibilities; do not ask the user to confirm these before querying tools.",
        "",
        "User messages up to the current turn:",
    ]
    for idx, turn in enumerate(visible_turns):
        turn_id = turn.get("turn_id", idx)
        utterance = turn_utterance(sample, turn, idx)
        if utterance:
            lines.append(f"[Turn {turn_id}] {utterance}")
    lines.extend(
        [
            "",
            "Re-plan from all messages above and output exactly one final mode: `<plan>`, `<clarification>`, or `<no_solution>`.",
        ]
    )
    return "\n".join(lines)


def build_multiturn_chat_user_message(sample: Dict[str, Any], turn_index: int, language: str) -> str:
    turns = [turn for turn in sample.get("turns", []) if isinstance(turn, dict)]
    if turn_index >= len(turns):
        return ""
    return turn_utterance(sample, turns[turn_index], turn_index)
