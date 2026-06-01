"""Chunk and merge long traveler experience simulations."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, Optional

from .experience_trace import EXPERIENCE_DIMENSIONS, canonical_experience_dimension_name
from .scoring import (
    computed_overall_scores,
    _dimension_score_1_5,
    _llm_reported_overall_detail,
    _score_from_1_5,
    _weighted_average,
    normalize_user_simulation_output,
)


def split_activity_refs(activity_refs: list[str], chunk_size: int) -> list[list[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [activity_refs[index : index + chunk_size] for index in range(0, len(activity_refs), chunk_size)]


def filter_experience_trace(experience_trace: Dict[str, Any], activity_refs: list[str]) -> Dict[str, Any]:
    ref_set = set(activity_refs)
    items_by_ref = {
        str(item.get("item_ref")): item
        for item in experience_trace.get("activity_trace", []) or []
        if isinstance(item, dict) and item.get("item_ref") not in (None, "")
    }
    activity_trace = [deepcopy(items_by_ref[ref]) for ref in activity_refs if ref in items_by_ref and ref in ref_set]
    filtered = deepcopy(experience_trace)
    filtered["expected_activity_refs"] = [item["item_ref"] for item in activity_trace]
    filtered["activity_trace"] = activity_trace
    filtered["activity_count"] = len(activity_trace)
    filtered["chunk_trace"] = {
        "is_chunk": True,
        "full_activity_count": experience_trace.get("activity_count"),
    }
    trace_audit = deepcopy(filtered.get("trace_audit") or {})
    trace_audit["chunk_refs_from_full_trace"] = True
    filtered["trace_audit"] = trace_audit
    return filtered


def _activity_ref_indices(activity_ref: str) -> Optional[tuple[int, int]]:
    match = re.fullmatch(r"D(\d+)-A(\d+)", str(activity_ref or "").strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def filter_plan(plan: Dict[str, Any], activity_refs: list[str]) -> Dict[str, Any]:
    by_day: dict[int, set[int]] = {}
    for activity_ref in activity_refs:
        parsed = _activity_ref_indices(activity_ref)
        if not parsed:
            continue
        day_idx, act_idx = parsed
        by_day.setdefault(day_idx, set()).add(act_idx)

    filtered = {
        key: deepcopy(value)
        for key, value in plan.items()
        if key != "daily_plans"
    }
    daily_plans = []
    for day_idx, day in enumerate(plan.get("daily_plans", []) or [], start=1):
        if not isinstance(day, dict) or day_idx not in by_day:
            continue
        filtered_day = {
            key: deepcopy(value)
            for key, value in day.items()
            if key != "activities"
        }
        activities = []
        for act_idx, activity in enumerate(day.get("activities") or [], start=1):
            if act_idx not in by_day[day_idx]:
                continue
            copied_activity = deepcopy(activity)
            copied_activity["original_item_ref"] = f"D{day_idx}-A{act_idx}"
            activities.append(copied_activity)
        if activities:
            filtered_day["activities"] = activities
            daily_plans.append(filtered_day)
    filtered["daily_plans"] = daily_plans
    filtered["plan_slice"] = {
        "is_slice": True,
        "source": "full plan filtered by original item_ref",
        "activity_refs": activity_refs,
    }
    return filtered


def merge_chunk_payloads(
    *,
    chunk_payloads: list[Dict[str, Any]],
    activity_simulations: list[Dict[str, Any]],
    expected_activity_refs: list[str],
    experience_trace: Dict[str, Any],
    model: str,
    turn_id: Optional[int],
    chunk_size: int,
) -> Dict[str, Any]:
    chunk_weights = [
        len(payload.get("chunk_activity_refs") or [])
        for payload in chunk_payloads
    ]
    llm_score_1_5_pairs = []
    llm_score_pairs = []
    llm_scale_inconsistent_count = 0
    dimension_scores: dict[str, list[tuple[float, int]]] = {}
    dimension_states: dict[str, list[str]] = {}
    missing_evidence = []
    audit_notes = []
    chunk_summaries = []
    for payload, weight in zip(chunk_payloads, chunk_weights):
        llm_overall = _llm_reported_overall_detail(payload)
        llm_score_1_5 = llm_overall.get("score_1_5")
        llm_score = llm_overall.get("score")
        if llm_score_1_5 is not None:
            llm_score_1_5_pairs.append((llm_score_1_5, weight))
        if llm_score is not None:
            llm_score_pairs.append((llm_score, weight))
        if llm_overall.get("score_scale_consistent") is False:
            llm_scale_inconsistent_count += 1
        dimensions = payload.get("experience_dimensions")
        if isinstance(dimensions, dict):
            for raw_name, detail in dimensions.items():
                if not isinstance(detail, dict):
                    continue
                name = canonical_experience_dimension_name(raw_name)
                if name not in EXPERIENCE_DIMENSIONS:
                    continue
                if detail.get("applicable", True) is False:
                    dimension_states.setdefault(name, []).append("not_applicable")
                    continue
                value = _dimension_score_1_5(detail)
                if value is not None:
                    dimension_scores.setdefault(name, []).append((value, weight))
                    dimension_states.setdefault(name, []).append("scored")
                else:
                    dimension_states.setdefault(name, []).append("missing")
        if isinstance(payload.get("missing_evidence"), list):
            missing_evidence.extend(payload["missing_evidence"])
        if payload.get("audit_notes") not in (None, ""):
            audit_notes.append(payload["audit_notes"])
        computed_scores = computed_overall_scores(payload)
        chunk_summaries.append(
            {
                "chunk_index": payload.get("chunk_index"),
                "activity_refs": payload.get("chunk_activity_refs") or [],
                "llm_reported_score": llm_score,
                "llm_reported_score_1_5": llm_score_1_5,
                "llm_raw_score": llm_overall.get("raw_score"),
                "llm_raw_score_1_5": llm_overall.get("raw_score_1_5"),
                "llm_score_scale_consistent": llm_overall.get("score_scale_consistent"),
                "score": computed_scores.get("recomputed_score"),
                "score_1_5": computed_scores.get("recomputed_score_1_5"),
                "trace_faithful": (payload.get("trace_faithfulness_check") or {}).get("faithful"),
            }
        )

    merged_llm_score_1_5 = _weighted_average(llm_score_1_5_pairs)
    merged_llm_score = (
        _score_from_1_5(merged_llm_score_1_5)
        if merged_llm_score_1_5 is not None
        else _weighted_average(llm_score_pairs)
    )
    merged = {
        "llm_reported_overall": {
            "score_1_5": merged_llm_score_1_5,
            "score": merged_llm_score,
            "aggregation": "activity-count-weighted average of chunk-level LLM self-reported overall scores",
            "score_normalized_from": "score_1_5" if merged_llm_score_1_5 is not None else "score",
            "chunk_score_scale_inconsistent_count": llm_scale_inconsistent_count,
            "authoritative": False,
        },
        "profile_summary": (chunk_payloads[0].get("profile_summary") if chunk_payloads else None),
        "activity_simulations": activity_simulations,
        "experience_dimensions": {
            name: {
                "score_1_5": _weighted_average(dimension_scores.get(name, [])),
                "score": _weighted_average(dimension_scores.get(name, [])),
                "applicable": bool(dimension_scores.get(name)) or not dimension_states.get(name)
                or any(state == "missing" for state in dimension_states.get(name, [])),
                "aggregation": "activity-count-weighted average across chunks",
                **(
                    {"not_applicable_reason": "not applicable in all chunks"}
                    if not dimension_scores.get(name)
                    and dimension_states.get(name)
                    and all(state == "not_applicable" for state in dimension_states[name])
                    else {}
                ),
            }
            for name in EXPERIENCE_DIMENSIONS
        },
        "trace_faithfulness": {
            "chunked": True,
            "chunk_count": len(chunk_payloads),
            "all_chunks_trace_faithful": all(
                (payload.get("trace_faithfulness_check") or {}).get("faithful")
                for payload in chunk_payloads
            ),
        },
        "missing_evidence": missing_evidence,
        "audit_notes": {
            "strategy": "chunked LLM simulation; final artifact revalidated against full experience trace",
            "chunk_notes": audit_notes,
        },
        "chunking": {
            "enabled": True,
            "chunk_size": chunk_size,
            "chunk_count": len(chunk_payloads),
            "chunks": chunk_summaries,
        },
        "input_experience_trace": experience_trace,
        "simulator_model": model,
        "turn_id": turn_id,
    }
    return normalize_user_simulation_output(
        merged,
        input_activity_count=len(expected_activity_refs),
        expected_activity_refs=expected_activity_refs,
        experience_trace=experience_trace,
    )
