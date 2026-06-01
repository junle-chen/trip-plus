"""Normalize and score traveler experience simulation outputs."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Dict, Optional

from .experience_trace import (
    EXPERIENCE_DIMENSIONS,
    canonical_experience_dimension_name,
    validate_simulation_against_trace,
)

SCORE_INCONSISTENCY_TOLERANCE = 0.05
def _as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _weighted_average(pairs: list[tuple[float, int]]) -> Optional[float]:
    total_weight = sum(weight for _value, weight in pairs if weight > 0)
    if total_weight <= 0:
        return None
    return round(sum(value * weight for value, weight in pairs if weight > 0) / total_weight, 4)


def _canonicalize_dimension_map(dimensions: Any) -> Dict[str, Any]:
    if not isinstance(dimensions, dict):
        return {}
    canonical: Dict[str, Any] = {}
    for raw_name, detail in dimensions.items():
        canonical_name = canonical_experience_dimension_name(raw_name)
        if canonical_name not in EXPERIENCE_DIMENSIONS:
            continue
        normalized_detail = deepcopy(detail) if isinstance(detail, dict) else {"value": detail}
        normalized_detail.setdefault("dimension", canonical_name)
        if canonical_name not in canonical or str(raw_name) == canonical_name:
            canonical[canonical_name] = normalized_detail
    return canonical


def _canonicalize_simulation_dimensions(payload: Dict[str, Any]) -> None:
    canonical_dimensions = _canonicalize_dimension_map(payload.get("experience_dimensions"))
    if canonical_dimensions:
        payload["experience_dimensions"] = canonical_dimensions

    for item in payload.get("activity_simulations") or []:
        if not isinstance(item, dict):
            continue
        canonical_updates = _canonicalize_dimension_map(item.get("dimension_updates"))
        if canonical_updates:
            item["dimension_updates"] = canonical_updates


def _dimension_score_1_5(detail: Any) -> Optional[float]:
    if not isinstance(detail, dict):
        return None
    value = _as_float(detail.get("score_1_5"))
    if value is None:
        fallback = _as_float(detail.get("score"))
        if fallback is not None and 1.0 <= fallback <= 5.0:
            value = fallback
    if value is None:
        return None
    return round(max(1.0, min(5.0, value)), 4)


def _score_from_1_5(score_1_5: Optional[float]) -> Optional[float]:
    if score_1_5 is None:
        return None
    return round(max(0.0, min(1.0, (score_1_5 - 1.0) / 4.0)), 4)


def _score_1_5_from_score(score: Optional[float]) -> Optional[float]:
    if score is None:
        return None
    return round(1.0 + 4.0 * max(0.0, min(1.0, score)), 4)


def _llm_reported_overall_detail(payload: Dict[str, Any]) -> Dict[str, Any]:
    reported = payload.get("llm_reported_overall")
    if isinstance(reported, dict):
        raw_score_1_5 = _as_float(reported.get("raw_score_1_5", reported.get("score_1_5")))
        raw_score = _as_float(reported.get("raw_score", reported.get("score")))
        reason = reported.get("reason")
    else:
        raw_score_1_5 = _as_float(payload.get("score_1_5"))
        raw_score = _as_float(payload.get("score"))
        reason = None

    clamped_score_1_5 = (
        round(max(1.0, min(5.0, raw_score_1_5)), 4)
        if raw_score_1_5 is not None
        else None
    )
    clamped_score = (
        round(max(0.0, min(1.0, raw_score)), 4)
        if raw_score is not None
        else None
    )

    if clamped_score_1_5 is not None:
        score_1_5 = clamped_score_1_5
        score = _score_from_1_5(score_1_5)
        normalized_from = "score_1_5"
    elif clamped_score is not None:
        score = clamped_score
        score_1_5 = _score_1_5_from_score(score)
        normalized_from = "score"
    else:
        score_1_5 = None
        score = None
        normalized_from = None

    if clamped_score_1_5 is not None and clamped_score is not None:
        expected_score = _score_from_1_5(clamped_score_1_5)
        score_scale_consistent = (
            expected_score is not None
            and abs(clamped_score - expected_score) <= SCORE_INCONSISTENCY_TOLERANCE
        )
    else:
        score_scale_consistent = None

    detail: Dict[str, Any] = {
        "score_1_5": score_1_5,
        "score": score,
        "raw_score_1_5": raw_score_1_5,
        "raw_score": raw_score,
        "score_normalized_from": normalized_from,
        "score_scale_consistent": score_scale_consistent,
        "source": "LLM self-reported overall score for comparison only",
        "authoritative": False,
    }
    if isinstance(reported, dict):
        for key in (
            "aggregation",
            "chunk_score_scale_inconsistent_count",
            "dimension_analysis",
            "evidence",
        ):
            if key in reported:
                detail[key] = reported[key]
    if reason not in (None, ""):
        detail["reason"] = reason
    return detail


def _llm_reported_overall_scores(payload: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    detail = _llm_reported_overall_detail(payload)
    return detail["score_1_5"], detail["score"]


def _recompute_authoritative_scores(payload: Dict[str, Any]) -> None:
    dimensions = payload.get("experience_dimensions")
    llm_overall = _llm_reported_overall_detail(payload)
    original_score_1_5 = llm_overall.get("score_1_5")
    original_score = llm_overall.get("score")
    scores: list[float] = []
    excluded_dimensions: list[str] = []
    missing_dimensions: list[str] = []

    if isinstance(dimensions, dict):
        for name in EXPERIENCE_DIMENSIONS:
            detail = dimensions.get(name)
            if not isinstance(detail, dict):
                missing_dimensions.append(name)
                continue
            if detail.get("applicable", True) is False:
                excluded_dimensions.append(name)
                continue
            value = _dimension_score_1_5(detail)
            if value is None:
                missing_dimensions.append(name)
                continue
            detail["score_1_5"] = value
            detail["score"] = value
            scores.append(value)
    else:
        missing_dimensions.extend(EXPERIENCE_DIMENSIONS)

    if scores:
        recomputed_score_1_5 = round(sum(scores) / len(scores), 4)
        recomputed_score = round(max(0.0, min(1.0, (recomputed_score_1_5 - 1.0) / 4.0)), 4)
    else:
        recomputed_score_1_5 = None
        recomputed_score = None

    score_1_5_inconsistent = (
        original_score_1_5 is not None
        and recomputed_score_1_5 is not None
        and abs(original_score_1_5 - recomputed_score_1_5) > SCORE_INCONSISTENCY_TOLERANCE
    )
    score_inconsistent = (
        original_score is not None
        and recomputed_score is not None
        and abs(original_score - recomputed_score) > SCORE_INCONSISTENCY_TOLERANCE
    )

    payload["llm_reported_overall"] = llm_overall
    payload["score_recalculation"] = {
        "authoritative": True,
        "source": "code mean of applicable experience_dimensions.score_1_5",
        "llm_reported_score_1_5": original_score_1_5,
        "llm_reported_score": original_score,
        "llm_raw_score_1_5": llm_overall.get("raw_score_1_5"),
        "llm_raw_score": llm_overall.get("raw_score"),
        "llm_score_scale_consistent": llm_overall.get("score_scale_consistent"),
        "recomputed_score_1_5": recomputed_score_1_5,
        "recomputed_score": recomputed_score,
        "score_inconsistency": bool(score_1_5_inconsistent or score_inconsistent),
        "excluded_dimensions": excluded_dimensions,
        "missing_dimensions": missing_dimensions,
    }
    payload.pop("score_1_5", None)
    payload.pop("score", None)


def computed_overall_scores(payload: Dict[str, Any]) -> Dict[str, Any]:
    score_info = payload.get("score_recalculation")
    return score_info if isinstance(score_info, dict) else {}


def normalize_user_simulation_output(
    payload: Dict[str, Any],
    *,
    input_activity_count: int,
    expected_activity_refs: Optional[list[str]] = None,
    experience_trace: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    activity_simulations = payload.get("activity_simulations")
    if not isinstance(activity_simulations, list):
        activity_simulations = []
        payload["activity_simulations"] = activity_simulations
    _canonicalize_simulation_dimensions(payload)

    simulated_refs: list[str] = []
    for item in activity_simulations:
        if not isinstance(item, dict):
            continue
        item_ref = item.get("item_ref")
        if not item_ref and item.get("day") not in (None, "") and item.get("activity_index") not in (None, ""):
            item_ref = f"D{item.get('day')}-A{item.get('activity_index')}"
            item["item_ref"] = item_ref
        if item_ref not in (None, ""):
            simulated_refs.append(str(item_ref).strip())

    expected_refs = [str(ref).strip() for ref in (expected_activity_refs or []) if str(ref).strip()]
    simulated_ref_counts = Counter(ref for ref in simulated_refs if ref)
    missing_refs = [ref for ref in expected_refs if ref not in simulated_ref_counts]
    duplicate_refs = sorted(ref for ref, count in simulated_ref_counts.items() if count > 1)
    unexpected_refs = (
        sorted(ref for ref in simulated_ref_counts if ref not in set(expected_refs))
        if expected_refs
        else []
    )
    exact_ref_coverage = (
        not expected_refs
        or (
            not missing_refs
            and not duplicate_refs
            and not unexpected_refs
            and len(simulated_refs) == len(expected_refs)
        )
    )

    activity_count_check = payload.get("activity_count_check")
    if not isinstance(activity_count_check, dict):
        activity_count_check = {}
    activity_count_check.update(
        {
            "input_activity_count": input_activity_count,
            "simulated_activity_count": len(activity_simulations),
            "expected_activity_refs": expected_refs,
            "simulated_activity_refs": simulated_refs,
            "missing_activity_refs": missing_refs,
            "duplicate_activity_refs": duplicate_refs,
            "unexpected_activity_refs": unexpected_refs,
            "all_activities_simulated": len(activity_simulations) == input_activity_count and exact_ref_coverage,
        }
    )
    payload["activity_count_check"] = activity_count_check
    payload["trace_faithfulness_check"] = validate_simulation_against_trace(payload, experience_trace)
    _recompute_authoritative_scores(payload)

    return payload


def assert_simulation_checks_pass(payload: Dict[str, Any]) -> None:
    activity_count_check = payload.get("activity_count_check") or {}
    if not activity_count_check.get("all_activities_simulated"):
        raise ValueError(
            "LLM user simulator did not simulate every plan activity: "
            f"input_activity_count={activity_count_check.get('input_activity_count')}, "
            f"simulated_activity_count={activity_count_check.get('simulated_activity_count')}, "
            f"missing_activity_refs={activity_count_check.get('missing_activity_refs')}, "
            f"duplicate_activity_refs={activity_count_check.get('duplicate_activity_refs')}, "
            f"unexpected_activity_refs={activity_count_check.get('unexpected_activity_refs')}"
        )
    trace_check = payload.get("trace_faithfulness_check") or {}
    if trace_check.get("faithful") is False:
        raise ValueError(
            "LLM user simulator output failed trace-faithfulness checks: "
            f"items_without_evidence={trace_check.get('items_without_evidence')}, "
            f"invalid_dimension_items={trace_check.get('invalid_dimension_items')}, "
            f"ungrounded_evidence_items={trace_check.get('ungrounded_evidence_items')}"
        )
    score_check = payload.get("score_recalculation") or {}
    if score_check.get("missing_dimensions"):
        raise ValueError(
            "LLM user simulator output missed required top-level experience dimensions: "
            f"missing_dimensions={score_check.get('missing_dimensions')}"
        )
    if score_check.get("recomputed_score") is None or score_check.get("recomputed_score_1_5") is None:
        raise ValueError("LLM user simulator output has no code-recomputed final score")
