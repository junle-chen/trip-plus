"""Deterministic experience trace for profile-conditioned user simulation.

The experience trace is the reliability layer for the LLM user simulator. It converts a
structured itinerary into one auditable event per activity and attaches only
evidence that can be traced back to the plan, profile, turn state, or evaluation
context. The LLM should synthesize lived experience from this trace, not invent
new facts.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, Optional

from evaluation.utils import extract_entity_name, slot_to_minutes


EXPERIENCE_DIMENSIONS = (
    "physical_comfort",
    "schedule_comfort",
    "environmental_comfort",
    "budget_comfort",
    "preference_satisfaction",
)

RUBRIC = {
    "scale": "1-5, higher is better",
    "anchors": {
        "5": "very good: strongly matched to the user with low burden",
        "4": "good: mostly comfortable with only minor issues",
        "3": "medium: acceptable, but with visible tradeoffs",
        "2": "poor: executable but clearly uncomfortable or stressful",
        "1": "very poor: severe discomfort, stress, or profile conflict",
    },
    "dimension_groups": {
        "physical": ["physical_comfort", "environmental_comfort"],
        "psychological": ["schedule_comfort", "budget_comfort", "preference_satisfaction"],
    },
}

OUTDOOR_MARKERS = (
    "mountain",
    "park",
    "garden",
    "forest",
    "lake",
    "river",
    "beach",
    "coast",
    "trail",
    "hike",
)

REST_MARKERS = ("rest", "break", "recovery", "free time")
FOOD_MARKERS = ("food", "cuisine", "local_food", "snack")
SCENERY_MARKERS = ("nature", "scenery", "photography", "landmark")
COMFORT_MARKERS = ("comfort", "relaxed", "luxury")

BUDGET_MARKERS = (
    "budget",
    "budget_sensitive",
    "budget_tight",
    "meal_avoid_expensive",
    "hotel_budget_first",
)

FATIGUE_MARKERS = ("avoid_long_walk", "knee_issue", "low_endurance", "elder")
HEAT_COLD_MARKERS = ("heat_sensitive", "cold_sensitive", "extreme_weather")
SCHEDULE_MARKERS = (
    "schedule_pacing",
    "avoid_dense_schedule",
    "avoid_early_departure",
    "avoid_late_arrival",
    "avoid_red_eye",
    "late_start",
    "midday_rest",
)


def build_experience_trace(
    *,
    query_record: Dict[str, Any],
    plan: Dict[str, Any],
    turn_id: Optional[int] = None,
    evaluation_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one deterministic evidence item per itinerary activity."""

    context = evaluation_context or {}
    meta = _meta_from_query_record(query_record)
    active_state = _active_turn_state(query_record, turn_id)
    user_model = _build_user_model(meta, active_state)
    environment = _environment_summary(meta, context)
    budget = _budget_summary(plan, meta, user_model)

    events = []
    missing_evidence = []
    daily_plans = plan.get("daily_plans") or []
    for day_idx, day in enumerate(daily_plans, start=1):
        if not isinstance(day, dict):
            continue
        for act_idx, activity in enumerate(day.get("activities") or [], start=1):
            event = _activity_event(day_idx, act_idx, activity)
            if event["duration_minutes"] is None:
                missing_evidence.append({
                    "item_ref": event["item_ref"],
                    "field": "duration_minutes",
                    "reason": "time_slot is missing or unparsable",
                })
            events.append(_activity_trace_item(event, activity, user_model, environment, budget))

    expected_refs = [item["item_ref"] for item in events]
    return {
        "method": "Traveler Experience Trace",
        "rubric": RUBRIC,
        "turn_context": {
            "current_turn_state_available": bool(active_state),
            "turn_selector_used": turn_id is not None,
        },
        "profile_source": user_model["profile_source"],
        "user_model": user_model,
        "environment": environment,
        "budget": budget,
        "expected_activity_refs": expected_refs,
        "activity_count": len(events),
        "activity_trace": events,
        "trace_audit": {
            "all_plan_activities_traced": len(expected_refs) == count_plan_activities(plan),
            "missing_evidence": missing_evidence,
            "allowed_claim_sources": [
                "plan.daily_plans[].activities[]",
                "query_record.meta_info.observable_profile",
                "query_record.meta_info.user_profile",
                "query_record.turns[].oracle_state_after_turn",
            ],
        },
    }


def validate_simulation_against_trace(
    simulation: Dict[str, Any],
    experience_trace: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return fail-closed audit checks for an LLM simulation artifact."""

    if not experience_trace:
        return {
            "status": "skipped",
            "reason": "experience_trace not provided",
            "faithful": False,
        }

    expected_refs = [
        str(ref).strip()
        for ref in experience_trace.get("expected_activity_refs", []) or []
        if str(ref).strip()
    ]
    expected_ref_set = set(expected_refs)
    items = simulation.get("activity_simulations")
    if not isinstance(items, list):
        items = []

    simulated_refs = []
    items_without_evidence = []
    invalid_dimension_items = []
    ungrounded_evidence_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_ref = str(item.get("item_ref") or "").strip()
        if item_ref:
            simulated_refs.append(item_ref)
        evidence = item.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            items_without_evidence.append(item_ref or "<missing item_ref>")
        if not _dimension_updates_valid(item.get("dimension_updates")):
            invalid_dimension_items.append(item_ref or "<missing item_ref>")
        if not _evidence_grounded(evidence, expected_ref_set, item_ref):
            ungrounded_evidence_items.append(item_ref or "<missing item_ref>")

    missing_refs = [ref for ref in expected_refs if ref not in simulated_refs]
    unexpected_refs = sorted(ref for ref in set(simulated_refs) if ref not in expected_ref_set)
    duplicate_refs = sorted(ref for ref in set(simulated_refs) if simulated_refs.count(ref) > 1)
    faithful = not (
        missing_refs
        or unexpected_refs
        or duplicate_refs
        or items_without_evidence
        or invalid_dimension_items
        or ungrounded_evidence_items
    )
    return {
        "status": "ok",
        "faithful": faithful,
        "expected_activity_refs": expected_refs,
        "simulated_activity_refs": simulated_refs,
        "missing_activity_refs": missing_refs,
        "unexpected_activity_refs": unexpected_refs,
        "duplicate_activity_refs": duplicate_refs,
        "items_without_evidence": items_without_evidence,
        "invalid_dimension_items": invalid_dimension_items,
        "ungrounded_evidence_items": ungrounded_evidence_items,
    }


def count_plan_activities(plan: Dict[str, Any]) -> int:
    return sum(
        len(day.get("activities") or [])
        for day in plan.get("daily_plans", []) or []
        if isinstance(day, dict)
    )


def plan_activity_refs(plan: Dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for day_idx, day in enumerate(plan.get("daily_plans", []) or [], start=1):
        if not isinstance(day, dict):
            continue
        for act_idx, _activity in enumerate(day.get("activities") or [], start=1):
            refs.append(f"D{day_idx}-A{act_idx}")
    return refs


def _meta_from_query_record(query_record: Dict[str, Any]) -> Dict[str, Any]:
    meta = query_record.get("meta_info") if isinstance(query_record, dict) else None
    if not isinstance(meta, dict):
        return {}
    base_meta = meta.get("base_query_meta")
    if not isinstance(base_meta, dict):
        return meta
    merged = dict(base_meta)
    merged.update(meta)
    return merged


def _active_turn_state(query_record: Dict[str, Any], turn_id: Optional[int]) -> Dict[str, Any]:
    if turn_id is None or not isinstance(query_record, dict):
        return {}
    turns = query_record.get("turns") or []
    if not isinstance(turns, list):
        return {}
    for idx, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        raw_turn_id = turn.get("turn_id", idx)
        try:
            matches = int(str(raw_turn_id).strip()) == int(turn_id)
        except (TypeError, ValueError):
            matches = idx == turn_id
        if matches and isinstance(turn.get("oracle_state_after_turn"), dict):
            return turn["oracle_state_after_turn"]
    return {}


def _build_user_model(meta: Dict[str, Any], active_state: Dict[str, Any]) -> Dict[str, Any]:
    observable = meta.get("observable_profile") or {}
    hidden = meta.get("user_profile") or {}
    signals = _collect_strings(observable) + _collect_strings(hidden) + _collect_strings(active_state)
    signal_text = " ".join(signals).lower()

    party = _party_summary(observable, hidden, active_state)
    sensitivities = {
        "fatigue": _level(signal_text, FATIGUE_MARKERS, party.get("has_elder")),
        "environment": _level(signal_text, HEAT_COLD_MARKERS, False),
        "schedule_stress": _level(signal_text, SCHEDULE_MARKERS, party.get("has_child")),
        "budget_stress": _level(signal_text, BUDGET_MARKERS, False),
        "preference_reward": "high" if any(
            token in signal_text for token in FOOD_MARKERS + SCENERY_MARKERS + COMFORT_MARKERS
        ) else "medium",
    }

    return {
        "profile_source": {
            "used_observable_profile": bool(observable),
            "used_user_profile": bool(hidden),
            "used_oracle_state_after_turn": bool(active_state),
        },
        "party": party,
        "sensitivities": sensitivities,
        "positive_preferences": _positive_preferences(signal_text),
        "negative_preferences": {
            "fatigue_sensitive": sensitivities["fatigue"] == "high",
            "weather_sensitive": sensitivities["environment"] == "high",
            "schedule_sensitive": sensitivities["schedule_stress"] == "high",
            "budget_sensitive": sensitivities["budget_stress"] == "high",
        },
        "raw_signal_excerpt": signals[:24],
    }


def _activity_trace_item(
    event: Dict[str, Any],
    activity: Dict[str, Any],
    user_model: Dict[str, Any],
    environment: Dict[str, Any],
    budget: Dict[str, Any],
) -> Dict[str, Any]:
    text = _activity_text(activity)
    return {
        "item_ref": event["item_ref"],
        "event": event,
        "experience_facts": _experience_facts(event, text, user_model, environment, budget),
        "llm_scope": {
            "may_infer": "subjective comfort scores from provided neutral facts only",
            "must_not_infer": [
                "unmentioned rest",
                "unmentioned weather/crowd/closure/delay",
                "restaurant quality unless supported",
                "new hidden preferences",
            ],
        },
    }


def _experience_facts(
    event: Dict[str, Any],
    text: str,
    user_model: Dict[str, Any],
    environment: Dict[str, Any],
    budget: Dict[str, Any],
) -> Dict[str, Any]:
    """Return neutral, non-scored facts for LLM subjective synthesis."""

    duration = event.get("duration_minutes")
    act_type = str(event.get("type") or "")
    return {
        key: value
        for key, value in {
            "activity_type": act_type,
            "duration_minutes": duration,
            "duration_bucket": _duration_bucket(duration),
            "experience_flags": _experience_flags(event, text),
            "cost": event.get("cost"),
            "budget_cost_relevance": _budget_cost_relevance(event, budget),
        }.items()
        if value not in (None, "", [], {})
    }


def _experience_flags(event: Dict[str, Any], text: str) -> list[str]:
    flags = []
    duration = event.get("duration_minutes")
    act_type = str(event.get("type") or "")
    if act_type in {"travel_city", "travel_intercity_public"}:
        flags.append("transport")
    if act_type == "travel_intercity_public":
        flags.append("intercity_transport")
    if act_type == "travel_city":
        flags.append("local_transport")
    if act_type == "attraction":
        flags.append("attraction")
        if _contains_any(text, OUTDOOR_MARKERS):
            flags.append("outdoor_marker_present")
        if duration is not None and duration >= 150:
            flags.append("long_attraction_block")
    if act_type == "meal":
        flags.append("meal")
    if act_type in {"hotel", "buffer"} and _contains_any(text, REST_MARKERS):
        flags.append("explicit_rest_or_recovery")

    start = _start_hour(event)
    if start is not None:
        if start < 7:
            flags.append("starts_before_7am")
        elif start < 8:
            flags.append("starts_before_8am")

    end = _end_hour(event)
    if end is not None:
        if end > 22:
            flags.append("ends_after_22")
        elif end > 21:
            flags.append("ends_after_21")
    return flags


def _activity_event(day_idx: int, act_idx: int, activity: Dict[str, Any]) -> Dict[str, Any]:
    details = activity.get("details") or {}
    start_min, end_min = slot_to_minutes(activity.get("time_slot"))
    duration = max(0, end_min - start_min) if start_min is not None and end_min is not None else None
    return {
        "item_ref": f"D{day_idx}-A{act_idx}",
        "day": day_idx,
        "activity_index": act_idx,
        "type": str(activity.get("type") or "other"),
        "name": _activity_name(activity),
        "time_slot": activity.get("time_slot"),
        "duration_minutes": duration,
        "mode": _first_present(details, ("mode", "transport_mode", "recommended_mode")),
        "cost": _numeric_value(_first_present(details, ("cost", "price", "ticket_price", "estimated_cost"))),
        "source": f"plan.daily_plans[{day_idx - 1}].activities[{act_idx - 1}]",
    }


def _environment_summary(meta: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    values = []
    for source in (
        meta.get("city_context") or {},
        meta.get("environment_reference") or {},
        context.get("city_context") or {},
        context.get("environment_reference") or {},
    ):
        if not isinstance(source, dict):
            continue
        values.extend(_collect_strings(source.get("city_tags")))
        values.extend(_collect_strings(source.get("seasonal_advisories")))
        values.extend(_collect_strings(source.get("environment_tags")))
    joined = " ".join(values).lower()
    risk_tags = []
    for marker in ("heat", "hot", "cold", "rain", "typhoon", "storm", "high_altitude", "extreme"):
        if marker in joined:
            risk_tags.append(marker)
    return {"risk_tags": sorted(set(risk_tags)), "raw_context_excerpt": values[:16]}


def _budget_summary(plan: Dict[str, Any], meta: Dict[str, Any], user_model: Dict[str, Any]) -> Dict[str, Any]:
    budget_limit = _find_budget_limit(meta)
    estimated_total = _numeric_value((plan.get("budget_summary") or {}).get("total_estimated_budget"))
    budget_sensitive = user_model["sensitivities"]["budget_stress"] == "high"
    applicable = budget_limit is not None or budget_sensitive
    reason = "explicit budget constraint" if budget_limit is not None else "budget-sensitive profile signal"
    margin_level = "unknown"
    if budget_limit and estimated_total is not None:
        ratio = estimated_total / budget_limit
        if ratio > 1.0:
            margin_level = "over_limit"
        elif ratio >= 0.85:
            margin_level = "near_limit"
        else:
            margin_level = "comfortable"
    return {
        "applicable": applicable,
        "reason": reason if applicable else "no budget constraint or budget-sensitive profile signal",
        "budget_limit": budget_limit,
        "estimated_total": estimated_total,
        "remaining_margin_level": margin_level,
    }


def _budget_cost_relevance(event: Dict[str, Any], budget: Dict[str, Any]) -> str:
    cost = event.get("cost")
    if not isinstance(cost, (int, float)) or cost <= 0:
        return "none"

    ratios = []
    for key in ("budget_limit", "estimated_total"):
        denominator = budget.get(key)
        if isinstance(denominator, (int, float)) and denominator > 0:
            ratios.append(float(cost) / float(denominator))
    max_ratio = max(ratios) if ratios else None
    if max_ratio is not None:
        if max_ratio >= 0.10:
            return "high"
        if max_ratio >= 0.03:
            return "moderate"

    if cost >= 1000:
        return "high"
    if cost >= 300:
        return "moderate"
    return "minor"


def _find_budget_limit(meta: Dict[str, Any]) -> Optional[float]:
    constraints = meta.get("hard_constraints") or {}
    if not isinstance(constraints, dict):
        return None
    for value in constraints.values():
        if not isinstance(value, dict):
            continue
        for key in ("max_budget", "budget", "total_budget"):
            parsed = _numeric_value(value.get(key))
            if parsed is not None:
                return parsed
    return None


def _duration_bucket(duration: Any) -> str:
    if not isinstance(duration, (int, float)):
        return "unknown"
    if duration >= 240:
        return "very_long"
    if duration >= 90:
        return "long"
    if duration >= 45:
        return "moderate"
    if duration >= 15:
        return "short"
    return "brief"


def _party_summary(*sources: Dict[str, Any]) -> Dict[str, Any]:
    text = " ".join(_collect_strings(list(sources))).lower()
    return {
        "has_child": any(token in text for token in ("child", "children", "kid", "stroller", "family")),
        "has_elder": any(token in text for token in ("elder", "older", "senior", "parent")),
        "raw_evidence": _collect_strings(list(sources))[:12],
    }


def _positive_preferences(signal_text: str) -> Dict[str, bool]:
    return {
        "food": any(marker in signal_text for marker in FOOD_MARKERS),
        "scenery": any(marker in signal_text for marker in SCENERY_MARKERS),
        "comfort": any(marker in signal_text for marker in COMFORT_MARKERS),
    }


def _level(signal_text: str, markers: Iterable[str], party_boost: Any) -> str:
    if party_boost or any(marker.lower() in signal_text for marker in markers):
        return "high"
    return "medium"


def _activity_name(activity: Dict[str, Any]) -> str:
    return extract_entity_name(activity)


def _activity_text(activity: Dict[str, Any]) -> str:
    return json.dumps(activity, ensure_ascii=False, sort_keys=True).lower()


def _contains_any(text: str, markers: Iterable[str]) -> bool:
    return any(marker.lower() in text for marker in markers)


def _first_present(payload: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def _numeric_value(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if value in (None, ""):
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _start_hour(event: Dict[str, Any]) -> Optional[float]:
    start_min, _end_min = slot_to_minutes(event.get("time_slot"))
    return start_min / 60.0 if start_min is not None else None


def _end_hour(event: Dict[str, Any]) -> Optional[float]:
    _start_min, end_min = slot_to_minutes(event.get("time_slot"))
    return end_min / 60.0 if end_min is not None else None


def _collect_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        strings = []
        for key, item in value.items():
            strings.append(str(key))
            strings.extend(_collect_strings(item))
        return strings
    if isinstance(value, (list, tuple, set)):
        strings = []
        for item in value:
            strings.extend(_collect_strings(item))
        return strings
    if value in (None, ""):
        return []
    return [str(value)]


def _dimension_updates_valid(updates: Any) -> bool:
    if not isinstance(updates, dict) or not updates:
        return False
    canonical_keys = {canonical_experience_dimension_name(key) for key in updates}
    if not set(EXPERIENCE_DIMENSIONS).issubset(canonical_keys):
        return False
    for key in EXPERIENCE_DIMENSIONS:
        matching_values = [
            value
            for raw_key, value in updates.items()
            if canonical_experience_dimension_name(raw_key) == key
        ]
        if not matching_values or not any(isinstance(value, dict) for value in matching_values):
            return False
    return True


def canonical_experience_dimension_name(name: Any) -> str:
    return str(name or "").strip()


def _evidence_grounded(evidence: Any, expected_refs: set[str], item_ref: str) -> bool:
    if not isinstance(evidence, list) or not evidence:
        return False
    allowed_global_sources = {
        "user_model",
        "environment",
        "budget",
        "EXPERIENCE_TRACE.user_model",
        "EXPERIENCE_TRACE.environment",
        "EXPERIENCE_TRACE.budget",
    }
    for entry in evidence:
        if isinstance(entry, str):
            continue
        if not isinstance(entry, dict):
            continue
        ref = str(entry.get("item_ref") or entry.get("source_item_ref") or item_ref or "").strip()
        source = str(entry.get("source") or entry.get("plan_source") or "").strip()
        if ref not in expected_refs:
            continue
        if (
            source.startswith("plan.daily_plans")
            or source.startswith("PLAN.daily_plans")
            or source.startswith("EXPERIENCE_TRACE.activity_trace")
            or source in allowed_global_sources
        ):
            return True
    return False
