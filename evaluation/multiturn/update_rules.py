"""Concrete must_update checkers for multi-turn fulfillment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..soft import evaluate_soft_checks
from ..utils import extract_entity_name, normalize_entity_name, slot_to_minutes
from .response_mode import _status_is_clarification, _status_is_unsat


def _extract_plan_attractions(plan: Dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for day in plan.get("daily_plans", []) or []:
        for act in day.get("activities", []) or []:
            if act.get("type") != "attraction":
                continue
            name = extract_entity_name(act, "attraction")
            if name:
                names.add(normalize_entity_name(name))
    return names


def _profile_rule_ids_for_turn(ground_truth: Dict[str, Any]) -> List[str]:
    rule_ids: List[str] = []
    oracle = ground_truth.get("verification_oracle") or {}
    if isinstance(oracle, dict):
        rule_ids.extend(
            str(item)
            for item in oracle.get("profile_rule_ids") or []
            if str(item).strip()
        )
    state = ground_truth.get("oracle_state_after_turn") or {}
    if isinstance(state, dict):
        for bucket in ("active_profile_deltas", "active_user_state_deltas"):
            for delta in state.get(bucket) or []:
                if not isinstance(delta, dict):
                    continue
                pref = delta.get("profile_preference") or {}
                if isinstance(pref, dict):
                    rule_ids.extend(
                        str(item)
                        for item in pref.get("rule_ids") or []
                        if str(item).strip()
                    )
    return list(dict.fromkeys(rule_ids))


def _family_rule_ids(meta: Dict[str, Any], family: str) -> List[str]:
    all_rule_ids = [
        str(rule_id)
        for rule_id in ((meta.get("user_profile") or {}).get("rule_ids") or [])
        if str(rule_id).strip()
    ]
    if family == "transport":
        return [rule_id for rule_id in all_rule_ids if rule_id.startswith("transport_")]
    if family == "hotel":
        return [rule_id for rule_id in all_rule_ids if rule_id == "hotel_value_first"]
    if family == "restaurant":
        return [
            rule_id
            for rule_id in all_rule_ids
            if rule_id
            in {
                "interest_local_food",
                "meal_avoid_expensive",
                "budget_guarded",
                "budget_tight_cap",
            }
        ]
    if family == "route":
        return [
            rule_id
            for rule_id in all_rule_ids
            if rule_id in {"mobility_accessibility", "schedule_pacing"}
        ]
    return all_rule_ids


def _soft_check_result(
    name: str,
    plan: Dict[str, Any],
    meta: Dict[str, Any],
    rule_ids: Optional[List[str]],
    *,
    threshold: float = 0.75,
    database_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    soft = evaluate_soft_checks(
        plan, meta, rule_ids=rule_ids or None, database_dir=database_dir
    )
    if not soft.get("evaluated_count"):
        return {
            "name": name,
            "passed": None,
            "message": "No deterministic soft check is active for this update",
            "details": soft,
        }
    score = float(soft["score"])
    passed = score >= threshold
    return {
        "name": name,
        "passed": passed,
        "message": None
        if passed
        else f"Soft preference score {score:.2f} < {threshold:.2f}",
        "details": soft,
    }


def _plan_text(plan: Dict[str, Any]) -> str:
    return json.dumps(plan, ensure_ascii=False, sort_keys=True)


def _activities_for_day(
    plan: Dict[str, Any], day_number: object
) -> List[Dict[str, Any]]:
    try:
        idx = int(day_number) - 1
    except (TypeError, ValueError):
        return []
    daily = plan.get("daily_plans", []) or []
    if idx < 0 or idx >= len(daily):
        return []
    return daily[idx].get("activities", []) or []


def _activity_name(activity: Dict[str, Any]) -> str:
    return extract_entity_name(activity)


def _activity_text(activity: Dict[str, Any]) -> str:
    return json.dumps(activity, ensure_ascii=False, sort_keys=True)


def _activity_duration(activity: Dict[str, Any]) -> int:
    start, end = slot_to_minutes(activity.get("time_slot"))
    if start is None or end is None:
        return 0
    return max(0, end - start)


def _overlap_minutes(activity: Dict[str, Any], window: List[int]) -> int:
    start, end = slot_to_minutes(activity.get("time_slot"))
    if start is None or end is None or len(window) != 2:
        return 0
    return max(0, min(end, int(window[1])) - max(start, int(window[0])))


def _looks_outdoor_attraction(activity: Dict[str, Any]) -> bool:
    if activity.get("type") != "attraction":
        return False
    name = _activity_name(activity)
    text = _activity_text(activity)
    text_lower = text.lower()
    name_lower = name.lower()
    duration = _activity_duration(activity)
    indoor_markers = (
        "museum",
        "memorial",
        "gallery",
        "exhibition",
        "exhibit",
        "science",
        "library",
        "theater",
        "theatre",
        "indoor",
        "mall",
        "shopping center",
        "shopping centre",
        "aquarium",
        "art center",
        "art centre",
    )
    if any(marker in text_lower for marker in indoor_markers):
        return False
    outdoor_markers = (
        "mountain",
        "canyon",
        "lake",
        "bay",
        "island",
        "park",
        "forest",
        "grassland",
        "wetland",
        "beach",
        "trail",
        "boardwalk",
        "temple",
        "square",
        "viewpoint",
        "scenic",
        "garden",
        "ancient town",
    )
    return duration >= 150 or any(marker in name_lower for marker in outdoor_markers)


def _has_buffer_or_indoor_on_day(
    plan: Dict[str, Any], adjustment: Dict[str, Any]
) -> bool:
    min_buffer = int(adjustment.get("min_buffer_minutes") or 30)
    for activity in _activities_for_day(plan, adjustment.get("day")):
        duration = _activity_duration(activity)
        if activity.get("type") == "buffer" and duration >= min_buffer:
            return True
        if activity.get("type") in {"hotel", "meal"} and duration >= min_buffer:
            return True
        if activity.get("type") == "attraction" and not _looks_outdoor_attraction(
            activity
        ):
            return True
    return False


def _has_hotel_or_rest_in_window(
    plan: Dict[str, Any], adjustment: Dict[str, Any]
) -> bool:
    min_rest = int(adjustment.get("min_rest_minutes") or 60)
    window = adjustment.get("time_window") or []
    rest_markers = (
        "rest",
        "hotel",
        "room",
        "midday rest",
        "free time",
        "buffer",
        "meal",
    )
    for activity in _activities_for_day(plan, adjustment.get("day")):
        overlap = _overlap_minutes(activity, window)
        if overlap < min_rest:
            continue
        activity_type = str(activity.get("type") or "")
        if activity_type in {"hotel", "buffer", "meal"}:
            return True
        if any(marker in _activity_text(activity) for marker in rest_markers):
            return True
    return False


def _has_indoor_or_buffer_in_window(
    plan: Dict[str, Any], adjustment: Dict[str, Any]
) -> bool:
    min_buffer = int(adjustment.get("min_buffer_minutes") or 30)
    window = adjustment.get("time_window") or []
    for activity in _activities_for_day(plan, adjustment.get("day")):
        overlap = _overlap_minutes(activity, window)
        if overlap < min_buffer:
            continue
        if activity.get("type") in {"hotel", "buffer", "meal"}:
            return True
        if activity.get("type") == "attraction" and not _looks_outdoor_attraction(
            activity
        ):
            return True
    return False


def _has_keyword(plan: Dict[str, Any], keywords: List[str]) -> bool:
    text = _plan_text(plan)
    return any(str(keyword) and str(keyword) in text for keyword in keywords)


def _return_day_has_buffer(plan: Dict[str, Any], min_buffer: int) -> bool:
    daily = plan.get("daily_plans", []) or []
    if not daily:
        return False
    activities = daily[-1].get("activities", []) or []
    for idx, activity in enumerate(activities):
        if activity.get("type") != "travel_intercity_public":
            continue
        prev_idx = idx - 1
        buffer_minutes = 0
        while prev_idx >= 0 and activities[prev_idx].get("type") == "buffer":
            buffer_minutes += _activity_duration(activities[prev_idx])
            prev_idx -= 1
        return buffer_minutes >= min_buffer
    # If no explicit return transport is present, use a weak route-safety proxy.
    return not any(
        activity.get("type") == "attraction" and _activity_duration(activity) > 0
        for activity in activities[-2:]
    )


def _first_day_light_after_delay(
    plan: Dict[str, Any], adjustment: Dict[str, Any]
) -> bool:
    activities = _activities_for_day(plan, 1)
    max_attractions = int(adjustment.get("max_first_day_attractions") or 2)
    attraction_count = sum(
        1 for activity in activities if activity.get("type") == "attraction"
    )
    if attraction_count > max_attractions:
        return False
    if not adjustment.get("requires_buffer_or_hotel_after_intercity"):
        return True
    for idx, activity in enumerate(activities):
        if activity.get("type") != "travel_intercity_public":
            continue
        following = activities[idx + 1 : idx + 4]
        return any(
            item.get("type") in {"hotel", "buffer", "meal"} for item in following
        )
    return any(activity.get("type") in {"hotel", "buffer"} for activity in activities)


def _environment_adjustment_results(
    plan: Dict[str, Any], ground_truth: Dict[str, Any]
) -> List[Dict[str, Any]]:
    state = ground_truth.get("oracle_state_after_turn") or {}
    events = state.get("active_environment_events") or []
    results: List[Dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        for adjustment in event.get("expected_adjustments") or []:
            if not isinstance(adjustment, dict):
                continue
            check = str(adjustment.get("check") or "")
            passed: Optional[bool]
            message: Optional[str] = None
            if check == "limit_outdoor_in_window":
                activities = _activities_for_day(plan, adjustment.get("day"))
                outdoor_minutes = sum(
                    _overlap_minutes(activity, adjustment.get("time_window") or [])
                    for activity in activities
                    if _looks_outdoor_attraction(activity)
                )
                max_minutes = int(adjustment.get("max_outdoor_minutes") or 90)
                passed = outdoor_minutes <= max_minutes
                message = (
                    None
                    if passed
                    else f"Outdoor attraction time in affected window is {outdoor_minutes}min > {max_minutes}min"
                )
            elif check == "has_indoor_or_buffer_on_day":
                passed = _has_buffer_or_indoor_on_day(plan, adjustment)
                message = (
                    None
                    if passed
                    else "Affected day lacks an indoor attraction, meal/hotel rest, or buffer block"
                )
            elif check == "has_indoor_or_buffer_in_window":
                passed = _has_indoor_or_buffer_in_window(plan, adjustment)
                message = (
                    None
                    if passed
                    else "Affected weather window lacks an indoor attraction, meal/hotel rest, or buffer block"
                )
            elif check == "has_hotel_or_rest_in_window":
                passed = _has_hotel_or_rest_in_window(plan, adjustment)
                message = (
                    None
                    if passed
                    else "Affected weather window lacks a hotel/meal/rest/buffer block"
                )
            elif check == "avoid_unavailable_poi":
                target = normalize_entity_name(adjustment.get("target") or "")
                plan_attractions = _extract_plan_attractions(plan)
                passed = bool(target) and target not in plan_attractions
                message = (
                    None
                    if passed
                    else f"Unavailable POI still appears in plan: {adjustment.get('target')}"
                )
            elif check == "first_day_light_after_transport_delay":
                passed = _first_day_light_after_delay(plan, adjustment)
                message = (
                    None
                    if passed
                    else "First day remains too activity-heavy after transport delay"
                )
            elif check == "first_day_buffer_for_local_practical":
                passed = _has_buffer_or_indoor_on_day(
                    plan,
                    {
                        "day": 1,
                        "min_buffer_minutes": adjustment.get("min_buffer_minutes", 30),
                    },
                )
                message = (
                    None
                    if passed
                    else "First day lacks buffer for local practical setup"
                )
            elif check == "avoid_peak_meal_window":
                peak = adjustment.get("peak_window") or [12 * 60, 13 * 60]
                peak_meals = [
                    activity
                    for day in plan.get("daily_plans", []) or []
                    for activity in day.get("activities", []) or []
                    if activity.get("type") == "meal"
                    and _overlap_minutes(activity, peak) > 0
                ]
                passed = not peak_meals
                message = (
                    None
                    if passed
                    else "Meal remains scheduled inside the specified peak queue window"
                )
            elif check == "avoid_peak_attraction_window":
                peak = adjustment.get("peak_window") or [10 * 60, 12 * 60]
                target_names = [
                    normalize_entity_name(name)
                    for name in adjustment.get("target_attractions")
                    or adjustment.get("target_attraction_names")
                    or []
                    if str(name).strip()
                ]
                target_set = set(target_names)
                peak_attractions = [
                    activity
                    for day in plan.get("daily_plans", []) or []
                    for activity in day.get("activities", []) or []
                    if activity.get("type") == "attraction"
                    and _overlap_minutes(activity, peak) > 0
                    and (
                        not target_set
                        or normalize_entity_name(_activity_name(activity)) in target_set
                    )
                ]
                passed = not peak_attractions
                if passed:
                    message = None
                elif target_set:
                    message = "Target high-queue attraction remains scheduled inside the specified peak queue window"
                else:
                    message = "Attraction remains scheduled inside the specified peak queue window"
            elif check == "return_day_transport_buffer":
                passed = _return_day_has_buffer(
                    plan, int(adjustment.get("min_buffer_minutes") or 60)
                )
                message = (
                    None
                    if passed
                    else "Return day lacks transport buffer before intercity segment"
                )
            elif check in {
                "include_queue_or_offpeak_note",
                "include_traffic_buffer_note",
                "include_backup_or_replacement_note",
                "include_practical_note",
            }:
                passed = _has_keyword(
                    plan, [str(item) for item in adjustment.get("keywords") or []]
                )
                message = (
                    None
                    if passed
                    else f"Plan text lacks expected note keywords for {check}"
                )
            elif check == "explain_unsolved":
                passed = _status_is_unsat(plan)
                message = (
                    None
                    if passed
                    else "Expected no-solution explanation for unavailable non-replaceable POI"
                )
            else:
                passed = None
                message = f"No deterministic checker registered for expected adjustment: {check}"
            results.append(
                {
                    "name": check,
                    "passed": passed,
                    "message": message,
                    "event_factor": event.get("factor"),
                    "event_type": event.get("type"),
                    "details": adjustment,
                }
            )
    return results


def _environment_update_result(
    name: str, plan: Dict[str, Any], ground_truth: Dict[str, Any]
) -> Dict[str, Any]:
    daily_plans = plan.get("daily_plans", []) or []
    if not daily_plans:
        return {
            "name": name,
            "passed": False,
            "message": "Missing daily_plans for environment-aware replanning",
        }
    adjustment_results = _environment_adjustment_results(plan, ground_truth)
    adjustment_bools = [
        item["passed"] for item in adjustment_results if item.get("passed") is not None
    ]
    passed = all(adjustment_bools) if adjustment_bools else True
    failed_messages = [
        item["message"]
        for item in adjustment_results
        if item.get("passed") is False and item.get("message")
    ]
    return {
        "name": name,
        "passed": passed,
        "message": None if passed else "; ".join(failed_messages),
        "details": {
            "event_adjustment_checks": adjustment_results,
        },
    }


def _latest_active_user_delta(
    ground_truth: Dict[str, Any], key: str
) -> Optional[Dict[str, Any]]:
    state = ground_truth.get("oracle_state_after_turn") or {}
    for delta in reversed(state.get("active_user_state_deltas") or []):
        if isinstance(delta, dict) and isinstance(delta.get(key), dict):
            return delta[key]
    return None


def _latest_active_request_delta(
    ground_truth: Dict[str, Any], key: str
) -> Optional[Dict[str, Any]]:
    state = ground_truth.get("oracle_state_after_turn") or {}
    for delta in reversed(state.get("active_request_resolutions") or []):
        if isinstance(delta, dict) and isinstance(delta.get(key), dict):
            return delta[key]
    return None


def _resolved_pacing_limit_result(
    plan: Dict[str, Any], ground_truth: Dict[str, Any]
) -> Dict[str, Any]:
    request = _latest_active_request_delta(ground_truth, "resolved_pacing_limit") or {}
    oracle = ground_truth.get("verification_oracle") or {}
    try:
        max_attractions = int(
            request.get("max_attractions_per_day")
            or oracle.get("max_attractions_per_day")
        )
    except (TypeError, ValueError):
        return {
            "name": "resolved_pacing_limit",
            "passed": None,
            "message": "Resolved pacing limit missing max_attractions_per_day",
        }
    violations: List[Dict[str, Any]] = []
    for idx, day in enumerate(plan.get("daily_plans", []) or [], start=1):
        attractions = [
            _activity_name(activity)
            for activity in day.get("activities", []) or []
            if activity.get("type") == "attraction"
        ]
        if len(attractions) > max_attractions:
            violations.append(
                {
                    "day": idx,
                    "count": len(attractions),
                    "attractions": attractions,
                }
            )
    passed = not violations
    return {
        "name": "resolved_pacing_limit",
        "passed": passed,
        "message": None
        if passed
        else f"Plan exceeds max attractions per day: {violations[:5]}",
        "details": {
            "max_attractions_per_day": max_attractions,
            "violations": violations,
        },
    }


def _non_itinerary_resolution_result(name: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    passed = _status_is_clarification(plan)
    return {
        "name": name,
        "passed": passed,
        "message": None
        if passed
        else "Expected a clarification/conflict response without daily_plans",
    }


def _generated_constraint_results(
    generated_constraints: Dict[str, Any],
    hard_constraints: Dict[str, Any],
    prefix: str,
    fallback_message: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for key, constraint in generated_constraints.items():
        if not str(key).startswith(prefix):
            continue
        detail = hard_constraints.get(key)
        passed = bool(detail and detail.get("passed"))
        results.append(
            {
                "name": key,
                "passed": passed,
                "message": None
                if passed
                else (detail or {}).get("message", fallback_message),
                "details": constraint,
            }
        )
    return results


def _meal_type_matches(actual: object, expected: object) -> bool:
    actual_text = str(actual or "").strip().lower()
    expected_text = str(expected or "").strip().lower()
    if not expected_text:
        return True
    aliases = {
        "lunch": {"lunch"},
        "dinner": {"dinner", "supper"},
        "breakfast": {"breakfast"},
    }
    expected_aliases = aliases.get(expected_text, {expected_text})
    return actual_text in expected_aliases or any(
        alias in actual_text for alias in expected_aliases
    )


def _restaurant_slot_result(
    plan: Dict[str, Any], constraint_key: str, constraint: Dict[str, Any]
) -> Dict[str, Any] | None:
    meal_day = constraint.get("meal_day")
    meal_type = constraint.get("meal_type")
    if meal_day in (None, "") and meal_type in (None, ""):
        return None
    acceptable = [
        normalize_entity_name(str(name))
        for name in constraint.get("acceptable_restaurant_names")
        or [constraint.get("restaurant_name")]
        if str(name or "").strip()
    ]
    if not acceptable:
        return {
            "name": f"{constraint_key}:meal_slot",
            "passed": False,
            "message": "No acceptable restaurant names available for meal-slot check",
        }
    activities = (
        _activities_for_day(plan, meal_day)
        if meal_day not in (None, "")
        else [
            activity
            for day in plan.get("daily_plans", []) or []
            for activity in day.get("activities", []) or []
        ]
    )
    seen_names: List[str] = []
    for activity in activities:
        if str(activity.get("type") or "").strip() != "meal":
            continue
        details = activity.get("details") or {}
        name = extract_entity_name(activity, "restaurant")
        if name:
            seen_names.append(name)
        if normalize_entity_name(name) not in acceptable:
            continue
        if _meal_type_matches(details.get("meal_type"), meal_type):
            return {
                "name": f"{constraint_key}:meal_slot",
                "passed": True,
                "message": None,
            }
        return {
            "name": f"{constraint_key}:meal_slot",
            "passed": False,
            "message": f"Restaurant is present but not in requested meal slot: day={meal_day}, meal_type={meal_type}",
        }
    return {
        "name": f"{constraint_key}:meal_slot",
        "passed": False,
        "message": f"Restaurant not found on requested day {meal_day}; day meal names={seen_names[:5]}",
    }


def _first_formal_activity_start_minutes(
    plan: Dict[str, Any], day_number: object
) -> Optional[int]:
    excluded_types = {"hotel", "meal", "buffer", "free_time", "rest"}
    starts: List[int] = []
    for activity in _activities_for_day(plan, day_number):
        activity_type = str(activity.get("type") or "").strip()
        if activity_type in excluded_types:
            continue
        start, _end = slot_to_minutes(activity.get("time_slot"))
        if start is not None:
            starts.append(start)
    return min(starts) if starts else None


def _late_start_update_result(
    plan: Dict[str, Any], ground_truth: Dict[str, Any]
) -> Dict[str, Any]:
    request = _latest_active_user_delta(ground_truth, "late_start_request") or {}
    oracle = ground_truth.get("verification_oracle") or {}
    day = request.get("day") or oracle.get("day")
    earliest = request.get("earliest_start_minutes") or oracle.get(
        "earliest_start_minutes"
    )
    try:
        earliest_minutes = int(earliest)
    except (TypeError, ValueError):
        return {
            "name": "late_start_request",
            "passed": None,
            "message": "Late-start request missing earliest_start_minutes",
        }
    actual = _first_formal_activity_start_minutes(plan, day)
    passed = actual is not None and actual >= earliest_minutes
    return {
        "name": "late_start_request",
        "passed": passed,
        "message": None
        if passed
        else f"First formal activity on day {day} starts at {actual}, earlier than requested {earliest_minutes}",
        "details": {
            "day": day,
            "earliest_start_minutes": earliest_minutes,
            "actual_first_formal_start_minutes": actual,
        },
    }


def _last_formal_activity_end_minutes(
    plan: Dict[str, Any], day_number: object
) -> Optional[int]:
    excluded_types = {"hotel", "meal", "buffer", "free_time", "rest"}
    ends: List[int] = []
    for activity in _activities_for_day(plan, day_number):
        activity_type = str(activity.get("type") or "").strip()
        if activity_type in excluded_types:
            continue
        _start, end = slot_to_minutes(activity.get("time_slot"))
        if end is not None:
            ends.append(end)
    return max(ends) if ends else None


def _has_midday_rest_block(
    plan: Dict[str, Any], request: Dict[str, Any]
) -> Tuple[bool, List[Dict[str, Any]]]:
    day = request.get("day")
    window = request.get("rest_window") or [12 * 60 + 30, 14 * 60 + 30]
    try:
        min_minutes = int(request.get("min_rest_minutes") or 45)
    except (TypeError, ValueError):
        min_minutes = 45
    allowed_types = {
        str(item)
        for item in request.get("allowed_activity_types")
        or ["meal", "hotel", "buffer", "rest", "free_time"]
    }
    evidence: List[Dict[str, Any]] = []
    for activity in _activities_for_day(plan, day):
        overlap = _overlap_minutes(activity, window)
        text = json.dumps(activity, ensure_ascii=False, sort_keys=True).lower()
        activity_type = str(activity.get("type") or "").strip()
        looks_like_rest = activity_type in allowed_types or any(
            marker in text
            for marker in (
                "rest",
                "midday rest",
                "hotel",
                "room",
                "free time",
                "buffer",
            )
        )
        if looks_like_rest and overlap >= min_minutes:
            evidence.append(
                {
                    "type": activity_type,
                    "name": _activity_name(activity),
                    "time_slot": activity.get("time_slot"),
                    "overlap_minutes": overlap,
                }
            )
    return bool(evidence), evidence


def _schedule_update_result(
    plan: Dict[str, Any], ground_truth: Dict[str, Any]
) -> Dict[str, Any]:
    request = _latest_active_user_delta(ground_truth, "schedule_update") or {}
    oracle = ground_truth.get("verification_oracle") or {}
    subtype = str(request.get("schedule_subtype") or "")
    if subtype == "early_finish":
        day = request.get("day") or oracle.get("day")
        latest = request.get("latest_formal_end_minutes") or oracle.get(
            "latest_formal_end_minutes"
        )
        try:
            latest_minutes = int(latest)
        except (TypeError, ValueError):
            return {
                "name": "schedule_update",
                "passed": None,
                "message": "Early-finish request missing latest_formal_end_minutes",
            }
        actual = _last_formal_activity_end_minutes(plan, day)
        passed = actual is not None and actual <= latest_minutes
        return {
            "name": "schedule_update",
            "passed": passed,
            "message": None
            if passed
            else f"Last formal activity on day {day} ends at {actual}, later than requested {latest_minutes}",
            "details": {
                "schedule_subtype": subtype,
                "day": day,
                "latest_formal_end_minutes": latest_minutes,
                "actual_last_formal_end_minutes": actual,
            },
        }
    if subtype == "midday_rest":
        passed, evidence = _has_midday_rest_block(plan, request)
        return {
            "name": "schedule_update",
            "passed": passed,
            "message": None
            if passed
            else "Requested midday rest window lacks a qualifying meal/hotel/buffer/rest block",
            "details": {
                "schedule_subtype": subtype,
                "day": request.get("day"),
                "rest_window": request.get("rest_window"),
                "min_rest_minutes": request.get("min_rest_minutes"),
                "evidence": evidence,
            },
        }
    return {
        "name": "schedule_update",
        "passed": None,
        "message": f"No deterministic checker registered for schedule_update subtype: {subtype or 'missing'}",
        "details": request,
    }


def _strenuous_activity_violations(plan: Dict[str, Any]) -> List[str]:
    markers = (
        "boardwalk",
        "trail",
        "hike",
        "hiking",
        "mountain",
        "climb",
        "canyon",
        "peak",
        "ridge",
        "great wall",
        "forest park",
        "highland",
        "cable car",
    )
    violations: List[str] = []
    for day in plan.get("daily_plans", []) or []:
        for activity in day.get("activities", []) or []:
            if activity.get("type") != "attraction":
                continue
            name = _activity_name(activity)
            name_lower = name.lower()
            duration = _activity_duration(activity)
            if any(marker in name_lower for marker in markers) or duration >= 210:
                violations.append(
                    name or str(activity.get("description") or "unnamed attraction")
                )
    return violations


def _party_update_result(
    plan: Dict[str, Any],
    ground_truth: Dict[str, Any],
    feasibility_results: Dict[str, Tuple[bool, Optional[str]]] | None,
) -> Dict[str, Any]:
    request = _latest_active_user_delta(ground_truth, "party_update") or {}
    cost_check = (feasibility_results or {}).get("cost_calculation_correctness")
    cost_passed = bool(cost_check and cost_check[0])
    subtype = str(request.get("party_subtype") or "")
    violations = (
        _strenuous_activity_violations(plan)
        if subtype in {"child_added", "elder_added"}
        else []
    )
    passed = cost_passed and not violations
    message_parts: List[str] = []
    if not cost_passed:
        message_parts.append(
            "Budget/cost arithmetic did not pass under updated people_number/room_number"
        )
    if violations:
        message_parts.append(
            f"Plan still includes strenuous activities after {subtype}: {violations[:5]}"
        )
    return {
        "name": "party_update",
        "passed": passed,
        "message": None if passed else "; ".join(message_parts),
        "details": {
            "party_subtype": subtype,
            "new_people_number": request.get("new_people_number"),
            "new_room_number": request.get("new_room_number"),
            "strenuous_activity_violations": violations,
            "cost_calculation_check": cost_check,
        },
    }


def _duration_update_result(
    plan: Dict[str, Any], ground_truth: Dict[str, Any]
) -> Dict[str, Any]:
    request = _latest_active_user_delta(ground_truth, "duration_update") or {}
    try:
        expected_days = int(request.get("new_days"))
    except (TypeError, ValueError):
        return {
            "name": "duration_update",
            "passed": None,
            "message": "Duration update missing new_days",
        }
    actual_days = len(plan.get("daily_plans", []) or [])
    passed = actual_days == expected_days
    return {
        "name": "duration_update",
        "passed": passed,
        "message": None
        if passed
        else f"Plan has {actual_days} day(s), expected {expected_days}",
        "details": {
            "expected_days": expected_days,
            "actual_days": actual_days,
            "change_type": request.get("change_type"),
            "new_return_date": request.get("new_return_date"),
        },
    }


def _dietary_update_result(
    plan: Dict[str, Any], ground_truth: Dict[str, Any]
) -> Dict[str, Any]:
    request = _latest_active_user_delta(ground_truth, "dietary_update") or {}
    banned = [
        str(item).lower()
        for item in request.get("banned_keywords")
        or [
            "sichuan",
            "hunan",
            "hot pot",
            "spicy",
            "chili",
            "chilli",
            "peppercorn",
            "mala",
        ]
        if str(item)
    ]
    violations: List[str] = []
    for day in plan.get("daily_plans", []) or []:
        for activity in day.get("activities", []) or []:
            if activity.get("type") != "meal":
                continue
            details = activity.get("details") or {}
            text = json.dumps(details, ensure_ascii=False, sort_keys=True).lower()
            if any(keyword in text for keyword in banned):
                violations.append(
                    extract_entity_name(activity, "restaurant")
                    or activity.get("description")
                    or "unnamed meal"
                )
    passed = not violations
    return {
        "name": "dietary_update",
        "passed": passed,
        "message": None
        if passed
        else f"Plan still includes restaurants conflicting with dietary update: {violations[:5]}",
        "details": {
            "restriction": request.get("restriction"),
            "banned_keywords": banned,
            "violations": violations,
        },
    }
