"""Shared helpers for explicit hard-constraint checks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..costing import compute_plan_cost
from ..scoring_config import normalize_intercity_mode
from ..utils import extract_entity_name, normalize_entity_name


def _normalized_name_set(values: List[str]) -> set[str]:
    return {normalize_entity_name(value) for value in values if value}


def calculate_hard_score(
    hard_results: Dict[str, Tuple[bool, Optional[str]]],
) -> Dict[str, Any]:
    """Apply the hard-constraint veto rule to individual check results."""
    passed_count = 0
    total = 0
    constraints = {}

    for constraint_name, (ok, msg) in hard_results.items():
        if ok is None:
            continue
        total += 1
        constraints[constraint_name] = {"passed": ok, "message": msg}
        if ok:
            passed_count += 1

    score = 1.0 if total > 0 and passed_count == total else 0.0
    return {"score": score, "constraints": constraints}


def _string_list(value: Any) -> List[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _resolve_acceptables(
    constraint_data: Dict[str, Any], plural_key: str, singular_key: str
) -> List[str]:
    values = _string_list(constraint_data.get(plural_key))
    if values:
        return values
    return _string_list(constraint_data.get(singular_key))


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _budget_component_value(plan: Dict[str, Any], key: str) -> Optional[float]:
    budget = plan.get("budget_summary") or {}
    if not isinstance(budget, dict):
        return None
    aliases = {
        "transportation": ("transportation", "transport"),
        "accommodation": ("accommodation", "hotel", "hotels"),
        "meals": ("meals", "meal", "food"),
        "attractions": ("attractions_and_tickets", "attractions", "tickets"),
    }
    for alias in aliases.get(key, (key,)):
        value = _safe_float(budget.get(alias))
        if value is not None:
            return value
    return None


def _budget_component_matches(
    plan: Dict[str, Any], meta: Dict[str, Any], keys: List[str]
) -> Tuple[bool, Optional[str]]:
    if not plan.get("daily_plans"):
        return False, "Plan missing daily_plans"
    if not isinstance(plan.get("budget_summary"), dict):
        return False, "Plan missing budget_summary"

    costs = compute_plan_cost(plan, meta)
    mismatches = []
    for key in keys:
        expected = _safe_float(costs.get(key))
        actual = _budget_component_value(plan, key)
        if expected is None:
            continue
        if actual is None:
            mismatches.append(f"{key}: missing")
            continue
        tolerance = max(1.0, abs(expected) * 0.02)
        if abs(actual - expected) > tolerance:
            mismatches.append(f"{key}: plan={actual:.2f}, expected={expected:.2f}")
    if mismatches:
        return (
            False,
            "Budget component mismatch under explicit party/room metadata: "
            + "; ".join(mismatches),
        )
    return True, None


def _transport_option_matches(
    plan_item: Dict[str, Any], option: Dict[str, Any], number_key: str
) -> bool:
    """Match a planned transport segment against a DB-backed acceptable option.

    Train/flight numbers alone are ambiguous because the same number can appear
    with multiple seat classes or prices.  If the query metadata carries
    acceptable_*_options, require the plan's per-person cost to match that
    option as well. This makes seat-class constraints auditable even when the
    converted plan does not expose seat_class explicitly.
    """
    if (
        str(plan_item.get(number_key, "")).strip()
        != str(option.get(number_key, "")).strip()
    ):
        return False
    option_price = _safe_float(option.get("price"))
    if option_price is None:
        return True
    plan_cost = _safe_float(plan_item.get("cost"))
    if plan_cost is None:
        return False
    return int(round(plan_cost)) == int(round(option_price))


def _transport_option_label(option: Dict[str, Any], number_key: str) -> str:
    number = option.get(number_key, "")
    price = option.get("price")
    return f"{number}@{price}" if price not in (None, "") else str(number)


def _transport_direction_satisfied(
    plan_items: List[Dict[str, Any]],
    constraint_data: Dict[str, Any],
    *,
    direction: str,
    transport_label: str,
    number_key: str,
    acceptable_numbers_key: str,
    singular_number_key: str,
    acceptable_options_key: str,
) -> Tuple[bool, Optional[str]]:
    acceptable_options = [
        option
        for option in constraint_data.get(acceptable_options_key, []) or []
        if isinstance(option, dict) and str(option.get(number_key, "")).strip()
    ]

    if acceptable_options:
        if any(
            _transport_option_matches(plan_item, option, number_key)
            for plan_item in plan_items
            for option in acceptable_options
        ):
            return True, None
        readable = [
            _transport_option_label(option, number_key) for option in acceptable_options
        ]
        return (
            False,
            f"Required {direction} {transport_label} not found in acceptable option set: {readable}",
        )

    candidate_numbers = _resolve_acceptables(
        constraint_data, acceptable_numbers_key, singular_number_key
    )
    if not candidate_numbers:
        return True, None
    plan_numbers = {str(item.get(number_key, "")).strip() for item in plan_items}
    if plan_numbers.intersection(candidate_numbers):
        return True, None
    return (
        False,
        f"Required {direction} {transport_label} not found in acceptable set: {candidate_numbers}",
    )


def _extract_flights_from_plan(plan: Dict) -> List[Dict]:
    """Extract all flight information from plan"""
    flights = []

    if "daily_plans" not in plan:
        return flights

    for day_plan in plan["daily_plans"]:
        if "activities" not in day_plan:
            continue

        for activity in day_plan["activities"]:
            if activity.get("type") == "travel_intercity_public":
                details = activity.get("details", {})
                mode = normalize_intercity_mode(details.get("mode"))
                # Check if it's a flight
                if mode == "flight":
                    flights.append(
                        {
                            "flight_no": details.get("number", ""),
                            "airline": details.get(
                                "number", ""
                            ),  # Airline usually in flight number
                            "cost": details.get("cost", ""),
                            "seat_class": details.get("seat_class")
                            or details.get("class")
                            or details.get("cabin_class", ""),
                        }
                    )

    return flights


def _extract_trains_from_plan(plan: Dict) -> List[Dict]:
    """Extract all train information from plan"""
    trains = []

    if "daily_plans" not in plan:
        return trains

    for day_plan in plan["daily_plans"]:
        if "activities" not in day_plan:
            continue

        for activity in day_plan["activities"]:
            if activity.get("type") == "travel_intercity_public":
                details = activity.get("details", {})
                mode = normalize_intercity_mode(details.get("mode"))
                # Check if it's a train
                if mode == "train":
                    trains.append(
                        {
                            "train_no": details.get("number", ""),
                            "cost": details.get("cost", ""),
                            "seat_class": details.get("seat_class")
                            or details.get("class")
                            or details.get("seat", ""),
                        }
                    )

    return trains


def _extract_hotels_from_plan(plan: Dict) -> List[Dict]:
    """Extract all hotel information from plan"""
    hotels = []

    if "daily_plans" not in plan:
        return hotels

    for day_plan in plan["daily_plans"]:
        accommodation = day_plan.get("accommodation")
        if accommodation:
            hotels.append(
                {
                    "name": extract_entity_name(accommodation, "hotel"),
                }
            )

    return hotels


def _extract_restaurants_from_plan(plan: Dict) -> List[Dict]:
    """Extract all restaurant information from plan"""
    restaurants = []

    if "daily_plans" not in plan:
        return restaurants

    for day_plan in plan["daily_plans"]:
        if "activities" not in day_plan:
            continue

        for activity in day_plan["activities"]:
            if activity.get("type") == "meal":
                restaurants.append(
                    {
                        "name": extract_entity_name(activity, "restaurant"),
                    }
                )

    return restaurants


def _extract_attractions_from_plan(plan: Dict) -> List[Dict]:
    """Extract all attraction information from plan"""
    attractions = []

    if "daily_plans" not in plan:
        return attractions

    for day_plan in plan["daily_plans"]:
        if "activities" not in day_plan:
            continue

        for activity in day_plan["activities"]:
            if activity.get("type") == "attraction":
                attractions.append(
                    {
                        "name": extract_entity_name(activity, "attraction"),
                    }
                )

    return attractions


# ============================================================================
# Budget Constraints
# ============================================================================
