"""Canonical itinerary cost calculation shared by evaluators."""

from __future__ import annotations

import re
from typing import Any, Dict


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _safe_int(value: Any, default: int = 1) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def travel_city_cost_multiplier(details: Dict[str, Any], cost: float, people_number: int) -> int:
    """Return the multiplier for a local-transfer unit cost."""
    if cost <= 0:
        return 1
    mode_text = " ".join(
        str(details.get(key) or "")
        for key in ("mode", "transport_mode", "recommended_mode", "pricing_rule")
    ).lower()
    taxi_tokens = ("taxi", "cab", "drive", "driving", "ride-hailing", "ride hailing")
    public_tokens = ("subway", "metro", "bus", "tram", "light rail")
    walking_tokens = ("walking", "walk")
    if any(token in mode_text for token in taxi_tokens):
        return max(1, (people_number + 3) // 4)
    if any(token in mode_text for token in public_tokens):
        return people_number
    if any(token in mode_text for token in walking_tokens):
        return 1
    return 1


def compute_plan_cost(plan: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, float]:
    """Compute itinerary cost from structured activities, not from model summary text.

    Rules match the budget arithmetic feasibility check:
    - one intercity fare per same-day intercity chain, multiplied by people;
    - accommodation for every day except the final day, multiplied by room count;
    - meals and attraction tickets multiplied by people;
    - local transfer cost multiplied according to mode.
    """
    daily_plans = plan.get("daily_plans", []) or []
    people_number = _safe_int(meta.get("people_number"), 1)
    room_number = _safe_int(meta.get("room_number"), 1)

    transportation_cost = 0.0
    accommodation_cost = 0.0
    meals_cost = 0.0
    attractions_cost = 0.0

    for day in daily_plans:
        day_intercity_cost = 0.0
        found_first = False
        for act in day.get("activities", []) or []:
            if act.get("type") == "travel_intercity_public" and not found_first:
                cost = _safe_float((act.get("details") or {}).get("cost"))
                if cost is not None:
                    day_intercity_cost = cost
                    found_first = True
        if day_intercity_cost > 0:
            transportation_cost += day_intercity_cost * people_number

    for day in daily_plans[:-1]:
        accom = day.get("accommodation")
        if isinstance(accom, dict):
            price = _safe_float(accom.get("price") or accom.get("price_per_night") or accom.get("cost"))
            if price is not None:
                accommodation_cost += price * room_number

    for day in daily_plans:
        for act in day.get("activities", []) or []:
            details = act.get("details") or {}
            if act.get("type") == "meal":
                cost = _safe_float(details.get("cost"))
                if cost is not None:
                    meals_cost += cost * people_number
            elif act.get("type") == "attraction":
                cost = _safe_float(details.get("cost"))
                if cost is not None:
                    attractions_cost += cost * people_number
            elif act.get("type") == "travel_city":
                cost = _safe_float(details.get("cost"))
                if cost is not None:
                    transportation_cost += cost * travel_city_cost_multiplier(details, cost, people_number)

    total = transportation_cost + accommodation_cost + meals_cost + attractions_cost
    return {
        "transportation": transportation_cost,
        "accommodation": accommodation_cost,
        "meals": meals_cost,
        "attractions": attractions_cost,
        "total": total,
    }


def compute_total_cost(plan: Dict[str, Any], meta: Dict[str, Any]) -> float:
    return compute_plan_cost(plan, meta)["total"]
