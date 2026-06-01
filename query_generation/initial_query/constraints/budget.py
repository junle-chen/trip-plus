"""Budget hard-constraint sampling and profile-conditioned budget adjustment."""

from __future__ import annotations

import math
import random
from typing import Any

from query_generation.common import ConstraintSpec, SampleContext, filtered_rows_by_direction, safe_float
from query_generation.initial_query.config import (
    BUDGET_SOFT_RULE_IDS,
    BUDGET_TRIGGER_PROB,
    normalize_interaction_archetype,
)
from query_generation.user_profile import get_rule_ids


def build_budget_constraint(
    ctx: SampleContext,
    chosen: list[ConstraintSpec],
    rng: random.Random | None = None,
    *,
    force_unsat: bool = False,
) -> ConstraintSpec:
    rng = rng or random.Random()
    hotel_price = restaurant_price = transport_price = ticket_total = 0.0
    for spec in chosen:
        if spec.key.startswith("hotel_"):
            hotel_price = safe_float(spec.data.get("hotel_price"))
        elif spec.key.startswith("restaurant_"):
            restaurant_price += safe_float(spec.data.get("price_per_person"))
        elif spec.key.startswith(("train_", "flight_")):
            transport_price += safe_float(spec.data.get("outbound_price")) + safe_float(spec.data.get("inbound_price"))
        elif spec.key.startswith("attraction_"):
            ticket_total += sum(safe_float(value) for value in spec.data.get("ticket_prices", []))

    nights = max(1, ctx.days - 1)
    people_number = max(1, ctx.people_number)
    room_number = max(1, ctx.room_number)
    estimated_meal_slots = max(2, ctx.days * 2)
    local_transport_buffer = people_number * max(1, ctx.days) * 40.0

    minimum_feasible_budget = transport_price * people_number
    minimum_feasible_budget += hotel_price * room_number * nights
    minimum_feasible_budget += restaurant_price * people_number * estimated_meal_slots
    minimum_feasible_budget += ticket_total * people_number
    minimum_feasible_budget += local_transport_buffer
    minimum_feasible_budget = max(minimum_feasible_budget, 1200.0)
    minimum_feasible_budget = math.ceil(minimum_feasible_budget / 100.0) * 100

    base_data = {
        "minimum_feasible_budget": int(minimum_feasible_budget),
        "estimated_meal_slots": estimated_meal_slots,
        "estimated_breakdown": {
            "transportation": round(transport_price * people_number, 2),
            "local_transport_buffer": round(local_transport_buffer, 2),
            "accommodation": round(hotel_price * room_number * nights, 2),
            "meals": round(restaurant_price * people_number * estimated_meal_slots, 2),
            "attractions": round(ticket_total * people_number, 2),
        },
    }

    if force_unsat:
        slack_ratio = rng.uniform(0.72, 0.88)
        max_budget = int(max(500, math.floor((minimum_feasible_budget * slack_ratio) / 100.0) * 100))
        return ConstraintSpec(
            "budget_constraint",
            {
                **base_data,
                "constraint_type": "budget_range_total",
                "is_unsat": True,
                "max_budget": max_budget,
                "budget_slack_ratio": round(slack_ratio, 4),
                "reason_code": "budget_below_minimum_feasible_cost",
                "reason_message": f"The budget cap is {max_budget} RMB, below the minimum feasible cost of {int(minimum_feasible_budget)} RMB.",
                "reason_keywords": ["budget", "minimum feasible cost", "insufficient", "unavailable"],
                "core_conflict": [
                    {"left": f"max_budget={max_budget}", "right": f"minimum_feasible_budget={int(minimum_feasible_budget)}"},
                ],
            },
            f"The total budget can be at most {max_budget} RMB.",
            f"The total budget can be at most {max_budget} RMB.",
        )

    min_budget = int(minimum_feasible_budget)
    slack_ratio = rng.uniform(1.05, 1.20)
    max_budget = int(math.ceil((minimum_feasible_budget * slack_ratio) / 100.0) * 100)
    return ConstraintSpec(
        "budget_constraint",
        {
            **base_data,
            "constraint_type": "budget_range_total",
            "is_unsat": False,
            "min_budget": min_budget,
            "max_budget": max_budget,
            "budget_slack_ratio": round(slack_ratio, 4),
        },
        f"I would like the total budget to stay between {min_budget} and {max_budget} RMB, ideally not exceeding {max_budget} RMB.",
        f"The total budget should stay between {min_budget} and {max_budget} RMB.",
    )


def budget_should_surface(
    derived: dict[str, Any],
    archetype: str,
    rng: random.Random,
) -> bool:
    probability = BUDGET_TRIGGER_PROB
    rule_ids = get_rule_ids(derived)
    if "budget_tight_cap" in rule_ids:
        probability += 0.20
    elif "budget_guarded" in rule_ids:
        probability += 0.08
    else:
        try:
            effective_budget = float((derived.get("derivation_context") or {}).get("upper_budget_per_person_day") or 0)
        except (TypeError, ValueError):
            effective_budget = 0.0
        if effective_budget >= 950:
            probability -= 0.04
    archetype = normalize_interaction_archetype(archetype)
    if archetype in {"request_resolution", "long_horizon_alignment"}:
        probability += 0.03
    return rng.random() < max(0.03, min(probability, 0.38))


def _round_budget_to_100(value: float) -> int:
    return max(500, int(round(value / 100.0) * 100))


def _profile_budget_slack_range(derived: dict[str, Any]) -> tuple[float, float]:
    rule_ids = get_rule_ids(derived)
    if "budget_tight_cap" in rule_ids:
        low, high = 1.06, 1.14
    elif "budget_guarded" in rule_ids:
        low, high = 1.14, 1.28
    else:
        low, high = 1.25, 1.45

    if "hotel_value_first" in rule_ids:
        high = min(high, 1.20)
    if "meal_avoid_expensive" in rule_ids:
        high = min(high, 1.22)
    return low, max(low + 0.02, high)


def apply_profile_budget_policy(
    budget_data: dict[str, Any],
    derived: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    minimum = float(budget_data.get("minimum_feasible_budget") or budget_data.get("min_budget") or 0)
    if minimum <= 0:
        return budget_data

    low, high = _profile_budget_slack_range(derived)
    slack_ratio = rng.uniform(low, high)
    max_budget = _round_budget_to_100(minimum * slack_ratio)
    adjusted = dict(budget_data)
    adjusted["is_unsat"] = False
    adjusted["min_budget"] = int(minimum)
    adjusted["max_budget"] = max(max_budget, int(minimum))
    adjusted["budget_slack_ratio"] = round(slack_ratio, 4)
    adjusted["budget_policy"] = {
        "type": "db_feasible_cost_plus_profile_slack",
        "profile_budget_rule": next((rule_id for rule_id in get_rule_ids(derived) if rule_id.startswith("budget_")), "budget_unspecified"),
        "basis": ["intercity_transport", "local_transport_buffer", "hotel_nights", "restaurant_meals", "attraction_tickets"],
    }
    return adjusted


def exclude_budget_soft_rules_when_hard_budget(profile: dict[str, Any], hard_constraints: dict[str, Any]) -> dict[str, Any]:
    if "budget_constraint" not in (hard_constraints or {}):
        return profile
    updated = dict(profile)
    updated["rule_ids"] = [
        rule_id for rule_id in list(profile.get("rule_ids") or []) if rule_id not in BUDGET_SOFT_RULE_IDS
    ]
    updated["rules"] = [
        rule
        for rule in list(profile.get("rules") or [])
        if not (isinstance(rule, dict) and rule.get("rule_id") in BUDGET_SOFT_RULE_IDS)
    ]
    return updated


def _cheapest_direction_route_price(rows: list[dict[str, str]], ctx: SampleContext, direction: str) -> float:
    direction_rows = filtered_rows_by_direction(rows, ctx, direction)
    prices_by_route: dict[str, float] = {}
    for row in direction_rows:
        route_idx = str(row.get("route_index") or "").strip()
        if not route_idx:
            route_idx = "|".join(
                str(row.get(key) or "").strip()
                for key in ("dep_station_code", "arr_station_code", "dep_datetime", "arr_datetime", "train_no", "flight_no")
            )
        price = safe_float(row.get("price"), 0.0)
        if price <= 0:
            continue
        prices_by_route[route_idx] = prices_by_route.get(route_idx, 0.0) + price
    return min(prices_by_route.values()) if prices_by_route else 0.0


def _estimate_intercity_transport_total(
    *,
    ctx: SampleContext,
    db: dict[str, list[dict[str, str]]],
    mode: str,
) -> float:
    category_order = ["flights", "trains"] if mode == "flight" else ["trains", "flights"]
    for category in category_order:
        rows = db.get(category, []) or []
        outbound = _cheapest_direction_route_price(rows, ctx, "outbound")
        inbound = _cheapest_direction_route_price(rows, ctx, "inbound")
        if outbound > 0 and inbound > 0:
            return round((outbound + inbound) * max(1, ctx.people_number), 2)
    return 0.0


def repair_budget_transport_estimate(
    budget_data: dict[str, Any],
    *,
    ctx: SampleContext,
    db: dict[str, list[dict[str, str]]],
    mode: str,
) -> dict[str, Any]:
    breakdown = dict(budget_data.get("estimated_breakdown", {}) or {})
    current_transport = safe_float(breakdown.get("transportation"), 0.0)
    db_transport = _estimate_intercity_transport_total(ctx=ctx, db=db, mode=mode)
    if db_transport <= current_transport:
        return budget_data

    repaired = dict(budget_data)
    breakdown["transportation"] = db_transport
    repaired["estimated_breakdown"] = breakdown
    old_minimum = safe_float(repaired.get("minimum_feasible_budget"), 0.0)
    repaired["minimum_feasible_budget"] = _round_budget_to_100(old_minimum + db_transport - current_transport)
    return repaired
