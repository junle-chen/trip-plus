"""Selection and dispatch for database-backed hard constraints."""

from __future__ import annotations

import random
from typing import Any

from query_generation.initial_query.constraints.attractions import build_attraction_constraint
from query_generation.initial_query.constraints.dining import build_restaurant_constraint
from query_generation.initial_query.constraints.lodging import build_hotel_constraint
from query_generation.initial_query.constraints.transport import build_transport_constraint
from query_generation.city_database import RouteOption
from query_generation.common import ConstraintSpec, SampleContext
from query_generation.initial_query.config import CATEGORY_ORDER
from query_generation.user_profile import get_rule_ids, get_schedule_variant


def sample_visible_hard_categories(
    observable: dict[str, Any],
    derived: dict[str, Any],
    rng: random.Random,
) -> list[str]:
    weights = {key: 1.0 for key in CATEGORY_ORDER}
    rule_ids = get_rule_ids(derived)
    schedule_variant = get_schedule_variant(derived)

    if "hotel_value_first" in rule_ids:
        weights["hotel"] += 1.0
    if schedule_variant == "relaxed":
        weights["transport"] += 1.4
        weights["hotel"] += 0.8
    if any(rule_id.startswith("interest_") for rule_id in rule_ids):
        weights["food"] += 1.0
        weights["attraction"] += 1.0
    if derived.get("party_type") in {"family_with_child", "family_with_elder"}:
        weights["hotel"] += 1.0
        weights["transport"] += 0.8
    if "food" in set(observable.get("interest_tags", []) or []):
        weights["food"] += 1.4
    attraction_interests = {"nature", "park", "history", "museum", "art", "amusement", "landmark", "shopping"}
    if attraction_interests & set(observable.get("interest_tags", []) or []):
        weights["attraction"] += 0.8

    remaining = dict(weights)
    picked: list[str] = []
    for _ in range(2):
        keys = list(remaining.keys())
        vals = [remaining[key] for key in keys]
        chosen = rng.choices(keys, weights=vals, k=1)[0]
        picked.append(chosen)
        remaining.pop(chosen, None)
    return picked


def build_hard_constraint_spec(
    category: str,
    *,
    ctx: SampleContext,
    db: dict[str, list[dict[str, str]]],
    option: RouteOption,
    rng: random.Random,
) -> ConstraintSpec:
    if category == "transport":
        return build_transport_constraint(ctx, db, rng, option.mode)
    if category == "hotel":
        return build_hotel_constraint(db, rng)
    if category == "food":
        return build_restaurant_constraint(db, rng)
    if category == "attraction":
        return build_attraction_constraint(db, rng)
    raise ValueError(f"Unknown category: {category}")


def build_all_hard_constraint_specs(
    *,
    ctx: SampleContext,
    db: dict[str, list[dict[str, str]]],
    option: RouteOption,
    rng: random.Random,
) -> dict[str, ConstraintSpec]:
    return {
        category: build_hard_constraint_spec(category, ctx=ctx, db=db, option=option, rng=rng)
        for category in CATEGORY_ORDER
    }
