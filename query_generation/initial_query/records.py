"""Structured record assembly for one English initial query."""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any

from query_generation.city_database import RouteOption
from query_generation.common import ConstraintSpec, SampleContext, common_room_number
from query_generation.initial_query.constraints.explicit_trip import ensure_explicit_trip_hard_constraints
from query_generation.initial_query.config import category_label
from query_generation.initial_query.constraints.budget import (
    apply_profile_budget_policy,
    budget_should_surface,
    build_budget_constraint,
    exclude_budget_soft_rules_when_hard_budget,
    repair_budget_transport_estimate,
)
from query_generation.initial_query.constraints.environment import (
    environment_reference,
    environment_signals,
    select_environment_hints,
    should_surface_environment,
)
from query_generation.initial_query.constraints.hard import (
    build_all_hard_constraint_specs,
    sample_visible_hard_categories,
)
from query_generation.user_profile import (
    ObservableProfileSampler,
    derive_profile_rules,
    get_rule_ids,
    get_schedule_variant,
)
from query_generation.initial_query.environment_context import (
    derive_city_context,
    derive_environmental_grounding,
)


def build_sample_context(sample_id: str, option: RouteOption, observable: dict[str, Any]) -> SampleContext:
    """Convert a sampled route and visible profile into shared query context."""
    party = observable.get("party_composition", {})
    people_number = (
        int(party.get("adults", 0) or 0)
        + len(list(party.get("children", []) or []))
        + len(list(party.get("elders", []) or []))
    )
    return SampleContext(
        sample_id=sample_id,
        org=option.origin_city,
        dest=option.dest_city,
        days=option.days,
        depart_date=option.depart_date,
        return_date=option.return_date,
        people_number=people_number,
        room_number=common_room_number(people_number),
        depart_weekday=datetime.strptime(option.depart_date, "%Y-%m-%d").weekday() + 1,
    )


def visible_constraint_payload(category: str, spec: ConstraintSpec) -> dict[str, Any]:
    """Build the visible constraint payload used by rendering and evaluation metadata."""
    constraint = dict(spec.data)
    query_bullet = str(spec.query_bullet or "").strip()
    context = str(constraint.get("constraint_context") or query_bullet).strip()
    if context:
        constraint["constraint_context"] = context
    return {
        "category": category,
        "category_label": category_label(category),
        "constraint_key": spec.key,
        "visible_hint": str(spec.visible_hint or "").strip(),
        "query_bullet": query_bullet,
        "constraint": constraint,
    }


def query_signature(record: dict[str, Any]) -> tuple[Any, ...]:
    """Return the deduplication key for the structured record before rendering."""
    meta = record["meta_info"]
    observable = meta["observable_profile"]
    profile = meta["user_profile"]
    soft_city = tuple(meta["soft_constraints"].get("city_soft_wishes", []) or [])
    return (
        meta["org"],
        meta["dest"][0],
        meta["depart_date"],
        tuple(meta["t0_structure"]["selected_categories"]),
        meta["t0_structure"]["budget_triggered"],
        meta["t0_structure"]["environment_triggered"],
        meta["interaction_archetype"],
        profile["party_type"],
        next((rule_id for rule_id in profile.get("rule_ids", []) if rule_id.startswith("budget_")), ""),
        next(iter(observable.get("interest_tags", []) or []), ""),
        soft_city[:1],
    )


def _city_soft_wishes(city_context: dict[str, Any], selected_categories: list[str], rng: random.Random) -> list[str]:
    wishes: list[str] = []
    if "attraction" not in selected_categories:
        scenery = list(city_context.get("signature_scenery", []) or [])
        if scenery:
            picked = rng.choice(scenery[: min(3, len(scenery))])
            wishes.append(f"I recently saw {picked}; if it fits the route, I would like to see it.")
    if "food" not in selected_categories:
        cuisines = list(city_context.get("signature_cuisines", []) or [])
        if cuisines:
            picked = cuisines[: min(2, len(cuisines))]
            wishes.append(f"I would also like to try some local specialties, such as {', '.join(picked)}.")
    return wishes[:1]


def _profile_soft_wish(observable: dict[str, Any], derived: dict[str, Any], selected_categories: list[str]) -> str | None:
    mobility_constraints = set(observable.get("mobility_constraints", []) or [])
    physical_rules = set(observable.get("physical_rules", []) or [])
    rest_preferences = set(observable.get("rest_preferences", []) or [])
    hate_tags = set(observable.get("hate_tags", []) or [])
    rule_ids = get_rule_ids(derived)
    schedule_variant = get_schedule_variant(derived)

    if "avoid_long_walk" in physical_rules:
        return "Stamina is average, so keep the route from being too tiring or walk-heavy."
    if schedule_variant == "relaxed":
        return "Please do not make the overall schedule too packed; a smoother pace is better."
    if "midday_rest" in rest_preferences:
        return "It would be better to leave some breathing room in the middle of the day."
    if "long_local_transfer" in hate_tags:
        return "Avoid overly long intracity transfers; keep the route smooth."
    if {"knee_issue", "wheelchair", "pregnant"} & mobility_constraints:
        return "Keep the route easy and avoid unnecessary back-and-forth movement."
    if "hotel_value_first" in rule_ids and "hotel" not in selected_categories:
        return "Please keep lodging reasonably priced; clean and convenient is enough."
    return None


def build_initial_query_record(
    *,
    sample_id: str,
    option: RouteOption,
    db: dict[str, list[dict[str, str]]],
    city_db_root: Path,
    interaction_archetype: str,
    sampler: ObservableProfileSampler,
    rng: random.Random,
) -> dict[str, Any]:
    """Assemble one structured single-turn record before visible query rendering."""
    profile_sample = sampler.sample(rng)
    observable = profile_sample["observable"]
    ctx = build_sample_context(sample_id, option, observable)
    derived = derive_profile_rules(observable, days=ctx.days, dest_price_level="medium")

    selected_categories = sample_visible_hard_categories(observable, derived, rng)
    all_specs = build_all_hard_constraint_specs(ctx=ctx, db=db, option=option, rng=rng)
    visible_specs = {category: all_specs[category] for category in selected_categories}

    budget_triggered = budget_should_surface(derived, interaction_archetype, rng)
    support_specs = list(all_specs.values())
    hard_constraints = {
        visible_specs[category].key: visible_specs[category].data
        for category in visible_specs
    }
    if budget_triggered:
        budget_spec = build_budget_constraint(ctx, support_specs, rng)
        budget_data = repair_budget_transport_estimate(
            budget_spec.data,
            ctx=ctx,
            db=db,
            mode=option.mode,
        )
        hard_constraints["budget_constraint"] = apply_profile_budget_policy(budget_data, derived, rng)
        derived = exclude_budget_soft_rules_when_hard_budget(derived, hard_constraints)

    city_context = derive_city_context(
        db,
        option.depart_date,
        city_name=option.dest_city,
        city_folder=option.dest_folder,
    )
    environmental_grounding = derive_environmental_grounding(
        city_db_root=city_db_root,
        option=option,
        city_context=city_context,
    )
    signals = environment_signals(
        observable=observable,
        environmental_grounding=environmental_grounding,
        city_context=city_context,
    )
    environment_triggered = should_surface_environment(
        archetype=interaction_archetype,
        observable=observable,
        signals=signals,
        rng=rng,
    )

    soft_constraints = {
        "city_soft_wishes": _city_soft_wishes(city_context, selected_categories, rng),
        "profile_soft_wish": _profile_soft_wish(observable, derived, selected_categories),
        "environment_hints": select_environment_hints(signals, environment_triggered),
    }
    visible_payloads = [
        visible_constraint_payload(category, visible_specs[category])
        for category in selected_categories
    ]
    for payload in visible_payloads:
        hard_constraints[payload["constraint_key"]] = payload.get(
            "constraint",
            hard_constraints.get(payload["constraint_key"], {}),
        )

    record = {
        "id": sample_id,
        "query": "",
        "meta_info": {
            "org": ctx.org,
            "dest": [ctx.dest],
            "days": ctx.days,
            "depart_date": ctx.depart_date,
            "depart_weekday": ctx.depart_weekday,
            "return_date": ctx.return_date,
            "people_number": ctx.people_number,
            "room_number": ctx.room_number,
            "solution_status": "sat",
            "persona": {
                "persona_id": profile_sample["persona_id"],
                "persona_name": profile_sample["persona_name"],
            },
            "interaction_archetype": interaction_archetype,
            "observable_profile": observable,
            "user_profile": derived,
            "hard_constraints": hard_constraints,
            "soft_constraints": soft_constraints,
            "city_context": city_context,
            "environmental_grounding": environmental_grounding,
            "environment_reference": environment_reference(city_context, environmental_grounding),
            "t0_structure": {
                "selected_categories": selected_categories,
                "selected_category_labels": [category_label(item) for item in selected_categories],
                "budget_triggered": budget_triggered,
                "environment_triggered": environment_triggered,
                "visible_constraint_payloads": visible_payloads,
                "route_mode": option.mode,
            },
            "route_mode": option.mode,
        },
    }
    ensure_explicit_trip_hard_constraints(record["meta_info"])
    return record
