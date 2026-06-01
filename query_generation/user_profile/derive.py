"""Deterministic hidden-rule derivation from observable traveler profiles."""

from __future__ import annotations

from typing import Any


Rule = dict[str, Any]


def _party_counts(observable: dict[str, Any]) -> tuple[int, int, int]:
    party = observable.get("party_composition", {})
    adults = int(party.get("adults", 0) or 0)
    children = len(list(party.get("children", []) or []))
    elders = len(list(party.get("elders", []) or []))
    return adults, children, elders


def derive_party_type(observable: dict[str, Any]) -> str:
    adults, children, elders = _party_counts(observable)
    if children > 0:
        return "family_with_child"
    if elders > 0:
        if adults == 0 and elders == 1:
            return "solo"
        return "family_with_elder"
    if adults <= 1:
        return "solo"
    if adults == 2:
        return "couple"
    return "friends"


def _make_rule(
    rule_id: str,
    category: str,
    trigger: str,
    planner_obligation: str,
    evaluation_basis: list[str],
    **extra: Any,
) -> Rule:
    rule = {
        "rule_id": rule_id,
        "category": category,
        "trigger": trigger,
        "planner_obligation": planner_obligation,
        "evaluation_basis": evaluation_basis,
    }
    rule.update({key: value for key, value in extra.items() if value not in (None, [], {})})
    return rule


def _append_unique(values: list[Any], new_values: list[Any]) -> list[Any]:
    for value in new_values:
        if value not in values:
            values.append(value)
    return values


def _merge_profile_rules(rules: list[Rule]) -> list[Rule]:
    merged: list[Rule] = []
    by_id: dict[str, Rule] = {}
    for rule in rules:
        rule_id = str(rule.get("rule_id") or "")
        if rule_id not in by_id:
            copied = dict(rule)
            copied["evaluation_basis"] = list(rule.get("evaluation_basis") or [])
            copied["preference_variants"] = list(rule.get("preference_variants") or [])
            if "preference_variant" in copied:
                copied["preference_variants"] = _append_unique(copied["preference_variants"], [copied.pop("preference_variant")])
            by_id[rule_id] = copied
            merged.append(copied)
            continue
        existing = by_id[rule_id]
        triggers = [part.strip() for part in str(existing.get("trigger") or "").split(";") if part.strip()]
        trigger = str(rule.get("trigger") or "").strip()
        if trigger and trigger not in triggers:
            triggers.append(trigger)
        existing["trigger"] = "; ".join(triggers)
        new_obligation = str(rule.get("planner_obligation") or "").strip()
        if new_obligation and new_obligation not in str(existing.get("planner_obligation") or ""):
            existing["planner_obligation"] = f"{existing.get('planner_obligation', '').rstrip()} {new_obligation}".strip()
        existing["evaluation_basis"] = _append_unique(existing.get("evaluation_basis", []), list(rule.get("evaluation_basis") or []))
        variants = list(rule.get("preference_variants") or [])
        if rule.get("preference_variant"):
            variants.append(rule["preference_variant"])
        existing["preference_variants"] = _append_unique(existing.get("preference_variants", []), variants)
        for key in ("interest_tags", "transport_preferences", "source_profile_values"):
            existing[key] = _append_unique(list(existing.get(key) or []), list(rule.get(key) or []))
    for rule in merged:
        for key in ("preference_variants", "interest_tags", "transport_preferences", "source_profile_values"):
            if key in rule and not rule[key]:
                rule.pop(key, None)
    return merged


def _budget_effective_per_person_day(
    observable: dict[str, Any],
    *,
    days: int,
    dest_price_level: str,
) -> float:
    adults, children, elders = _party_counts(observable)
    party_size = max(1, adults + children + elders)
    budget_range = observable.get("budget_range", [0, 0]) or [0, 0]
    upper_budget = float(budget_range[1])
    effective = upper_budget / max(1, days) / party_size
    effective /= {"low": 0.9, "medium": 1.0, "high": 1.2}.get(dest_price_level, 1.0)
    if str(observable.get("accommodation_style", "")).strip() == "luxury":
        effective *= 1.15
    return effective


def derive_profile_rules(
    observable: dict[str, Any],
    *,
    days: int,
    dest_price_level: str = "medium",
) -> dict[str, Any]:
    """Derive rule-based latent preferences without low/medium/high labels."""
    adults, children, elders = _party_counts(observable)
    party_type = derive_party_type(observable)
    constraints = set(observable.get("mobility_constraints", []) or [])
    hate_tags = set(observable.get("hate_tags", []) or [])
    interest_tags = set(observable.get("interest_tags", []) or [])
    rest_preferences = set(observable.get("rest_preferences", []) or [])
    transport_preferences = set(observable.get("transport_preferences", []) or [])
    physical_rules = set(observable.get("physical_rules", []) or [])
    schedule_seed_rules = set(observable.get("schedule_rules", []) or [])
    accommodation_style = str(observable.get("accommodation_style", "comfort") or "comfort")
    effective_budget = _budget_effective_per_person_day(
        observable,
        days=days,
        dest_price_level=dest_price_level,
    )

    rules: list[Rule] = []

    if effective_budget < 500:
        rules.append(
            _make_rule(
                "budget_tight_cap",
                "budget",
                f"upper_budget_per_person_day={effective_budget:.1f} < 500 after destination price adjustment",
                "Surface explicit budget more often and prefer lower-cost hotels, restaurants, and transport.",
                ["total_cost", "hotel.price", "restaurant.price_per_person", "transport.price"],
            )
        )
    elif effective_budget < 950:
        rules.append(
            _make_rule(
                "budget_guarded",
                "budget",
                f"500 <= upper_budget_per_person_day={effective_budget:.1f} < 950 after destination price adjustment",
                "Keep cost visible but allow moderate comfort tradeoffs.",
                ["total_cost", "hotel.price", "restaurant.price_per_person", "transport.price"],
            )
        )
    else:
        # High budget headroom is kept in derivation_context, not as a scored
        # soft-preference rule.
        pass

    if accommodation_style == "budget":
        rules.append(
            _make_rule(
                "hotel_value_first",
                "hotel",
                "accommodation_style=budget",
                "Prefer lower hotel prices unless an explicit hotel hard constraint says otherwise.",
                ["hotel.price", "hotel.score"],
            )
        )

    if (
        elders
        or children
        or "avoid_long_walk" in physical_rules
        or {"knee_issue", "stroller"} & constraints
    ):
        rules.append(
            _make_rule(
                "mobility_accessibility",
                "mobility",
                "elder/child/avoid_long_walk/knee_issue/stroller observed",
                "Reduce long walking segments and avoid high-intensity consecutive transfers.",
                ["local_transport.distance_meters", "duration_minutes", "daily attraction count"],
                preference_variant="walking_sensitive",
            )
        )
    if "long_local_transfer" in hate_tags or "avoid_transfer" in transport_preferences or {"knee_issue", "stroller"} & constraints:
        rules.append(
            _make_rule(
                "mobility_accessibility",
                "mobility",
                "long_local_transfer disliked, avoid_transfer preferred, or mobility constraint observed",
                "Prefer geographically coherent routes and fewer transfers.",
                ["local_transport.duration_minutes", "distance_meters", "route continuity"],
                preference_variant="route_transfer_sensitive",
            )
        )

    if "heat_sensitive" in constraints:
        rules.append(
            _make_rule(
                "weather_avoid_heat_exposure",
                "weather",
                "heat_sensitive observed",
                "Avoid long outdoor exposure in high-temperature periods.",
                ["weather.temperature_max_c", "outdoor attraction timing", "local_transport mode"],
            )
        )
    if "cold_sensitive" in constraints:
        rules.append(
            _make_rule(
                "weather_avoid_cold_exposure",
                "weather",
                "cold_sensitive observed",
                "Avoid long outdoor exposure in cold periods.",
                ["weather.temperature_min_c", "outdoor attraction timing", "local_transport mode"],
            )
        )
    if "extreme_weather" in hate_tags:
        rules.append(
            _make_rule(
                "weather_need_backup",
                "weather",
                "extreme_weather disliked",
                "Prefer weather-robust routing and indoor alternatives when weather risk appears.",
                ["weather_code", "precipitation_mm", "indoor/outdoor attraction mix"],
            )
        )

    if (
        "relaxed" in schedule_seed_rules
        or {"midday_rest", "late_start", "afternoon_low_intensity"} & rest_preferences
        or "overpacked_schedule" in hate_tags
    ):
        rules.append(
            _make_rule(
                "schedule_pacing",
                "schedule",
                "relaxed pace, rest preference, or overpacked_schedule disliked",
                "Avoid overpacked days and include rest or buffer time.",
                ["daily attraction count", "rest blocks", "first activity start time"],
                preference_variant="relaxed",
            )
        )
    elif "dense" in schedule_seed_rules and "long_walk_ok" in physical_rules and not (children or elders):
        rules.append(
            _make_rule(
                "schedule_pacing",
                "schedule",
                "dense schedule seed and long_walk_ok observed without child/elder care load",
                "A denser itinerary is acceptable if travel times and opening hours remain feasible.",
                ["daily attraction count", "opening hours", "transport duration"],
                preference_variant="dense",
            )
        )
    else:
        rules.append(
            _make_rule(
                "schedule_pacing",
                "schedule",
                "default moderate pacing",
                "Keep daily activity load feasible without forcing extra rest.",
                ["daily attraction count", "transport duration", "opening hours"],
                preference_variant="moderate",
            )
        )

    transport_rule_map = {
        "prefer_direct": ("transport_avoid_transfer", "Prefer direct intercity options.", ["transfers", "duration"]),
        "avoid_red_eye": ("transport_avoid_red_eye", "Avoid red-eye or very late-night transport.", ["departure_time", "arrival_time"]),
        "prefer_train": ("transport_prefer_train", "Prefer train when feasible.", ["transport.mode", "price", "duration"]),
        "prefer_flight": ("transport_prefer_flight", "Prefer flight when feasible.", ["transport.mode", "price", "duration"]),
        "avoid_transfer": ("transport_avoid_transfer", "Avoid transfer-heavy transport.", ["transfers", "duration"]),
        "avoid_early_departure": ("transport_avoid_early_departure", "Avoid very early departure.", ["departure_time"]),
        "avoid_late_arrival": ("transport_avoid_late_arrival", "Avoid very late arrival.", ["arrival_time"]),
    }
    for preference in sorted(transport_preferences):
        if preference in transport_rule_map:
            rule_id, obligation, basis = transport_rule_map[preference]
            rules.append(
                _make_rule(
                    rule_id,
                    "transport",
                    f"transport_preferences includes {preference}",
                    obligation,
                    basis,
                    transport_preferences=[preference],
                )
            )

    interest_rule_map = {
        "food": ("interest_local_food", "Prefer local or high-rated restaurants.", ["restaurant.cuisine", "rating", "price_per_person"]),
        "nature": ("interest_outdoor_nature", "Prefer nature, park, or low-intensity outdoor attractions.", ["attraction_type", "rating"]),
        "park": ("interest_outdoor_nature", "Prefer nature, park, or low-intensity outdoor attractions.", ["attraction_type", "ticket_price"]),
        "history": ("interest_culture", "Prefer historical, cultural, museum, or memorial venues.", ["attraction_type", "rating"]),
        "museum": ("interest_culture", "Prefer historical, cultural, museum, or memorial venues.", ["attraction_type", "opening_time"]),
        "art": ("interest_art", "Prefer art museums or galleries.", ["attraction_type", "opening_time"]),
        "shopping": ("interest_shopping", "Prefer shopping streets or commercial districts.", ["attraction_type", "opening_time"]),
        "landmark": ("interest_landmark", "Prefer representative city landmarks.", ["attraction_type", "rating"]),
        "amusement": ("interest_amusement", "Prefer amusement or entertainment attractions.", ["attraction_type", "ticket_price"]),
    }
    for interest in sorted(interest_tags):
        if interest in interest_rule_map:
            rule_id, obligation, basis = interest_rule_map[interest]
            rules.append(
                _make_rule(
                    rule_id,
                    "interest",
                    f"interest_tags includes {interest}",
                    obligation,
                    basis,
                    interest_tags=[interest],
                )
            )

    if "expensive_meal" in hate_tags:
        rules.append(
            _make_rule(
                "meal_avoid_expensive",
                "food",
                "hate_tags includes expensive_meal",
                "Prefer lower or moderate restaurant price_per_person unless a restaurant hard constraint overrides it.",
                ["restaurant.price_per_person"],
            )
        )
    if "red_eye_transport" in hate_tags:
        rules.append(
            _make_rule(
                "transport_avoid_red_eye",
                "transport",
                "hate_tags includes red_eye_transport",
                "Avoid late-night or red-eye intercity segments.",
                ["departure_time", "arrival_time"],
            )
        )

    rules = _merge_profile_rules(rules)
    return {
        "party_type": party_type,
        "rule_ids": [rule["rule_id"] for rule in rules],
        "rules": rules,
        "derivation_context": {
            "days": days,
            "dest_price_level": dest_price_level,
            "upper_budget_per_person_day": round(effective_budget, 1),
            "party_size": adults + children + elders,
            "physical_rule_seeds": sorted(physical_rules),
            "schedule_rule_seeds": sorted(schedule_seed_rules),
        },
    }


def get_rule_ids(profile: dict[str, Any]) -> set[str]:
    return set(profile.get("rule_ids", []) or [])


def get_schedule_variant(profile: dict[str, Any]) -> str:
    for rule in profile.get("rules", []) or []:
        if not isinstance(rule, dict) or rule.get("rule_id") != "schedule_pacing":
            continue
        variants = rule.get("preference_variants") or []
        if isinstance(variants, str):
            variants = [variants]
        variant_set = {str(item).strip() for item in variants if str(item).strip()}
        if "relaxed" in variant_set:
            return "relaxed"
        if "dense" in variant_set:
            return "dense"
    return "moderate"
