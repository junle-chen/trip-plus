"""Environment signals and hidden references used by initial and multi-turn queries."""

from __future__ import annotations

import random
from typing import Any

from query_generation.initial_query.config import (
    PRACTICAL_ENV_SIGNAL_TYPES,
    normalize_interaction_archetype,
)


def environment_reference(city_context: dict[str, Any], environmental_grounding: dict[str, Any]) -> dict[str, Any]:
    return {
        "city_tags": list(city_context.get("city_tags", []) or []),
        "city_tag_references": list(city_context.get("city_tag_references", []) or []),
        "feedback_triggers": list(city_context.get("feedback_triggers", []) or []),
        "generation_hints": list(city_context.get("generation_hints", []) or []),
        "evaluable_planning_checks": list(city_context.get("evaluable_planning_checks", []) or []),
        "date_tags": list(environmental_grounding.get("date_tags", []) or []),
        "weather_tags": list(environmental_grounding.get("weather_tags", []) or []),
        "notes": list(environmental_grounding.get("notes", []) or []),
    }


def environment_signals(
    *,
    observable: dict[str, Any],
    environmental_grounding: dict[str, Any],
    city_context: dict[str, Any],
) -> list[dict[str, str]]:
    signals: list[dict[str, str]] = []
    mobility_constraints = set(observable.get("mobility_constraints", []) or [])
    notes = list(environmental_grounding.get("notes", []) or [])
    holiday_tags = set(environmental_grounding.get("holiday_signals", {}).get("holiday_tags", []) or [])
    seasonal_advisories = set(city_context.get("seasonal_advisories", []) or [])
    city_tags = set(city_context.get("city_tags", []) or [])
    avg_temp_max = environmental_grounding.get("avg_temp_max_c")
    avg_temp_min = environmental_grounding.get("avg_temp_min_c")

    def _temp_at_most(value: Any, threshold: float) -> bool:
        return isinstance(value, (int, float)) and value <= threshold

    def _temp_at_least(value: Any, threshold: float) -> bool:
        return isinstance(value, (int, float)) and value >= threshold

    cold_by_weather = (
        _temp_at_most(avg_temp_min, 0)
        or _temp_at_most(avg_temp_max, 5)
        or ("cold_sensitive" in mobility_constraints and (_temp_at_most(avg_temp_min, 5) or _temp_at_most(avg_temp_max, 12)))
    )
    heat_by_weather = (
        _temp_at_least(avg_temp_max, 32)
        or ("heat_sensitive" in mobility_constraints and _temp_at_least(avg_temp_max, 28))
    )

    if (
        "winter_outdoor_exposure" in seasonal_advisories
        or cold_by_weather
        or any(
            any(marker in note.lower() for marker in ("winter", "outdoor exposure", "low temperature", "severe cold", "cold"))
            for note in notes
        )
    ):
        signals.append(
            {
                "type": "cold",
                "hint": "If it is very cold in winter, avoid stacking too many long outdoor periods together.",
                "reason": "Cold conditions make warmth and continuous outdoor exposure important.",
            }
        )
    if "winter_warm_escape" in seasonal_advisories:
        signals.append(
            {
                "type": "winter_warm_escape",
                "hint": "For a winter trip there, keep the outdoor parts comfortable and not too strenuous.",
                "reason": "Warm southern or coastal winter destinations are better suited to comfortable outdoor experiences.",
            }
        )
    if (
        "extreme_heat" in seasonal_advisories
        or heat_by_weather
        or any(
            any(marker in note.lower() for marker in ("high temperature", "hot weather", "extreme heat", "sun exposure"))
            for note in notes
        )
    ):
        signals.append(
            {
                "type": "heat",
                "hint": "Do not pack the hottest, sunniest part of the day too tightly; a smoother route would be better.",
                "reason": "In hot weather, midday sun exposure and physical load matter more.",
            }
        )
    if any(any(marker in note.lower() for marker in ("rain", "precipitation", "thunderstorm")) for note in notes):
        signals.append(
            {
                "type": "rain",
                "hint": "If it rains, the plan should still feel manageable and leave some room to adjust.",
                "reason": "Unstable weather makes a more robust route preferable.",
            }
        )
    if "high_altitude_adaptation" in seasonal_advisories or any("high altitude" in note.lower() for note in notes):
        signals.append(
            {
                "type": "altitude",
                "hint": "Right after arriving at a high-altitude destination, keep the early schedule steady rather than too intense.",
                "reason": "High-altitude cities require acclimatization and care with continuous outdoor intensity.",
            }
        )
    if "cross_border_region" in city_tags:
        signals.append(
            {
                "type": "cross_border",
                "hint": "For a cross-border destination such as Hong Kong, documents, local connectivity, and payment should be confirmed in advance.",
                "reason": "Cross-border destinations require extra attention to documents, entry, roaming/SIM cards, and local payment.",
            }
        )
    if {"local_currency_hkd", "octopus_transit_payment"} & city_tags:
        signals.append(
            {
                "type": "local_payment",
                "hint": "For Hong Kong, it is better to plan for HKD, small cash, and Octopus/local transit payment in advance.",
                "reason": "Local currency and transit payment cards affect transport, dining, and small purchases after arrival.",
            }
        )
    if "tibet_permit_sensitive" in city_tags:
        signals.append(
            {
                "type": "document_permit",
                "hint": "If Tibet-entry documents or special visitor status are involved, avoid making the itinerary too last-minute.",
                "reason": "Destinations such as Lhasa may require permits, document checks, and advance handling.",
            }
        )
    if "xinjiang_security_check_region" in city_tags:
        signals.append(
            {
                "type": "security_check",
                "hint": "In Xinjiang, document checks and security screening may be more frequent, so transfer timing should not be too tight.",
                "reason": "Document checks and security screening affect buffer time at stations, attractions, and hotels.",
            }
        )
    if "time_zone_shift_far_west" in city_tags:
        signals.append(
            {
                "type": "time_shift",
                "hint": "In Urumqi, local daily rhythm and sunset are later than in eastern China, so do not copy an eastern-city schedule directly.",
                "reason": "Far-western regions have noticeable rhythm/daylight shifts that affect departures, meals, and evening activities.",
            }
        )
    if {"real_name_reservation_city", "museum_reservation_city", "theme_park_reservation_city"} & city_tags:
        signals.append(
            {
                "type": "reservation",
                "hint": "Popular museums and attractions should be reserved ahead of time rather than left for a last-minute decision.",
                "reason": "Real-name reservations, capacity limits, or popular ticketing can affect core-attraction feasibility.",
            }
        )
    if {"ferry_island_transfer", "island_or_peninsula_transfer"} & city_tags:
        signals.append(
            {
                "type": "ferry_transfer",
                "hint": "If the route involves an island or ferry, leave extra time and do not place it too close to return transport.",
                "reason": "Ferries, ports, and island transfers are more affected by weather, queues, and schedules.",
            }
        )
    if {"steep_walk_city", "mountain_transfer_city", "old_town_walk_city"} & city_tags:
        signals.append(
            {
                "type": "walk_transfer",
                "hint": "In this kind of city, walking and slopes can be tiring, so avoid too much detouring.",
                "reason": "Mountain cities, old towns, and water towns can make walking, transfer time, and senior/child comfort more demanding.",
            }
        )
    high_crowd_attractions = list(city_context.get("high_crowd_attractions", []) or [])
    high_queue_attractions = list(city_context.get("high_queue_attractions", []) or [])
    popularity_counts = dict(city_context.get("popularity_tag_counts", {}) or {})
    queue_sensitive = bool({"queueing", "overpacked_schedule"} & set(observable.get("hate_tags", []) or []))
    note_text = " ".join(str(note).lower() for note in notes)
    crowd_window = bool(holiday_tags or any(token in note_text for token in ("weekend", "holiday", "crowd", "queue")))
    if crowd_window or (queue_sensitive and (high_crowd_attractions or high_queue_attractions)):
        example_names = high_queue_attractions[:2] or high_crowd_attractions[:2]
        example_text = " such as " + ", ".join(example_names) if example_names else ""
        popularity_reason = (
            f" The destination has high-crowd or queue-risk attractions{example_text}."
            if example_names or popularity_counts
            else ""
        )
        signals.append(
            {
                "type": "crowd",
                "hint": f"Those dates may be crowded{example_text}, so avoid filling the plan with places that are packed or queue-heavy.",
                "reason": "Weekends, holidays, or popular POIs are more likely to involve crowds and queues." + popularity_reason,
            }
        )
    if {"cold_sensitive", "heat_sensitive"} & mobility_constraints:
        for signal in signals:
            signal["reason"] += " The party is more sensitive to physical comfort."
    return signals


def should_surface_environment(
    *,
    archetype: str,
    observable: dict[str, Any],
    signals: list[dict[str, str]],
    rng: random.Random,
) -> bool:
    if not signals:
        return False
    if any(signal.get("type") in {"cross_border", "document_permit", "security_check", "local_payment"} for signal in signals):
        return True
    probability = 0.10
    archetype = normalize_interaction_archetype(archetype)
    if archetype in {"environment_driven_replanning", "long_horizon_alignment"}:
        probability += 0.10
    mobility_constraints = set(observable.get("mobility_constraints", []) or [])
    if {"cold_sensitive", "heat_sensitive", "pregnant", "wheelchair", "knee_issue"} & mobility_constraints:
        probability += 0.12
    if any(signal["type"] in {"cold", "heat", "altitude"} for signal in signals):
        probability += 0.10
    return rng.random() < min(probability, 0.55)


def select_environment_hints(signals: list[dict[str, str]], triggered: bool) -> list[dict[str, str]]:
    if not triggered:
        return []
    practical = [signal for signal in signals if signal.get("type") in PRACTICAL_ENV_SIGNAL_TYPES]
    return (practical or signals)[:1]
