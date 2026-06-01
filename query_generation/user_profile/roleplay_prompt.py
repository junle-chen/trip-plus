"""Roleplay prompt used to render profile-aware initial user messages."""

from __future__ import annotations

import json
from typing import Any

from .derive import get_schedule_variant


ROLEPLAY_STYLE = {
    "P01": "A young traveler: direct, casual, and not overly formal.",
    "P02": "A couple traveler: gentle wording, with attention to atmosphere and experience.",
    "P03": "A family decision-maker who naturally considers children or elders.",
    "P04": "A practical family-with-child traveler who cares about rhythm and rest.",
    "P05": "A senior traveler or someone arranging for seniors: steady wording, avoiding unnecessary strain.",
    "P06": "A culture-focused traveler who naturally mentions representative local content.",
    "P07": "A budget-sensitive student traveler: plain-spoken and practical.",
    "P08": "A business traveler extending the trip: concise, efficiency-focused.",
    "P09": "A food-first traveler who proactively mentions what they want to eat.",
    "P10": "A nature-scenery traveler who notices outdoor conditions, views, and physical load.",
    "P11": "A friend-group traveler: casual and conversational.",
}


def _profile_expression_guidance(payload: dict[str, Any]) -> list[str]:
    derived = payload.get("derived_user_profile", {}) or {}
    rule_ids = set(derived.get("rule_ids", []) or [])
    schedule_variant = get_schedule_variant(derived)
    explicit_budget = payload.get("budget_requirement")
    constraints = payload.get("constraints", {})
    if isinstance(constraints, dict) and constraints.get("budget_requirement") is not None:
        explicit_budget = constraints.get("budget_requirement")
    guidance: list[str] = []

    def add(text: str) -> None:
        if text and text not in guidance:
            guidance.append(text)

    if "mobility_accessibility" in rule_ids:
        add("Use natural wording such as not too exhausting, less backtracking, or a smoother route to imply mobility preferences.")
    if schedule_variant == "relaxed":
        add("Express wanting a lighter pace, buffer time, and no rushing between places.")
    elif schedule_variant == "dense":
        add("Express that a somewhat compact itinerary is acceptable as long as it remains feasible.")
    if "hotel_value_first" in rule_ids:
        add("Show that lodging should not be too expensive and only needs to be clean and convenient.")
    if "budget_tight_cap" in rule_ids or "budget_guarded" in rule_ids or "meal_avoid_expensive" in rule_ids:
        if explicit_budget is None:
            add("You may reveal frugality or not-too-expensive preferences, but do not invent a specific budget.")
        else:
            add("If the user prompt gives a budget, preserve it exactly; do not create another amount from the profile.")
    effective_budget = (derived.get("derivation_context") or {}).get("upper_budget_per_person_day")
    try:
        high_budget_headroom = effective_budget is not None and float(effective_budget) >= 950
    except (TypeError, ValueError):
        high_budget_headroom = False
    if high_budget_headroom:
        add("Do not force a frugal persona; it is natural to prioritize experience, comfort, or efficiency.")
    if "weather_avoid_heat_exposure" in rule_ids:
        add("If the environment is hot, naturally mention not staying outdoors through the strongest sun.")
    if "weather_avoid_cold_exposure" in rule_ids:
        add("If the environment is cold, naturally mention not staying outside too long in the cold.")
    if "weather_need_backup" in rule_ids:
        add("If there is rain or extreme weather, mention wanting a more robust plan or a backup.")
    if any(rule_id.startswith("transport_") for rule_id in rule_ids):
        add("Express transport preferences conversationally, such as direct, not too early, not too late, or fewer transfers; do not write a rule list.")
    if any(rule_id.startswith("interest_") for rule_id in rule_ids):
        add("Mention at most one interest type or city-specific feature casually; do not list attractions or foods mechanically.")
    return guidance[:5]


def _compact_profile_context(payload: dict[str, Any]) -> str:
    observable_profile = dict(payload.get("observable_profile", {}) or {})
    observable_profile.pop("budget_range", None)
    explicit_budget = payload.get("budget_requirement")
    constraints = payload.get("constraints", {})
    if isinstance(constraints, dict) and constraints.get("budget_requirement") is not None:
        explicit_budget = constraints.get("budget_requirement")
    profile_context = {
        "persona": payload.get("persona", {}),
        "observable_profile": observable_profile,
        "latent_user_state": payload.get("derived_user_profile", {}),
        "profile_expression_guidance": _profile_expression_guidance(payload),
        "budget_guidance": (
            "The explicit budget for this trip must come only from budget_requirement in the user prompt; do not invent amounts from the long-term profile."
            if explicit_budget is not None
            else "The long-term profile may shape frugal or comfort-oriented wording, but do not state a specific budget amount."
        ),
    }
    if payload.get("profile_focus"):
        profile_context["profile_focus"] = payload.get("profile_focus")
    if payload.get("implicit_needs"):
        profile_context["implicit_needs"] = payload.get("implicit_needs")
    return json.dumps(profile_context, ensure_ascii=False, indent=2)


def build_user_roleplay_system_prompt(payload: dict[str, Any]) -> str:
    persona = payload["persona"]
    style_hint = ROLEPLAY_STYLE.get(persona["persona_id"], "Speak like a real traveler, not too formally.")
    profile_context = _compact_profile_context(payload)
    return (
        "You are not the travel assistant; you are the traveler writing a message to a travel-planning assistant.\n"
        "Speak from the user's point of view. Do not explain rules, summarize the payload, or write like a product manager.\n"
        f"Voice style: {style_hint}\n"
        "Use the following profile only to role-play the user and decide what they naturally care about. Do not recite field names:\n"
        f"{profile_context}\n"
        "Output principles:\n"
        "1. Say only what a real user would say. Do not turn every condition into a list or a coworker-style requirement summary.\n"
        "2. Selected constraints or explicit budgets in the user prompt are requirements spoken in this turn; preserve their meaning.\n"
        "3. derived_user_profile contains implicit needs/preferences derived from the profile. Convert only 1-3 of them into natural cues or tone; do not recite each rule.\n"
        "4. Make implicit needs inferable, not hard-coded. For example, say a parent has knee trouble and the route should not be exhausting, not minimize walking distance.\n"
        "5. Do not expose rule_id, planner_obligation, trigger, or evaluation_basis from derived_user_profile.\n"
        "6. If there is an explicit budget, preserve it. If not, do not invent an amount; only imply frugal or comfort-oriented tendencies.\n"
        "7. The message focus must reflect this persona, and different personas should clearly care about different things.\n"
        "8. If environmental_grounding is present, translate weather, temperature, weekend/holiday crowding, or similar factors into concerns a traveler would naturally mention.\n"
        "9. Casual, fragmented, slightly jumpy wording is allowed, like a real chat message.\n"
        "10. Do not explain every profile dimension. Leave one or two things implicit rather than writing a complete specification.\n"
        "11. Output only one English user message."
    )
