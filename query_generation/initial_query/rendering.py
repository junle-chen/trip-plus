"""Visible English user-query rendering for initial-query records."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from agent.call_llm import call_llm
from query_generation.common import visible_weekday
from query_generation.user_profile import build_user_roleplay_system_prompt
from query_generation.initial_query.visibility.checks import query_passes_render_sanity
from query_generation.initial_query.visibility.repair import normalize_query_punctuation


def build_initial_render_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Extract only the fields the renderer may expose in the initial request."""
    meta = record["meta_info"]
    selected_constraints = []
    for item in meta["t0_structure"]["visible_constraint_payloads"]:
        selected_constraints.append(
            {
                "category": item["category"],
                "category_label": item["category_label"],
                "constraint_key": item["constraint_key"],
                "visible_hint": item["visible_hint"],
            }
        )
    return {
        "persona": meta["persona"],
        "interaction_archetype": meta["interaction_archetype"],
        "trip_frame": {
            "org": meta["org"],
            "dest": meta["dest"][0],
            "days": meta["days"],
            "depart_date": meta["depart_date"],
            "return_date": meta["return_date"],
            "people_number": meta["people_number"],
            "room_number": meta["room_number"],
            "depart_weekday": visible_weekday(meta["depart_weekday"]),
        },
        "observable_profile": {
            "party_composition": meta["observable_profile"].get("party_composition", {}),
        },
        "derived_user_profile": {
            "party_type": meta["user_profile"].get("party_type"),
            "rules_reserved_for_later": True,
        },
        "selected_category_constraints": selected_constraints,
        "budget_requirement": meta["hard_constraints"].get("budget_constraint"),
    }


def build_initial_render_prompt(payload: dict[str, Any]) -> str:
    """Build the user-message rendering prompt for optional LLM wording."""
    return (
        "Simulate a real English-speaking traveler writing the first message to a travel-planning assistant.\n"
        "This is the initial request. Real users do not state every need at once, so keep the message restrained.\n"
        "The initial request should fully state the trip frame and naturally express 1-2 explicit database-verifiable hard constraints. "
        "Do not surface later-turn material such as weather, city feel, documents, local payment, reservations, ferries, pacing comfort, or recovery needs.\n"
        "The result should read like a person casually describing a trip idea in chat, not filling out a form.\n\n"
        "Style:\n"
        "Blend departure details, party information, and a small number of explicit requirements into one natural message.\n"
        "Do not organize it as transportation/hotel/food/attractions sections, and do not expand every need into its own sentence.\n"
        "Some conversational looseness is fine, such as giving the reason for the trip and then adding one or two hard requirements.\n"
        "\n"
        "Profile adaptation:\n"
        "In this turn, use the profile only for voice and party wording. Do not prematurely mention stamina, heat/cold sensitivity, relaxed pace, reduced walking, or rest needs.\n"
        "If the party includes parents, seniors, or children, it is fine to mention traveling with them, but do not automatically add care requirements.\n"
        "Do not say 'according to the user profile', 'my preference is', or expose labels such as mobility_accessibility, budget_guarded, or evaluation_basis.\n\n"
        "Rules:\n"
        "1. Preserve origin, destination, departure date, return date, weekday, trip length, party size, and room count.\n"
        "   Departure and return dates must include the four-digit year; include the weekday naturally in the date phrase.\n"
        "2. The only explicit hard constraints in this initial request are `selected_category_constraints` and `budget_requirement`; preserve their meaning without exposing category names or field names.\n"
        "   Comparative or range terms such as cheapest, highest-rated, closest, all, required name, time window, or price range must keep their original force.\n"
        "3. If `budget_requirement` is present, mention only its min_budget/max_budget. If absent, do not invent a budget.\n"
        "4. Do not proactively mention cold/heat, altitude, crowding, relaxed pace, rest, reduced walking, or backup plans in the initial request; leave those for later turns or environment changes.\n"
        "5. Do not proactively mention practical city reminders such as visas, permits, local currency, transit cards, SIM cards, real-name reservations, ferries, security checks, or document handling.\n"
        "6. Do not turn destination tags into user worries such as cold weather, altitude discomfort, summer heat, or strict security.\n"
        "7. Do not invent new explicit constraints, especially new dates, party size, budgets, must-visit places, or hotel requirements.\n"
        "8. Avoid checklist wording and section labels such as transportation, hotels, food, attractions, first, second, or requirements.\n"
        "9. Casual phrasing is allowed, such as 'I'm thinking about...', 'also', 'if it fits', or 'nothing too fancy'.\n"
        "10. Let the profile affect wording and tradeoffs, but do not make the query look like a profile display. Do not cover budget, hotel, transport, food, attractions, and weather all in one message.\n"
        "11. selected_category_constraints are visible hard constraints for this turn. Do not weaken cheapest, highest-rated, closest, must-visit, required time window, or price range into optional preferences.\n"
        "12. Usually write one paragraph, about 80-180 English words.\n"
        "13. Output only the user message text. Do not output JSON or explanations.\n\n"
        f"Input payload:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def llm_render_initial_query(record: dict[str, Any], *, model: str) -> str:
    """Render one visible initial query with the configured LLM."""
    payload = build_initial_render_payload(record)
    system_prompt = build_user_roleplay_system_prompt(payload) + (
        "\n\nAdditional initial-request constraint: explicitly express only trip_frame, "
        "selected_category_constraints, and budget_requirement. User profile, city tags, and environment signals "
        "may affect voice only; they must not become stated requirements in this turn. Do not mention cold/heat, "
        "altitude, relaxed pace, reduced walking, rest, or backup plans."
    )
    prompt = build_initial_render_prompt(payload)
    response = call_llm(
        config_name=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        request_overrides={"temperature": 0.85, "max_tokens": 320},
    )
    return str(response.choices[0].message.content).strip()


def _llm_render_or_fallback_initial_query(record: dict[str, Any], *, model: str) -> str:
    query = llm_render_initial_query(record, model=model)
    if query_passes_render_sanity(query, record["meta_info"]):
        return query
    return fallback_render_initial_query(record)


def render_initial_query_candidates(
    candidates: list[dict[str, Any]],
    *,
    model: str,
    skip_llm: bool,
    workers: int,
) -> None:
    """Render candidate records in place, using deterministic fallback when needed."""
    if skip_llm:
        for candidate in candidates:
            candidate["record"]["query"] = fallback_render_initial_query(candidate["record"])
        return

    workers = max(1, min(int(workers or 1), len(candidates)))
    if workers == 1:
        for candidate in candidates:
            candidate["record"]["query"] = _llm_render_or_fallback_initial_query(candidate["record"], model=model)
        return

    future_to_candidate: dict[Any, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for candidate in candidates:
            future = executor.submit(_llm_render_or_fallback_initial_query, candidate["record"], model=model)
            future_to_candidate[future] = candidate

        for future in as_completed(future_to_candidate):
            candidate = future_to_candidate[future]
            candidate["record"]["query"] = future.result()


def fallback_render_initial_query(record: dict[str, Any]) -> str:
    """Render an English query without calling an LLM."""
    meta = record["meta_info"]
    people_text = "1 traveler" if meta["people_number"] == 1 else f"{meta['people_number']} travelers"
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_idx = int(meta.get("depart_weekday") or 0) - 1
    depart_weekday = weekday_names[weekday_idx] if 0 <= weekday_idx < 7 else ""
    depart_text = meta["depart_date"] + (f" ({depart_weekday})" if depart_weekday else "")
    trip = (
        f"I want to travel from {meta['org']} to {meta['dest'][0]} for {meta['days']} days, "
        f"departing on {depart_text} and returning on {meta['return_date']}. "
        f"There will be {people_text}, and please plan for {meta['room_number']} room(s)."
    )
    visible_specs = meta["t0_structure"]["visible_constraint_payloads"]
    visible_hints = [
        str(item["visible_hint"]).strip().rstrip(".;, ")
        for item in visible_specs
        if str(item.get("visible_hint", "")).strip()
    ]
    primary_needs = " " + " ".join(f"{hint}." for hint in visible_hints) if visible_hints else ""
    tail_bits = []
    if "budget_constraint" in meta["hard_constraints"]:
        budget = meta["hard_constraints"]["budget_constraint"]
        if "min_budget" in budget and "max_budget" in budget:
            tail_bits.append(f"I would like the total budget to stay around {budget['min_budget']}-{budget['max_budget']} RMB.")
        elif "max_budget" in budget:
            tail_bits.append(f"Please keep the total budget under {budget['max_budget']} RMB if possible.")
    return normalize_query_punctuation(trip + primary_needs + (" " + " ".join(tail_bits) if tail_bits else ""))
