"""Build structured multi-turn records from single-turn seed records."""

from __future__ import annotations

import copy
import random
import re
from pathlib import Path
from typing import Any

from query_generation.common import BASE_DIR, common_room_number, load_csv, load_json, safe_float
from query_generation.initial_query.config import (
    INTERACTION_ARCHETYPES,
    INTERACTION_ARCHETYPE_WEIGHTS,
    normalize_interaction_archetype,
)
from query_generation.multiturn_query.config import CHECK_INITIAL, CHECK_PRESERVE


def _sample_suffix(sample_id: object) -> str:
    text = str(sample_id or "").strip()
    text = text[3:] if text.startswith("id_") else text
    match = re.fullmatch(r"(?:mt_)?single_(\d+)", text)
    return match.group(1) if match else text


def _load_sample_db(base_query_id: object, database_root: Path) -> dict[str, list[dict[str, str]]]:
    suffix = _sample_suffix(base_query_id)
    candidates = [
        database_root / f"id_{suffix}",
        database_root / suffix,
        database_root / str(base_query_id),
    ]
    root = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    return {
        "hotels": load_csv(root / "hotels" / "hotels.csv") if (root / "hotels" / "hotels.csv").exists() else [],
        "restaurants": load_csv(root / "restaurants" / "restaurants.csv") if (root / "restaurants" / "restaurants.csv").exists() else [],
        "attractions": load_csv(root / "attractions" / "attractions.csv") if (root / "attractions" / "attractions.csv").exists() else [],
    }


def _hard_constraint_keys(meta: dict[str, Any]) -> list[str]:
    hard = meta.get("hard_constraints") or {}
    return sorted(str(key) for key in hard if str(key).strip())


def _base_people(meta: dict[str, Any]) -> int:
    try:
        return max(1, int(meta.get("people_number") or 1))
    except (TypeError, ValueError):
        return 1


def _best_attraction(db: dict[str, list[dict[str, str]]], excluded: set[str]) -> str | None:
    rows = [
        row for row in db.get("attractions", [])
        if str(row.get("attraction_name") or "").strip()
        and str(row.get("attraction_name") or "").strip() not in excluded
    ]
    rows.sort(key=lambda row: (-safe_float(row.get("rating")), safe_float(row.get("ticket_price"), 999999), row.get("attraction_name", "")))
    return str(rows[0].get("attraction_name")).strip() if rows else None


def _best_restaurant(db: dict[str, list[dict[str, str]]]) -> dict[str, Any] | None:
    rows = [row for row in db.get("restaurants", []) if str(row.get("restaurant_name") or "").strip()]
    if not rows:
        return None
    rows.sort(key=lambda row: (-safe_float(row.get("rating")), safe_float(row.get("price_per_person"), 999999), row.get("restaurant_name", "")))
    row = rows[0]
    acceptable = [
        str(item.get("restaurant_name")).strip()
        for item in rows
        if safe_float(item.get("rating")) == safe_float(row.get("rating"))
    ][:5]
    return {
        "name": str(row.get("restaurant_name")).strip(),
        "nearby_attraction": str(row.get("nearby_attraction_name") or "").strip(),
        "cuisine_type": str(row.get("cuisine") or "").split(";")[-1].strip(),
        "price_per_person": safe_float(row.get("price_per_person")),
        "rating": safe_float(row.get("rating")),
        "acceptable_restaurant_names": acceptable or [str(row.get("restaurant_name")).strip()],
    }


def _best_hotel(db: dict[str, list[dict[str, str]]]) -> dict[str, Any] | None:
    rows = [row for row in db.get("hotels", []) if str(row.get("name") or "").strip()]
    if not rows:
        return None
    rows.sort(key=lambda row: (-safe_float(row.get("score")), safe_float(row.get("price"), 999999), row.get("name", "")))
    row = rows[0]
    acceptable = [
        str(item.get("name")).strip()
        for item in rows
        if safe_float(item.get("score")) == safe_float(row.get("score"))
    ][:5]
    return {
        "name": str(row.get("name")).strip(),
        "hotel_star": int(safe_float(row.get("hotel_star"))),
        "score": safe_float(row.get("score")),
        "price": safe_float(row.get("price")),
        "services": [part.strip() for part in str(row.get("services") or "").split(";") if part.strip()],
        "acceptable_hotel_names": acceptable or [str(row.get("name")).strip()],
    }


def _selected_names(meta: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for constraint in (meta.get("hard_constraints") or {}).values():
        if not isinstance(constraint, dict):
            continue
        for key, value in constraint.items():
            if key.endswith("_name") and isinstance(value, str) and value.strip():
                names.add(value.strip())
            if key.endswith("_names") and isinstance(value, list):
                names.update(str(item).strip() for item in value if str(item).strip())
    return names


def _oracle_state(active_hard: list[str], active_user_deltas: list[dict[str, Any]], active_events: list[dict[str, Any]], expectation: str) -> dict[str, Any]:
    return {
        "active_hard_constraints": list(active_hard),
        "active_user_state_deltas": copy.deepcopy(active_user_deltas),
        "active_environment_events": copy.deepcopy(active_events),
        "feasibility_status": "unsolved" if expectation == "no_solution" else "solved",
        "response_expectation": expectation,
    }


def _make_turn(
    *,
    turn_id: int,
    utterance: str,
    active_hard: list[str],
    active_user_deltas: list[dict[str, Any]],
    active_events: list[dict[str, Any]],
    must_update: list[str],
    expectation: str = "plan",
    checks: list[str] | None = None,
    blocking_constraints: list[str] | None = None,
    unsolved_reason: str | None = None,
    oracle_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    oracle = _oracle_state(active_hard, active_user_deltas, active_events, expectation)
    if oracle_extra:
        oracle.update(oracle_extra)
    return {
        "turn_id": turn_id,
        "utterance": utterance,
        "must_preserve": list(active_hard),
        "must_update": must_update,
        "feasibility_status": "unsolved" if expectation == "no_solution" else "solved",
        "response_expectation": expectation,
        "unsolved_reason": unsolved_reason,
        "blocking_constraints": blocking_constraints or [],
        "oracle_state_after_turn": oracle,
        "verification_oracle": {
            "checks": checks or [CHECK_PRESERVE],
            "hard_constraint_keys": list(active_hard),
        },
    }


def _initial_turn(record: dict[str, Any], active_hard: list[str]) -> dict[str, Any]:
    return _make_turn(
        turn_id=0,
        utterance=str(record.get("query") or "").strip(),
        active_hard=active_hard,
        active_user_deltas=[],
        active_events=[],
        must_update=["initial_plan"],
        checks=[CHECK_INITIAL],
    )


def _party_turn(turn_id: int, meta: dict[str, Any], active_hard: list[str], active_user_deltas: list[dict[str, Any]], active_events: list[dict[str, Any]]) -> dict[str, Any]:
    current_people = _base_people(meta)
    new_people = current_people + 1
    new_rooms = common_room_number(new_people)
    delta = {
        "party_update": {
            "party_subtype": "adult_added",
            "new_people_number": new_people,
            "new_room_number": new_rooms,
            "source": "turn_state_delta",
        }
    }
    active_user_deltas.append(copy.deepcopy(delta))
    return _make_turn(
        turn_id=turn_id,
        utterance=(
            f"One more adult may join us, so please update the plan for {new_people} travelers "
            f"and {new_rooms} room(s). Keep the earlier transport, lodging, restaurant, attraction, and budget requirements unchanged."
        ),
        active_hard=active_hard,
        active_user_deltas=active_user_deltas,
        active_events=active_events,
        must_update=["party_update"],
    )


def _late_start_turn(turn_id: int, active_hard: list[str], active_user_deltas: list[dict[str, Any]], active_events: list[dict[str, Any]]) -> dict[str, Any]:
    delta = {
        "late_start_request": {
            "day": 2,
            "earliest_start_time": "10:30",
            "earliest_start_minutes": 630,
            "scope": "first_formal_activity",
            "source": "turn_state_delta",
        }
    }
    active_user_deltas.append(copy.deepcopy(delta))
    return _make_turn(
        turn_id=turn_id,
        utterance="For Day 2, please start the first formal activity no earlier than 10:30. Keep the earlier hard requirements in place.",
        active_hard=active_hard,
        active_user_deltas=active_user_deltas,
        active_events=active_events,
        must_update=["late_start_request"],
        checks=[CHECK_PRESERVE, "late_start_request"],
        oracle_extra={"earliest_start_minutes": 630, "day": 2},
    )


def _early_finish_turn(turn_id: int, active_hard: list[str], active_user_deltas: list[dict[str, Any]], active_events: list[dict[str, Any]]) -> dict[str, Any]:
    delta = {
        "schedule_update": {
            "schedule_subtype": "early_finish",
            "day": 2,
            "latest_formal_end_time": "20:00",
            "latest_formal_end_minutes": 1200,
            "scope": "last_formal_activity",
            "source": "turn_state_delta",
        }
    }
    active_user_deltas.append(copy.deepcopy(delta))
    return _make_turn(
        turn_id=turn_id,
        utterance="For the evening of Day 2, please keep things light and make the last formal activity end by 20:00.",
        active_hard=active_hard,
        active_user_deltas=active_user_deltas,
        active_events=active_events,
        must_update=["schedule_update"],
        checks=[CHECK_PRESERVE, "schedule_update"],
        oracle_extra={"latest_formal_end_minutes": 1200, "day": 2},
    )


def _add_attraction_turn(turn_id: int, name: str, active_hard: list[str], active_user_deltas: list[dict[str, Any]], active_events: list[dict[str, Any]]) -> dict[str, Any]:
    delta = {
        "add_attraction": {
            "name": name,
            "insert_day": 2,
            "source": "turn_state_delta",
        }
    }
    active_user_deltas.append(copy.deepcopy(delta))
    return _make_turn(
        turn_id=turn_id,
        utterance=f"I also want to add {name} to the itinerary, preferably on Day 2. Please do not remove the places or constraints I already gave.",
        active_hard=active_hard,
        active_user_deltas=active_user_deltas,
        active_events=active_events,
        must_update=["add_attraction"],
        checks=[CHECK_PRESERVE, "add_attraction"],
    )


def _restaurant_turn(turn_id: int, restaurant: dict[str, Any], active_hard: list[str], active_user_deltas: list[dict[str, Any]], active_events: list[dict[str, Any]]) -> dict[str, Any]:
    anchor = restaurant.get("nearby_attraction") or "the route"
    delta = {
        "restaurant_requirement": {
            "restaurant_name": restaurant["name"],
            "meal_day": 2,
            "meal_type": "dinner",
            "anchor": anchor,
            "cuisine_type": restaurant.get("cuisine_type"),
            "candidate_reference": {
                "name": restaurant["name"],
                "nearby_attraction": anchor,
                "cuisine_type": restaurant.get("cuisine_type"),
                "price_per_person": restaurant.get("price_per_person"),
                "rating": restaurant.get("rating"),
                "acceptable_restaurant_names": restaurant.get("acceptable_restaurant_names") or [restaurant["name"]],
                "source": "sample_database",
            },
            "source": "turn_state_delta",
        }
    }
    active_user_deltas.append(copy.deepcopy(delta))
    return _make_turn(
        turn_id=turn_id,
        utterance=f"For dinner on Day 2, please use {restaurant['name']} near {anchor}. Keep all earlier requirements unchanged.",
        active_hard=active_hard,
        active_user_deltas=active_user_deltas,
        active_events=active_events,
        must_update=["restaurant_requirement"],
        checks=[CHECK_PRESERVE, "restaurant_requirement"],
    )


def _hotel_turn(turn_id: int, hotel: dict[str, Any], active_hard: list[str], active_user_deltas: list[dict[str, Any]], active_events: list[dict[str, Any]]) -> dict[str, Any]:
    delta = {
        "hotel_requirement": {
            "hotel_name": hotel["name"],
            "candidate_reference": {
                "name": hotel["name"],
                "hotel_star": hotel.get("hotel_star"),
                "score": hotel.get("score"),
                "price": hotel.get("price"),
                "services": hotel.get("services") or [],
                "acceptable_hotel_names": hotel.get("acceptable_hotel_names") or [hotel["name"]],
                "source": "sample_database",
            },
            "source": "turn_state_delta",
        }
    }
    active_user_deltas.append(copy.deepcopy(delta))
    return _make_turn(
        turn_id=turn_id,
        utterance=f"I want to make the hotel choice explicit now: please use {hotel['name']} if it can fit the earlier requirements.",
        active_hard=active_hard,
        active_user_deltas=active_user_deltas,
        active_events=active_events,
        must_update=["hotel_requirement"],
        checks=[CHECK_PRESERVE, "hotel_requirement"],
    )


def _budget_turn(turn_id: int, meta: dict[str, Any], active_hard: list[str], active_user_deltas: list[dict[str, Any]], active_events: list[dict[str, Any]]) -> dict[str, Any]:
    current_budget = (meta.get("hard_constraints") or {}).get("budget_constraint") or {}
    current_cap = int(current_budget.get("max_budget") or max(3000, _base_people(meta) * 1200))
    new_cap = max(1000, int(current_cap * 0.9 // 100 * 100))
    delta = {
        "budget_update": {
            "new_max_budget": new_cap,
            "source": "turn_state_delta",
        }
    }
    active_user_deltas.append(copy.deepcopy(delta))
    return _make_turn(
        turn_id=turn_id,
        utterance=f"I also want to tighten the total budget to no more than {new_cap} RMB. Please preserve the earlier hard requirements as much as possible.",
        active_hard=active_hard,
        active_user_deltas=active_user_deltas,
        active_events=active_events,
        must_update=["budget_update"],
        checks=[CHECK_PRESERVE, "budget_update"],
    )


def _clarification_turn(turn_id: int, active_hard: list[str], active_user_deltas: list[dict[str, Any]], active_events: list[dict[str, Any]]) -> dict[str, Any]:
    return _make_turn(
        turn_id=turn_id,
        utterance=(
            "I may want to add another dinner, but I have not decided the day, cuisine, or per-person budget yet. "
            "Please ask me to clarify before choosing a restaurant."
        ),
        active_hard=active_hard,
        active_user_deltas=active_user_deltas,
        active_events=active_events,
        must_update=["clarification_or_ranked_options"],
        expectation="clarification",
        checks=["response_expectation"],
    )


def _environment_turn(turn_id: int, meta: dict[str, Any], active_hard: list[str], active_user_deltas: list[dict[str, Any]], active_events: list[dict[str, Any]]) -> dict[str, Any]:
    dest = (meta.get("dest") or ["the destination"])[0]
    event = {
        "environment_event": {
            "event_type": "weather_risk",
            "city_name": dest,
            "message": f"{dest} may have rain or unstable outdoor conditions during the trip.",
            "source": "turn_state_delta",
        }
    }
    active_events.append(copy.deepcopy(event["environment_event"]))
    return _make_turn(
        turn_id=turn_id,
        utterance=(
            f"I just saw that {dest} may have rain or unstable outdoor conditions during the trip. "
            "Please adjust outdoor time and transfers, but keep the core places and earlier hard requirements."
        ),
        active_hard=active_hard,
        active_user_deltas=active_user_deltas,
        active_events=active_events,
        must_update=["environment_aware_replanning"],
        checks=[CHECK_PRESERVE, "environment_aware_replanning"],
    )


def _final_turn(turn_id: int, active_hard: list[str], active_user_deltas: list[dict[str, Any]], active_events: list[dict[str, Any]]) -> dict[str, Any]:
    return _make_turn(
        turn_id=turn_id,
        utterance="Please now give me the final integrated itinerary with all earlier requirements and later updates applied together.",
        active_hard=active_hard,
        active_user_deltas=active_user_deltas,
        active_events=active_events,
        must_update=["final_integrated_plan"],
        checks=[CHECK_PRESERVE, "final_integrated_plan"],
    )


def _record_interaction_type(record: dict[str, Any], rng: random.Random) -> str:
    meta = record.get("meta_info") or {}
    raw = normalize_interaction_archetype(str(meta.get("interaction_archetype") or ""))
    if raw in INTERACTION_ARCHETYPES:
        return raw
    labels = [label for label, _weight in INTERACTION_ARCHETYPE_WEIGHTS]
    weights = [weight for _label, weight in INTERACTION_ARCHETYPE_WEIGHTS]
    return rng.choices(labels, weights=weights, k=1)[0]


def build_multiturn_record(record: dict[str, Any], *, seed: int, database_root: Path, forced_interaction_type: str | None = None) -> dict[str, Any]:
    base_id = str(record.get("id") or "")
    rng = random.Random(seed + sum(ord(ch) for ch in base_id))
    meta = copy.deepcopy(record.get("meta_info") or {})
    interaction_type = forced_interaction_type or _record_interaction_type(record, rng)
    meta.pop("interaction_archetype", None)
    active_hard = _hard_constraint_keys(meta)
    active_user_deltas: list[dict[str, Any]] = []
    active_events: list[dict[str, Any]] = []
    db = _load_sample_db(base_id, database_root)
    selected_names = _selected_names(meta)
    attraction = _best_attraction(db, selected_names)
    restaurant = _best_restaurant(db)
    hotel = _best_hotel(db)

    turns = [_initial_turn(record, active_hard)]
    next_turn = 1
    if interaction_type == "user_state_evolution":
        turns.append(_party_turn(next_turn, meta, active_hard, active_user_deltas, active_events))
        next_turn += 1
        turns.append(_late_start_turn(next_turn, active_hard, active_user_deltas, active_events))
        next_turn += 1
        turns.append(_add_attraction_turn(next_turn, attraction, active_hard, active_user_deltas, active_events) if attraction else _budget_turn(next_turn, meta, active_hard, active_user_deltas, active_events))
    elif interaction_type == "request_resolution":
        turns.append(_clarification_turn(next_turn, active_hard, active_user_deltas, active_events))
        next_turn += 1
        if restaurant:
            turns.append(_restaurant_turn(next_turn, restaurant, active_hard, active_user_deltas, active_events))
            next_turn += 1
        turns.append(_final_turn(next_turn, active_hard, active_user_deltas, active_events))
    elif interaction_type == "environment_driven_replanning":
        turns.append(_environment_turn(next_turn, meta, active_hard, active_user_deltas, active_events))
        next_turn += 1
        turns.append(_early_finish_turn(next_turn, active_hard, active_user_deltas, active_events))
        next_turn += 1
        turns.append(_final_turn(next_turn, active_hard, active_user_deltas, active_events))
    else:
        turns.append(_party_turn(next_turn, meta, active_hard, active_user_deltas, active_events))
        next_turn += 1
        turns.append(_environment_turn(next_turn, meta, active_hard, active_user_deltas, active_events))
        next_turn += 1
        if hotel:
            turns.append(_hotel_turn(next_turn, hotel, active_hard, active_user_deltas, active_events))
            next_turn += 1
        elif attraction:
            turns.append(_add_attraction_turn(next_turn, attraction, active_hard, active_user_deltas, active_events))
            next_turn += 1
        turns.append(_final_turn(next_turn, active_hard, active_user_deltas, active_events))

    return {
        "id": f"mt_{base_id}",
        "base_query_id": base_id,
        "query": str(record.get("query") or ""),
        "interaction_type": interaction_type,
        "turns": turns,
        "meta_info": {
            "base_query_meta": meta,
        },
    }


def generate_multiturn_records(args: Any) -> list[dict[str, Any]]:
    records = load_json(Path(args.input))
    if not isinstance(records, list):
        raise ValueError(f"Input query file must contain a JSON list: {args.input}")
    if args.count:
        records = records[: args.count]
    database_root = Path(args.database_root)
    return [
        build_multiturn_record(record, seed=args.seed, database_root=database_root)
        for record in records
        if isinstance(record, dict)
    ]


def refresh_environment_records_from_template(records: list[dict[str, Any]], args: Any) -> list[dict[str, Any]]:
    if not args.refresh_environment_from:
        return records
    template_path = Path(args.refresh_environment_from)
    if not template_path.is_absolute():
        template_path = BASE_DIR / template_path
    if not template_path.exists():
        raise FileNotFoundError(f"Environment refresh template not found: {template_path}")

    source_records = load_json(Path(args.input))
    source_by_id = {str(record.get("id")): record for record in source_records if isinstance(record, dict)}
    template_records = load_json(template_path)
    if not isinstance(template_records, list):
        raise ValueError(f"Environment refresh template must be a JSON list: {template_path}")

    refreshed = []
    for old_record in template_records:
        base_id = str((old_record or {}).get("base_query_id") or "").strip()
        source = source_by_id.get(base_id)
        if not source:
            raise ValueError(f"Environment refresh base_query_id not found in input: {base_id}")
        refreshed_record = build_multiturn_record(
            source,
            seed=args.seed,
            database_root=Path(args.database_root),
            forced_interaction_type="environment_driven_replanning",
        )
        refreshed_record["id"] = f"mt_env_{base_id}"
        refreshed_record["meta_info"]["refresh_batch"] = Path(args.environment_output or args.refresh_environment_from).stem
        refreshed.append(refreshed_record)

    return [record for record in records if record.get("interaction_type") != "environment_driven_replanning"] + refreshed
