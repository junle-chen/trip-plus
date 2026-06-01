"""Ground-truth derivation for static multi-turn travel queries.

Multi-turn records store the user-visible dialogue plus hidden oracle state in
``turns``. This module materializes that oracle state into per-turn evaluator
metadata so the existing single-plan evaluators can be reused turn by turn.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List

from query_generation.initial_query.constraints.explicit_trip import (
    refresh_explicit_trip_hard_constraints,
)


def load_query_records(query_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(query_path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported query file format: {query_path}")


def base_meta_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return the single-turn metadata that a multi-turn record extends."""
    meta = record.get("meta_info") or {}
    base_meta = meta.get("base_query_meta") if isinstance(meta, dict) else None
    if isinstance(base_meta, dict):
        return copy.deepcopy(base_meta)
    return copy.deepcopy(meta if isinstance(meta, dict) else {})


def _active_user_state_deltas(oracle_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    deltas = oracle_state.get("active_user_state_deltas") or []
    return [delta for delta in deltas if isinstance(delta, dict)]


def _delta_identity(delta: Dict[str, Any]) -> str:
    return json.dumps(delta, sort_keys=True, ensure_ascii=False)


def _current_user_state_deltas(
    record: Dict[str, Any], turn: Dict[str, Any], oracle_state: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Return deltas newly introduced by this turn, not the full active state."""
    active = _active_user_state_deltas(oracle_state)
    turns = record.get("turns") or []
    current_index = None
    for index, candidate in enumerate(turns):
        if candidate is turn or candidate.get("turn_id") == turn.get("turn_id"):
            current_index = index
            break
    if current_index is None or current_index <= 0:
        return copy.deepcopy(active)
    previous_oracle = turns[current_index - 1].get("oracle_state_after_turn") or {}
    previous_keys = {
        _delta_identity(delta) for delta in _active_user_state_deltas(previous_oracle)
    }
    return [
        copy.deepcopy(delta)
        for delta in active
        if _delta_identity(delta) not in previous_keys
    ]


def _inferred_must_update_from_current_deltas(
    current_user_deltas: Iterable[Dict[str, Any]],
) -> List[str]:
    mapping = {
        "add_attraction": "add_attraction",
        "party_update": "party_update",
        "restaurant_requirement": "restaurant_requirement",
        "hotel_requirement": "hotel_requirement",
        "budget_update": "budget_update",
        "schedule_update": "schedule_update",
        "duration_update": "duration_update",
        "dietary_update": "dietary_update",
        "late_start_request": "late_start_request",
    }
    inferred: List[str] = []
    for delta in current_user_deltas:
        for key in delta:
            update_name = mapping.get(str(key))
            if update_name and update_name not in inferred:
                inferred.append(update_name)
    return inferred


def _current_generated_constraints(
    generated_constraints: Dict[str, Dict[str, Any]],
    current_user_deltas: Iterable[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    current_attractions = {
        str(delta.get("add_attraction", {}).get("name") or "").strip()
        for delta in current_user_deltas
        if isinstance(delta.get("add_attraction"), dict)
    }
    current_restaurants = {
        str(
            delta.get("restaurant_requirement", {}).get("restaurant_name")
            or (delta.get("restaurant_requirement", {}).get("candidate_reference") or {}).get(
                "name"
            )
            or ""
        ).strip()
        for delta in current_user_deltas
        if isinstance(delta.get("restaurant_requirement"), dict)
    }
    current_hotels = {
        str(
            delta.get("hotel_requirement", {}).get("hotel_name")
            or (delta.get("hotel_requirement", {}).get("candidate_reference") or {}).get(
                "name"
            )
            or ""
        ).strip()
        for delta in current_user_deltas
        if isinstance(delta.get("hotel_requirement"), dict)
    }
    has_current_budget = any(
        isinstance(delta.get("budget_update"), dict) for delta in current_user_deltas
    )

    current: Dict[str, Dict[str, Any]] = {}
    for key, constraint in generated_constraints.items():
        if not isinstance(constraint, dict):
            continue
        if str(key).startswith("attraction_"):
            names = {
                str(name).strip()
                for name in constraint.get("attraction_names", []) or []
                if str(name).strip()
            }
            if names & current_attractions:
                current[key] = copy.deepcopy(constraint)
        elif str(key).startswith("restaurant_"):
            names = {
                str(constraint.get("restaurant_name") or "").strip(),
                *[
                    str(name).strip()
                    for name in constraint.get("acceptable_restaurant_names", []) or []
                    if str(name).strip()
                ],
            }
            if names & current_restaurants:
                current[key] = copy.deepcopy(constraint)
        elif str(key).startswith("hotel_"):
            names = {
                str(constraint.get("hotel_name") or "").strip(),
                *[
                    str(name).strip()
                    for name in constraint.get("acceptable_hotel_names", []) or []
                    if str(name).strip()
                ],
            }
            if names & current_hotels:
                current[key] = copy.deepcopy(constraint)
        elif key == "budget_constraint" and has_current_budget:
            current[key] = copy.deepcopy(constraint)
    return current


def _party_total(composition: Dict[str, Any]) -> int:
    return (
        int(composition.get("adults") or 0)
        + len(composition.get("children") or [])
        + len(composition.get("elders") or [])
    )


def _fit_party_composition(
    meta: Dict[str, Any], target_people: int, party_update: Dict[str, Any]
) -> None:
    observable = meta.get("observable_profile")
    if not isinstance(observable, dict):
        return
    composition = observable.get("party_composition")
    if not isinstance(composition, dict):
        return

    current_total = _party_total(composition)
    if target_people <= 0:
        return

    updated = copy.deepcopy(composition)
    subtype = str(party_update.get("party_subtype") or "")

    if current_total < target_people:
        add_count = target_people - current_total
        if subtype == "child_added":
            requested = int(party_update.get("added_children") or add_count)
            children = list(updated.get("children") or [])
            for _ in range(max(1, min(add_count, requested))):
                children.append({"age": None, "note": "added in user turn"})
            updated["children"] = children
        elif subtype == "elder_added":
            requested = int(party_update.get("added_elders") or add_count)
            elders = list(updated.get("elders") or [])
            for _ in range(max(1, min(add_count, requested))):
                elders.append({"age": None, "mobility_note": "added in user turn"})
            updated["elders"] = elders
        else:
            updated["adults"] = int(updated.get("adults") or 0) + add_count
        observable["party_composition"] = updated
        return

    if current_total == target_people:
        return

    remove_count = current_total - target_people

    adults = int(updated.get("adults") or 0)
    removed_adults = min(adults, remove_count)
    updated["adults"] = adults - removed_adults
    remove_count -= removed_adults

    # If adults are not enough, remove non-minor companions next. Children are
    # preserved as long as possible because child-related constraints/profile
    # rules are more behaviorally significant for planning.
    elders = list(updated.get("elders") or [])
    if remove_count and elders:
        updated["elders"] = elders[:-remove_count] if remove_count < len(elders) else []
        remove_count = max(0, remove_count - len(elders))

    children = list(updated.get("children") or [])
    if remove_count and children:
        updated["children"] = (
            children[:-remove_count] if remove_count < len(children) else []
        )

    observable["party_composition"] = updated


def _apply_party_updates(
    meta: Dict[str, Any], active_user_deltas: Iterable[Dict[str, Any]]
) -> None:
    for delta in active_user_deltas:
        party_update = delta.get("party_update")
        if not isinstance(party_update, dict):
            continue
        if party_update.get("new_people_number") not in (None, ""):
            meta["people_number"] = party_update["new_people_number"]
            try:
                _fit_party_composition(
                    meta, int(party_update["new_people_number"]), party_update
                )
            except (TypeError, ValueError):
                pass
        if party_update.get("new_room_number") not in (None, ""):
            meta["room_number"] = party_update["new_room_number"]
        user_profile = meta.get("user_profile")
        if isinstance(user_profile, dict):
            derivation = user_profile.get("derivation_context")
            if isinstance(derivation, dict) and party_update.get(
                "new_people_number"
            ) not in (None, ""):
                derivation["party_size"] = party_update["new_people_number"]


def _apply_duration_updates(
    meta: Dict[str, Any], active_user_deltas: Iterable[Dict[str, Any]]
) -> None:
    latest: Dict[str, Any] | None = None
    for delta in active_user_deltas:
        duration_update = delta.get("duration_update")
        if isinstance(duration_update, dict):
            latest = duration_update
    if not latest:
        return
    if latest.get("new_days") not in (None, ""):
        meta["days"] = latest["new_days"]
    if latest.get("new_return_date"):
        meta["return_date"] = latest["new_return_date"]
    elif meta.get("depart_date") and latest.get("new_days") not in (None, ""):
        try:
            depart = datetime.strptime(str(meta["depart_date"]), "%Y-%m-%d")
            meta["return_date"] = (
                depart + timedelta(days=int(latest["new_days"]) - 1)
            ).strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            pass


def _base_active_hard_constraints(
    base_meta: Dict[str, Any], oracle_state: Dict[str, Any]
) -> Dict[str, Any]:
    base_constraints = base_meta.get("hard_constraints") or {}
    if not isinstance(base_constraints, dict):
        return {}
    active_keys = oracle_state.get("active_hard_constraints") or []
    if not active_keys:
        return copy.deepcopy(base_constraints)
    active_key_set = {str(key) for key in active_keys}
    return {
        key: copy.deepcopy(value)
        for key, value in base_constraints.items()
        if str(key) in active_key_set
    }


def _added_attraction_constraints(
    active_user_deltas: Iterable[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    constraints: Dict[str, Dict[str, Any]] = {}
    seen_names: set[str] = set()
    for delta in active_user_deltas:
        added = delta.get("add_attraction")
        if not isinstance(added, dict):
            continue
        name = str(added.get("name") or "").strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        key = f"attraction_turn_added_{len(constraints) + 1}"
        constraints[key] = {
            "constraint_context": f"Multi-turn user requested adding attraction: {name}",
            "constraint_type": "turn_added_must_visit",
            "attraction_names": [name],
            "source": added.get("source") or "turn_state_delta",
            "insert_day": added.get("insert_day"),
        }
    return constraints


def _added_restaurant_constraints(
    active_user_deltas: Iterable[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    constraints: Dict[str, Dict[str, Any]] = {}
    seen_names: set[str] = set()
    for delta in active_user_deltas:
        requirement = delta.get("restaurant_requirement")
        if not isinstance(requirement, dict):
            continue
        reference = requirement.get("candidate_reference") or {}
        name = str(
            requirement.get("restaurant_name") or reference.get("name") or ""
        ).strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        acceptable = [
            str(item).strip()
            for item in reference.get("acceptable_restaurant_names") or [name]
            if str(item).strip()
        ]
        key = f"restaurant_turn_added_{len(constraints) + 1}"
        constraints[key] = {
            "constraint_context": f"Multi-turn user requested adding restaurant: {name}",
            "constraint_type": "turn_added_restaurant_requirement",
            "restaurant_name": name,
            "acceptable_restaurant_names": acceptable or [name],
            "attraction_name": requirement.get("anchor")
            or reference.get("nearby_attraction"),
            "cuisine_type": requirement.get("cuisine_type")
            or reference.get("cuisine_type"),
            "meal_day": requirement.get("meal_day"),
            "meal_type": requirement.get("meal_type"),
            "source": reference.get("source") or "turn_state_delta",
        }
    return constraints


def _added_hotel_constraints(
    active_user_deltas: Iterable[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    constraints: Dict[str, Dict[str, Any]] = {}
    seen_names: set[str] = set()
    for delta in active_user_deltas:
        requirement = delta.get("hotel_requirement")
        if not isinstance(requirement, dict):
            continue
        reference = requirement.get("candidate_reference") or {}
        name = str(requirement.get("hotel_name") or reference.get("name") or "").strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        acceptable = [
            str(item).strip()
            for item in reference.get("acceptable_hotel_names") or [name]
            if str(item).strip()
        ]
        key = f"hotel_turn_added_{len(constraints) + 1}"
        constraints[key] = {
            "constraint_context": f"Multi-turn user requested adding hotel: {name}",
            "constraint_type": "turn_added_hotel_requirement",
            "hotel_name": name,
            "acceptable_hotel_names": acceptable or [name],
            "requirement_type": reference.get("requirement_type"),
            "hotel_star": reference.get("hotel_star"),
            "hotel_score": reference.get("score"),
            "hotel_price": reference.get("price"),
            "required_service_label": reference.get("required_service_label"),
            "services": reference.get("services") or [],
            "source": reference.get("source") or "turn_state_delta",
        }
    return constraints


def _budget_update_constraints(
    base_meta: Dict[str, Any], active_user_deltas: Iterable[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Any] | None = None
    for delta in active_user_deltas:
        update = delta.get("budget_update")
        if isinstance(update, dict):
            latest = update
    if not latest:
        return {}
    new_cap = latest.get("new_max_budget")
    if new_cap in (None, ""):
        return {}
    base_budget = (base_meta.get("hard_constraints") or {}).get("budget_constraint")
    constraint = copy.deepcopy(base_budget) if isinstance(base_budget, dict) else {}
    constraint.update(
        {
            "constraint_context": f"Multi-turn user lowered total budget cap to {new_cap} RMB",
            "constraint_type": "budget_range_total",
            "max_budget": new_cap,
            "source": latest.get("source") or "turn_state_delta",
        }
    )
    if latest.get("minimum_feasible_budget") not in (None, ""):
        constraint["minimum_feasible_budget"] = latest["minimum_feasible_budget"]
        constraint.setdefault("min_budget", latest["minimum_feasible_budget"])
    if latest.get("budget_slack_ratio") not in (None, ""):
        constraint["budget_slack_ratio"] = latest["budget_slack_ratio"]
    return {"budget_constraint": constraint}


def _unsat_reason_from_turn(turn: Dict[str, Any]) -> Dict[str, Any]:
    reason = str(turn.get("unsolved_reason") or "").strip()
    blocking = [
        str(item)
        for item in turn.get("blocking_constraints") or []
        if str(item).strip()
    ]
    joined = f"{reason} {' '.join(blocking)}".lower()
    keywords = list(blocking)
    if any(
        term in joined
        for term in (
            "budget",
            "cost",
            "price",
            "minimum cost",
            "total budget",
            "fare",
            "upper limit",
            "per night",
            "per person",
        )
    ):
        keywords.extend(
            [
                "budget",
                "total budget",
                "cost",
                "fare",
                "price",
                "upper limit",
                "over budget",
                "exceed",
                "insufficient",
            ]
        )
    if any(
        term in joined
        for term in (
            "refuses",
            "relax",
            "hard",
            "constraint",
            "preserve",
            "unchanged",
            "do not relax",
        )
    ):
        keywords.extend(
            [
                "hard constraint",
                "preserve",
                "unchanged",
                "do not relax",
                "infeasible",
                "no solution",
                "cannot satisfy",
            ]
        )
    if "hotel" in joined or "accommodation" in joined:
        keywords.extend(["accommodation", "hotel"])
    if any(term in joined for term in ("transport", "train", "flight")):
        keywords.extend(["transport", "train", "flight"])

    generic_tokens = {
        "user",
        "refuses",
        "relax",
        "infeasible",
        "added",
        "condition",
        "request",
        "constraint",
        "constraints",
        "multi",
        "turn",
    }
    for token in reason.replace("-", " ").replace("_", " ").split():
        token = token.strip()
        if len(token) >= 4 and token.lower() not in generic_tokens:
            keywords.append(token)
    deduped_keywords: List[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        normalized = keyword.lower().replace(" ", "")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_keywords.append(keyword)
    return {
        "reason": reason
        or "multi-turn request is unsatisfiable under active constraints",
        "reason_keywords": deduped_keywords[:12],
        "blocking_constraints": blocking,
    }


def derive_turn_ground_truth(
    record: Dict[str, Any], turn: Dict[str, Any]
) -> Dict[str, Any]:
    """Derive per-turn evaluator metadata and oracle details from a query turn."""
    base_meta = base_meta_from_record(record)
    oracle_state = copy.deepcopy(turn.get("oracle_state_after_turn") or {})
    active_user_deltas = _active_user_state_deltas(oracle_state)
    current_user_deltas = _current_user_state_deltas(record, turn, oracle_state)

    turn_meta = copy.deepcopy(base_meta)
    active_constraints = _base_active_hard_constraints(base_meta, oracle_state)
    generated_constraints: Dict[str, Dict[str, Any]] = {}
    generated_constraints.update(_added_attraction_constraints(active_user_deltas))
    generated_constraints.update(_added_restaurant_constraints(active_user_deltas))
    generated_constraints.update(_added_hotel_constraints(active_user_deltas))
    generated_constraints.update(
        _budget_update_constraints(base_meta, active_user_deltas)
    )
    current_generated_constraints = _current_generated_constraints(
        generated_constraints, current_user_deltas
    )
    active_constraints.update(generated_constraints)
    turn_meta["hard_constraints"] = active_constraints
    _apply_party_updates(turn_meta, active_user_deltas)
    _apply_duration_updates(turn_meta, active_user_deltas)
    refresh_explicit_trip_hard_constraints(turn_meta)

    response_expectation = (
        str(
            turn.get("response_expectation")
            or oracle_state.get("response_expectation")
            or "plan"
        )
        .strip()
        .lower()
    )
    expected_status = (
        str(
            turn.get("feasibility_status")
            or oracle_state.get("feasibility_status")
            or "solved"
        )
        .strip()
        .lower()
    )
    if (
        response_expectation in {"infeasible", "no_solution"}
        or expected_status == "unsolved"
    ):
        response_expectation = "no_solution"
        expected_status = "unsat"
    turn_meta["solution_status"] = "unsat" if expected_status == "unsat" else "sat"
    turn_meta["response_expectation"] = response_expectation
    if turn_meta["solution_status"] == "unsat":
        turn_meta["unsat_reason"] = _unsat_reason_from_turn(turn)

    turn_meta["turn_ground_truth"] = {
        "sample_id": record.get("id"),
        "base_query_id": record.get("base_query_id"),
        "turn_id": turn.get("turn_id"),
        "interaction_type": turn.get("interaction_type")
        or record.get("interaction_type"),
        "must_preserve": copy.deepcopy(turn.get("must_preserve") or []),
        "must_update": copy.deepcopy(turn.get("must_update") or []),
        "inferred_must_update": _inferred_must_update_from_current_deltas(
            current_user_deltas
        ),
        "current_user_state_deltas": current_user_deltas,
        "oracle_state_after_turn": oracle_state,
        "verification_oracle": copy.deepcopy(turn.get("verification_oracle") or {}),
        "generated_hard_constraints": generated_constraints,
        "current_generated_hard_constraints": current_generated_constraints,
        "expected_solution_status": turn_meta["solution_status"],
        "response_expectation": response_expectation,
    }
    return {
        "turn_id": turn.get("turn_id"),
        "utterance": turn.get("utterance"),
        "meta_info": turn_meta,
        "ground_truth": copy.deepcopy(turn_meta["turn_ground_truth"]),
    }


def derive_record_ground_truth(record: Dict[str, Any]) -> Dict[str, Any]:
    turns = record.get("turns") or []
    if not turns:
        base_meta = base_meta_from_record(record)
        turn = {
            "turn_id": 0,
            "utterance": record.get("query") or record.get("base_query"),
            "interaction_type": record.get("interaction_type", "single_turn"),
            "must_preserve": [],
            "must_update": ["initial_plan"],
            "feasibility_status": base_meta.get("solution_status", "sat"),
            "oracle_state_after_turn": {
                "active_hard_constraints": sorted(
                    (base_meta.get("hard_constraints") or {}).keys()
                ),
                "feasibility_status": base_meta.get("solution_status", "sat"),
            },
            "verification_oracle": {"checks": ["satisfy_initial_hard_constraints"]},
        }
        turns = [turn]
    return {
        "id": record.get("id"),
        "base_query_id": record.get("base_query_id"),
        "interaction_type": record.get("interaction_type"),
        "turns": [
            derive_turn_ground_truth(record, turn)
            for turn in turns
            if isinstance(turn, dict)
        ],
    }


def derive_ground_truth_from_queries(query_path: Path) -> List[Dict[str, Any]]:
    return [
        derive_record_ground_truth(record) for record in load_query_records(query_path)
    ]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Materialize per-turn ground truth from multi-turn query records."
    )
    parser.add_argument("--query-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    records = derive_ground_truth_from_queries(args.query_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
