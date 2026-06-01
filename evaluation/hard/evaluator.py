"""Dispatcher for explicit hard-constraint evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .budget import _eval_budget_constraint
from .places import (
    _eval_attraction_constraint,
    _eval_hotel_constraint,
    _eval_restaurant_constraint,
)
from .transport import _eval_flight_constraint, _eval_train_constraint
from .trip import (
    _eval_intercity_round_trip_mode_required,
    _eval_party_composition_required,
    _eval_party_size_required,
    _eval_room_count_required,
    _eval_trip_date_range_required,
)


def iter_normalized_hard_constraints(
    meta: Dict[str, Any],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Return hard constraints in evaluator-native shape.

    Profiled initial-query samples store visible prompt metadata around the actual
    constraint as {category, constraint_key, visible_hint, constraint}. The
    evaluator dispatches by the concrete constraint key, so unwrap that shape
    while keeping flat constraints unchanged.
    """
    normalized: List[Tuple[str, Dict[str, Any]]] = []
    for constraint_key, constraint_data in (meta.get("hard_constraints") or {}).items():
        if not isinstance(constraint_data, dict):
            continue
        nested = constraint_data.get("constraint")
        nested_key = constraint_data.get("constraint_key")
        if isinstance(nested, dict) and isinstance(nested_key, str) and nested_key:
            normalized.append((nested_key, nested))
        else:
            normalized.append((constraint_key, constraint_data))
    return normalized


def eval_hard(
    plan: Dict[str, Any], meta: Dict[str, Any]
) -> Dict[str, Tuple[Optional[bool], Optional[str]]]:
    """Evaluate every explicit hard constraint attached to the query metadata."""
    res: Dict[str, Tuple[Optional[bool], Optional[str]]] = {}

    if "hard_constraints" not in meta:
        return res

    for constraint_key, constraint_data in iter_normalized_hard_constraints(meta):
        try:
            if constraint_key == "trip_date_range_required":
                result = _eval_trip_date_range_required(constraint_data, plan, meta)
            elif constraint_key == "party_size_required":
                result = _eval_party_size_required(constraint_data, plan, meta)
            elif constraint_key == "room_count_required":
                result = _eval_room_count_required(constraint_data, plan, meta)
            elif constraint_key == "party_composition_required":
                result = _eval_party_composition_required(constraint_data, plan, meta)
            elif constraint_key == "intercity_round_trip_mode_required":
                result = _eval_intercity_round_trip_mode_required(
                    constraint_data, plan, meta
                )
            elif constraint_key.startswith("flight_"):
                result = _eval_flight_constraint(
                    constraint_key, constraint_data, plan, meta
                )
            elif constraint_key.startswith("train_"):
                result = _eval_train_constraint(
                    constraint_key, constraint_data, plan, meta
                )
            elif constraint_key.startswith("hotel_"):
                result = _eval_hotel_constraint(
                    constraint_key, constraint_data, plan, meta
                )
            elif constraint_key.startswith("restaurant_"):
                result = _eval_restaurant_constraint(
                    constraint_key, constraint_data, plan, meta
                )
            elif constraint_key.startswith("attraction_"):
                result = _eval_attraction_constraint(
                    constraint_key, constraint_data, plan, meta
                )
            elif constraint_key == "budget_constraint":
                result = _eval_budget_constraint(constraint_data, plan, meta)
            else:
                result = (None, f"Unknown constraint type: {constraint_key}")

            res[constraint_key] = result
        except Exception as e:
            res[constraint_key] = (False, f"Evaluation error: {str(e)}")

    return res
