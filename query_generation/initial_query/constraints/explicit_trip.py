"""Utilities for promoting explicit trip metadata into hard constraints."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


EXPLICIT_TRIP_HARD_CONSTRAINT_KEYS = (
    "trip_date_range_required",
    "party_size_required",
    "room_count_required",
    "intercity_round_trip_mode_required",
    "party_composition_required",
)


def _as_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _destinations(meta: dict[str, Any]) -> list[str]:
    dest = meta.get("dest")
    if isinstance(dest, list):
        return [str(item) for item in dest if str(item).strip()]
    if dest not in (None, ""):
        return [str(dest)]
    return []


def _party_composition(meta: dict[str, Any]) -> dict[str, Any] | None:
    observable = meta.get("observable_profile") or {}
    if not isinstance(observable, dict):
        return None
    composition = observable.get("party_composition")
    if not isinstance(composition, dict):
        return None
    children = composition.get("children") or []
    elders = composition.get("elders") or []
    if not children and not elders:
        return None
    return deepcopy(composition)


def build_explicit_trip_hard_constraints(meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build deterministic hard constraints from explicit trip metadata."""
    constraints: dict[str, dict[str, Any]] = {}

    days = _as_int(meta.get("days"))
    people_number = _as_int(meta.get("people_number"))
    room_number = _as_int(meta.get("room_number"))
    depart_date = str(meta.get("depart_date") or "").strip()
    return_date = str(meta.get("return_date") or "").strip()
    route_mode = str(meta.get("route_mode") or "").strip().lower()
    org = str(meta.get("org") or "").strip()
    destinations = _destinations(meta)

    if depart_date and return_date and days:
        constraints["trip_date_range_required"] = {
            "constraint_context": "The itinerary must use the explicit departure/return dates and trip length from the user request.",
            "depart_date": depart_date,
            "return_date": return_date,
            "days": days,
        }
    if people_number:
        constraints["party_size_required"] = {
            "constraint_context": "The itinerary budget must be calculated for the explicit party size from the user request.",
            "people_number": people_number,
        }
    if room_number:
        constraints["room_count_required"] = {
            "constraint_context": "The accommodation budget must be calculated for the explicit room count from the user request.",
            "room_number": room_number,
        }
    if route_mode in {"flight", "train"}:
        constraints["intercity_round_trip_mode_required"] = {
            "constraint_context": "All intercity public transportation segments must use the explicit round-trip mode from the user request.",
            "route_mode": route_mode,
            "origin": org,
            "destinations": destinations,
        }

    composition = _party_composition(meta)
    if composition and people_number:
        constraints["party_composition_required"] = {
            "constraint_context": "The itinerary budget must preserve the explicit child/elder party composition from the user request.",
            "people_number": people_number,
            "party_composition": composition,
        }

    return constraints


def ensure_explicit_trip_hard_constraints(meta: dict[str, Any]) -> list[str]:
    """Add missing explicit trip hard constraints to a meta_info object.

    Returns the keys that were inserted.
    """
    hard_constraints = meta.setdefault("hard_constraints", {})
    if not isinstance(hard_constraints, dict):
        meta["hard_constraints"] = {}
        hard_constraints = meta["hard_constraints"]

    inserted: list[str] = []
    for key, value in build_explicit_trip_hard_constraints(meta).items():
        if key not in hard_constraints:
            hard_constraints[key] = value
            inserted.append(key)
    return inserted


def refresh_explicit_trip_hard_constraints(meta: dict[str, Any]) -> None:
    """Overwrite or insert explicit trip hard-constraint payloads from current meta fields."""
    hard_constraints = meta.setdefault("hard_constraints", {})
    if not isinstance(hard_constraints, dict):
        meta["hard_constraints"] = {}
        hard_constraints = meta["hard_constraints"]
    for key, value in build_explicit_trip_hard_constraints(meta).items():
        hard_constraints[key] = value
