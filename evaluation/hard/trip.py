"""Trip metadata hard-constraint checks."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ..scoring_config import normalize_intercity_mode
from .common import _budget_component_matches, _safe_int


def _expected_trip_dates(depart_date: str, days: int) -> List[str]:
    start = datetime.strptime(depart_date, "%Y-%m-%d").date()
    return [(start + timedelta(days=offset)).isoformat() for offset in range(days)]


def _eval_trip_date_range_required(
    constraint_data: Dict, plan: Dict, meta: Dict
) -> Tuple[bool, Optional[str]]:
    daily_plans = plan.get("daily_plans") or []
    expected_days = _safe_int(meta.get("days") or constraint_data.get("days"))
    depart_date = str(
        meta.get("depart_date") or constraint_data.get("depart_date") or ""
    ).strip()
    return_date = str(
        meta.get("return_date") or constraint_data.get("return_date") or ""
    ).strip()

    if not expected_days or expected_days <= 0:
        return False, "Trip date constraint missing valid days"
    if len(daily_plans) != expected_days:
        return False, f"Plan has {len(daily_plans)} days, expected {expected_days}"
    if not depart_date or not return_date:
        return False, "Trip date constraint missing depart_date or return_date"

    try:
        expected_dates = _expected_trip_dates(depart_date, expected_days)
    except ValueError:
        return False, f"Invalid depart_date in trip date constraint: {depart_date}"
    if expected_dates[-1] != return_date:
        return (
            False,
            f"Date range inconsistent: {depart_date} + {expected_days} days ends on {expected_dates[-1]}, expected return_date={return_date}",
        )

    plan_dates = [
        str(day.get("date") or "").strip()
        for day in daily_plans
        if isinstance(day, dict)
    ]
    if any(plan_dates):
        if len(plan_dates) != expected_days or any(not value for value in plan_dates):
            return (
                False,
                "Plan has partial date annotations; every daily plan must include date",
            )
        if plan_dates != expected_dates:
            return (
                False,
                f"Plan dates {plan_dates} do not match expected {expected_dates}",
            )
    return True, None


def _eval_party_size_required(
    constraint_data: Dict, plan: Dict, meta: Dict
) -> Tuple[bool, Optional[str]]:
    expected_people = _safe_int(
        meta.get("people_number") or constraint_data.get("people_number")
    )
    if not expected_people or expected_people <= 0:
        return False, "Party size constraint missing valid people_number"
    return _budget_component_matches(
        plan, meta, ["transportation", "meals", "attractions"]
    )


def _eval_room_count_required(
    constraint_data: Dict, plan: Dict, meta: Dict
) -> Tuple[bool, Optional[str]]:
    expected_rooms = _safe_int(
        meta.get("room_number") or constraint_data.get("room_number")
    )
    if not expected_rooms or expected_rooms <= 0:
        return False, "Room count constraint missing valid room_number"
    return _budget_component_matches(plan, meta, ["accommodation"])


def _eval_party_composition_required(
    constraint_data: Dict, plan: Dict, meta: Dict
) -> Tuple[bool, Optional[str]]:
    expected_people = _safe_int(
        meta.get("people_number") or constraint_data.get("people_number")
    )
    if not expected_people or expected_people <= 0:
        return False, "Party composition constraint missing valid people_number"
    return _budget_component_matches(
        plan, meta, ["transportation", "meals", "attractions"]
    )


def _eval_intercity_round_trip_mode_required(
    constraint_data: Dict, plan: Dict, meta: Dict
) -> Tuple[bool, Optional[str]]:
    expected_mode = normalize_intercity_mode(
        meta.get("route_mode") or constraint_data.get("route_mode")
    )
    if expected_mode not in {"flight", "train"}:
        return (
            False,
            f"Unsupported or missing route_mode for explicit intercity mode constraint: {expected_mode}",
        )

    daily_plans = plan.get("daily_plans") or []
    intercity_modes: List[str] = []
    for day in daily_plans:
        for activity in day.get("activities", []) or []:
            if activity.get("type") != "travel_intercity_public":
                continue
            details = activity.get("details") or {}
            intercity_modes.append(normalize_intercity_mode(details.get("mode")))

    if not intercity_modes:
        return False, f"Required round-trip intercity mode {expected_mode} not found"
    wrong_modes = [
        mode or "<missing>" for mode in intercity_modes if mode != expected_mode
    ]
    if wrong_modes:
        return (
            False,
            f"Intercity mode mismatch: expected all {expected_mode}, got {intercity_modes}",
        )
    return True, None


# ============================================================================
# Flight Constraints
# ============================================================================
