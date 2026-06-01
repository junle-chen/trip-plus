"""Intercity transport soft-preference checks."""

from __future__ import annotations

from typing import Any, Dict, List

from ..scoring_config import normalize_intercity_mode
from ..utils import slot_to_minutes
from .common import (
    _contains_any,
    _count_violation,
    _iter_activities,
    _json_text,
    _level_result,
    _not_applicable,
    _transfer_count_from_details,
    _worst_severity,
)


def _intercity_acts(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        act
        for _day, act in _iter_activities(plan)
        if act.get("type") == "travel_intercity_public"
    ]


def _score_transport(plan: Dict[str, Any], rule_id: str) -> Dict[str, Any]:
    """Score intercity transport timing, transfer, and mode preferences."""
    acts = _intercity_acts(plan)
    if not acts:
        return _not_applicable(
            rule_id, "No intercity segment is present.", {"intercity_segments": 0}
        )

    details = [act.get("details") or {} for act in acts]
    modes = [str(item.get("mode") or "").lower() for item in details]
    starts_early = 0
    arrives_late = 0
    red_eye = 0
    transfer_mentions = 0
    structured_transfer_count = 0
    structured_transfer_segments = 0
    for act in acts:
        start, end = slot_to_minutes(act.get("time_slot"))
        if start is not None and start < 7 * 60:
            starts_early += 1
        if end is not None and end > 21 * 60:
            arrives_late += 1
        if (start is not None and start < 6 * 60) or (
            end is not None and end > 22 * 60
        ):
            red_eye += 1
        text = _json_text(act)
        if _contains_any(text, ("transfer", "connection", "layover")):
            transfer_mentions += 1
        transfer_count = _transfer_count_from_details(act.get("details") or {})
        if transfer_count > 0:
            structured_transfer_segments += 1
            structured_transfer_count += transfer_count

    if rule_id == "transport_avoid_red_eye":
        violations = [
            _count_violation("red_eye_segments", red_eye, minor_at=1, major_at=1),
            _count_violation("late_arrivals", arrives_late, minor_at=1, major_at=3),
        ]
    elif rule_id == "transport_avoid_early_departure":
        violations = [
            _count_violation("early_departures", starts_early, minor_at=1, major_at=2)
        ]
    elif rule_id == "transport_avoid_late_arrival":
        violations = [
            _count_violation("late_arrivals", arrives_late, minor_at=1, major_at=2)
        ]
    elif rule_id == "transport_avoid_transfer":
        violations = [
            _count_violation(
                "transfer_mentions", transfer_mentions, minor_at=1, major_at=2
            ),
            _count_violation(
                "structured_transfer_segments",
                structured_transfer_segments,
                minor_at=1,
                major_at=2,
            ),
            _count_violation(
                "structured_transfer_count",
                structured_transfer_count,
                minor_at=1,
                major_at=2,
            ),
        ]
    elif rule_id == "transport_prefer_train":
        train_count = sum(
            1 for mode in modes if normalize_intercity_mode(mode) == "train"
        )
        violations = [
            {
                "name": "preferred_train_mode_missing",
                "value": train_count == 0,
                "severity": "none" if train_count else "major",
            }
        ]
    elif rule_id == "transport_prefer_flight":
        flight_count = sum(
            1 for mode in modes if normalize_intercity_mode(mode) == "flight"
        )
        violations = [
            {
                "name": "preferred_flight_mode_missing",
                "value": flight_count == 0,
                "severity": "none" if flight_count else "major",
            }
        ]
    else:
        violations = []

    severity = _worst_severity(violations)
    return _level_result(
        rule_id,
        severity,
        "transport timing/mode is compatible with profile",
        "transport choice has a profile violation",
        {
            "modes": modes,
            "early_departures": starts_early,
            "late_arrivals": arrives_late,
            "red_eye_segments": red_eye,
            "transfer_mentions": transfer_mentions,
            "structured_transfer_segments": structured_transfer_segments,
            "structured_transfer_count": structured_transfer_count,
        },
        thresholds={
            "early_departure_before": "07:00",
            "late_arrival_after": "21:00",
            "red_eye_departure_before": "06:00",
            "red_eye_arrival_after": "22:00",
        },
        threshold_source="human_rubric",
        violations=violations,
    )
