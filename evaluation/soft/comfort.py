"""Comfort, pacing, mobility, and weather soft-preference checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .common import (
    _above_violation,
    _attraction_visit_count,
    _city_tags,
    _contains_any,
    _count_violation,
    _daily_activities,
    _day_span_minutes,
    _duration_minutes,
    _first_start_minutes,
    _has_rest_block,
    _iter_activities,
    _json_text,
    _last_end_minutes,
    _level_result,
    _local_transfer_minutes,
    _not_applicable,
    _outdoor_minutes_by_day,
    _rule_variants,
    _safe_average,
    _source_rule_ids,
    _worst_severity,
)
from .config import INDOOR_MARKERS


def _schedule_variant(
    source_rule_ids: Iterable[str], rule_metadata: Optional[List[Dict[str, Any]]] = None
) -> str:
    variants = set(_rule_variants(rule_metadata))
    if "relaxed" in variants:
        return "relaxed"
    if "dense" in variants:
        return "dense"
    return "moderate"


def _score_schedule(
    plan: Dict[str, Any],
    rule_id: str,
    source_rule_ids: Optional[Iterable[str]] = None,
    rule_metadata: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Score whether daily pacing matches the profile's pacing variant."""
    source_rule_ids = list(source_rule_ids or [])
    pacing_variant = _schedule_variant(source_rule_ids, rule_metadata)
    days = _daily_activities(plan)
    counts = [_attraction_visit_count(acts) for acts in days]
    spans = [_day_span_minutes(acts) for acts in days]
    early_starts = sum(
        1
        for acts in days
        if (start := _first_start_minutes(acts)) is not None and start < 8 * 60
    )
    late_ends = sum(
        1
        for acts in days
        if (end := _last_end_minutes(acts)) is not None and end > 21 * 60
    )
    rest_days = sum(1 for acts in days if _has_rest_block(acts))
    max_count = max(counts or [0])
    avg_count = _safe_average([float(value) for value in counts], default=0.0)
    max_span_hours = max((span / 60 for span in spans), default=0.0)
    earliest_start = min(
        (start for acts in days if (start := _first_start_minutes(acts)) is not None),
        default=None,
    )
    latest_end = max(
        (end for acts in days if (end := _last_end_minutes(acts)) is not None),
        default=None,
    )

    if pacing_variant == "relaxed":
        thresholds = {
            "max_daily_attraction_count": {"none": "<=3", "minor": "4", "major": ">=5"},
            "avg_daily_attraction_count": {
                "none": "<=3",
                "minor": ">3 and <=4",
                "major": ">4",
            },
            "max_day_span_hours": {
                "none": "<=10",
                "minor": ">10 and <=12",
                "major": ">12",
            },
            "early_start": {
                "none": ">=08:00",
                "minor": "07:00-08:00",
                "major": "<07:00",
            },
            "late_end": {"none": "<=21:00", "minor": "21:00-22:00", "major": ">22:00"},
            "rest_block": {
                "none": "present",
                "minor": "missing",
                "major": "missing with dense day",
            },
        }
        violations = [
            _above_violation(
                "max_daily_attraction_count",
                float(max_count),
                soft_boundary=3,
                hard_boundary=4,
                unit="visits",
            ),
            _above_violation(
                "avg_daily_attraction_count",
                float(avg_count or 0),
                soft_boundary=3,
                hard_boundary=4,
                unit="visits",
            ),
            _above_violation(
                "max_day_span_hours",
                max_span_hours,
                soft_boundary=10,
                hard_boundary=12,
                unit="hours",
            ),
        ]
        if earliest_start is not None:
            early_value = max(0.0, (8 * 60 - earliest_start) / 60)
            violations.append(
                _above_violation(
                    "early_start_hours_before_08",
                    early_value,
                    soft_boundary=0,
                    hard_boundary=1,
                    unit="hours",
                )
            )
        if latest_end is not None:
            late_value = max(0.0, (latest_end - 21 * 60) / 60)
            violations.append(
                _above_violation(
                    "late_end_hours_after_21",
                    late_value,
                    soft_boundary=0,
                    hard_boundary=1,
                    unit="hours",
                )
            )
        if days and rest_days == 0:
            violations.append(
                {
                    "name": "rest_block_missing",
                    "value": True,
                    "severity": "major" if max_count >= 5 else "minor",
                }
            )
    elif pacing_variant == "moderate":
        thresholds = {
            "max_daily_attraction_count": {
                "none": "<=4",
                "minor": "5-6",
                "major": ">=7",
            },
            "avg_daily_attraction_count": {
                "none": "<=4",
                "minor": ">4 and <=5",
                "major": ">5",
            },
            "max_day_span_hours": {
                "none": "<=12",
                "minor": ">12 and <=13.5",
                "major": ">13.5",
            },
            "late_end": {"none": "<=21:00", "minor": "21:00-22:00", "major": ">22:00"},
        }
        violations = [
            _above_violation(
                "max_daily_attraction_count",
                float(max_count),
                soft_boundary=4,
                hard_boundary=6,
                unit="visits",
            ),
            _above_violation(
                "avg_daily_attraction_count",
                float(avg_count or 0),
                soft_boundary=4,
                hard_boundary=5,
                unit="visits",
            ),
            _above_violation(
                "max_day_span_hours",
                max_span_hours,
                soft_boundary=12,
                hard_boundary=13.5,
                unit="hours",
            ),
        ]
        if latest_end is not None:
            late_value = max(0.0, (latest_end - 21 * 60) / 60)
            violations.append(
                _above_violation(
                    "late_end_hours_after_21",
                    late_value,
                    soft_boundary=0,
                    hard_boundary=1,
                    unit="hours",
                )
            )
    else:
        thresholds = {
            "extreme_daily_attraction_count": {
                "none": "<=6",
                "minor": "7-8",
                "major": ">=9",
            },
            "extreme_day_span_hours": {
                "none": "<=14",
                "minor": ">14 and <=16",
                "major": ">16",
            },
            "late_end": {"none": "<=22:00", "minor": "22:00-23:00", "major": ">23:00"},
        }
        violations = [
            _above_violation(
                "max_daily_attraction_count",
                float(max_count),
                soft_boundary=6,
                hard_boundary=8,
                unit="visits",
            ),
            _above_violation(
                "max_day_span_hours",
                max_span_hours,
                soft_boundary=14,
                hard_boundary=16,
                unit="hours",
            ),
        ]
        if latest_end is not None:
            late_value = max(0.0, (latest_end - 22 * 60) / 60)
            violations.append(
                _above_violation(
                    "late_end_hours_after_22",
                    late_value,
                    soft_boundary=0,
                    hard_boundary=1,
                    unit="hours",
                )
            )

    severity = _worst_severity(violations)
    return _level_result(
        rule_id,
        severity,
        "schedule pacing is compatible with profile",
        "schedule has a profile pacing violation",
        {
            "daily_attraction_counts": counts,
            "rest_days": rest_days,
            "early_starts": early_starts,
            "late_ends": late_ends,
            "max_day_span_hours": round(max_span_hours, 4),
            "pacing_variant": pacing_variant,
            "source_rule_ids": source_rule_ids,
        },
        thresholds=thresholds,
        threshold_source="benchmark_design_rubric",
        violations=violations,
    )


def _transport_duration_thresholds(
    database_dir: Optional[Path | str],
) -> tuple[Dict[str, float], str]:
    return {
        "long_local_transfer_minutes": 60.0,
        "very_long_local_transfer_minutes": 90.0,
    }, "benchmark_design_rubric"


def _score_mobility(
    plan: Dict[str, Any],
    rule_id: str,
    database_dir: Optional[Path | str] = None,
    source_rule_ids: Optional[Iterable[str]] = None,
    rule_metadata: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Score walking and local-transfer burden from structured activities."""
    source_rule_ids = _source_rule_ids(source_rule_ids or [], rule_metadata)
    variants = set(_rule_variants(rule_metadata))
    transfers = _local_transfer_minutes(plan)
    transfer_thresholds, transfer_source = _transport_duration_thresholds(database_dir)
    long_threshold = transfer_thresholds["long_local_transfer_minutes"]
    very_long_threshold = transfer_thresholds["very_long_local_transfer_minutes"]
    long_transfers = sum(1 for value in transfers if value > long_threshold)
    very_long_transfers = sum(1 for value in transfers if value > very_long_threshold)
    total_by_day: Dict[int, int] = {}
    for day_idx, act in _iter_activities(plan):
        if act.get("type") == "travel_city":
            total_by_day[day_idx] = total_by_day.get(day_idx, 0) + _duration_minutes(
                act
            )
    outdoor_totals = _outdoor_minutes_by_day(plan, database_dir)
    thresholds = {
        **transfer_thresholds,
        "max_daily_local_transfer_minutes": 240,
        "high_outdoor_minutes": 240,
        "very_high_outdoor_minutes": 360,
    }
    threshold_sources = {
        "long_local_transfer_minutes": transfer_source,
        "very_long_local_transfer_minutes": transfer_source,
        "max_daily_local_transfer_minutes": "benchmark_design_rubric",
        "high_outdoor_minutes": "human_rubric_with_db_outdoor_classification",
        "very_high_outdoor_minutes": "human_rubric_with_db_outdoor_classification",
    }

    check_route = not variants or "route_transfer_sensitive" in variants
    check_walking = not variants or "walking_sensitive" in variants
    violations: List[Dict[str, Any]] = []
    avg_transfer = _safe_average(
        [float(value) for value in total_by_day.values()], default=0.0
    )
    if check_route:
        violations.extend(
            [
                _above_violation(
                    "max_daily_local_transfer_minutes",
                    float(max(total_by_day.values() or [0])),
                    soft_boundary=240,
                    hard_boundary=300,
                    unit="minutes",
                ),
                _count_violation(
                    "route_long_local_transfer_count",
                    long_transfers,
                    minor_at=2,
                    major_at=4,
                ),
                _count_violation(
                    "route_very_long_local_transfer_count",
                    very_long_transfers,
                    minor_at=1,
                    major_at=2,
                ),
            ]
        )
    if check_walking:
        violations.extend(
            [
                _count_violation(
                    "mobility_long_local_transfer_count",
                    long_transfers,
                    minor_at=1,
                    major_at=3,
                ),
                _count_violation(
                    "mobility_very_long_local_transfer_count",
                    very_long_transfers,
                    minor_at=1,
                    major_at=2,
                ),
                _above_violation(
                    "max_daily_outdoor_minutes",
                    float(max(outdoor_totals or [0])),
                    soft_boundary=240,
                    hard_boundary=360,
                    unit="minutes",
                ),
            ]
        )

    severity = _worst_severity(violations)
    return _level_result(
        rule_id,
        severity,
        "mobility burden is compatible with profile",
        "local movement or outdoor load has a profile violation",
        {
            "long_local_transfers": long_transfers,
            "very_long_local_transfers": very_long_transfers,
            "daily_local_transfer_minutes": total_by_day,
            "avg_daily_local_transfer_minutes": round(float(avg_transfer or 0), 4),
            "daily_outdoor_minutes": outdoor_totals,
            "source_rule_ids": source_rule_ids,
        },
        thresholds=thresholds,
        threshold_source="benchmark_design_rubric",
        threshold_sources=threshold_sources,
        violations=violations,
    )


def _score_weather(
    plan: Dict[str, Any],
    meta: Dict[str, Any],
    rule_id: str,
    database_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Score exposure mitigation for heat, cold, or general weather risk.

    If the city/environment metadata does not activate the relevant risk, the
    rule is marked non-applicable and excluded from the soft denominator.
    """
    tags = _city_tags(meta)
    tags_text = " ".join(tags).lower()
    heat_risk = any(token in tags_text for token in ("heat", "hot", "high_temperature"))
    cold_risk = any(
        token in tags_text
        for token in ("cold", "winter", "snow", "ice", "low_temperature")
    )
    weather_risk = (
        heat_risk
        or cold_risk
        or any(token in tags_text for token in ("rain", "typhoon", "storm"))
    )

    if rule_id == "weather_avoid_heat_exposure" and not heat_risk:
        return _not_applicable(
            rule_id,
            "No heat-risk city/date tag is active.",
            {"city_tags": sorted(tags)},
        )
    if rule_id == "weather_avoid_cold_exposure" and not cold_risk:
        return _not_applicable(
            rule_id,
            "No cold-risk city/date tag is active.",
            {"city_tags": sorted(tags)},
        )
    if rule_id == "weather_need_backup" and not weather_risk:
        return _not_applicable(
            rule_id,
            "No major weather-risk city/date tag is active.",
            {"city_tags": sorted(tags)},
        )

    midday = max(_outdoor_minutes_by_day(plan, database_dir, midday_only=True) or [0])
    total = max(_outdoor_minutes_by_day(plan, database_dir) or [0])
    late = max(_outdoor_minutes_by_day(plan, database_dir, late_only=True) or [0])
    text = _json_text(plan)
    thresholds = {
        "midday_heat_outdoor_minutes": {
            "none": "<=90",
            "minor": ">90 and <=120",
            "major": ">120",
        },
        "total_heat_outdoor_minutes": {
            "none": "<=240",
            "minor": ">240 and <=300",
            "major": ">300",
        },
        "cold_outdoor_minutes": {
            "none": "<=240",
            "minor": ">240 and <=300",
            "major": ">300",
        },
        "late_cold_outdoor_minutes": {
            "none": "<=90",
            "minor": ">90 and <=120",
            "major": ">120",
        },
        "backup_weather_outdoor_minutes": {
            "none": "<=240",
            "minor": ">240 and <=300",
            "major": ">300",
        },
    }

    if rule_id == "weather_avoid_heat_exposure":
        violations = [
            _above_violation(
                "midday_heat_outdoor_minutes",
                float(midday),
                soft_boundary=90,
                hard_boundary=120,
                unit="minutes",
            ),
            _above_violation(
                "total_heat_outdoor_minutes",
                float(total),
                soft_boundary=240,
                hard_boundary=300,
                unit="minutes",
            ),
        ]
    elif rule_id == "weather_avoid_cold_exposure":
        violations = [
            _above_violation(
                "cold_outdoor_minutes",
                float(total),
                soft_boundary=240,
                hard_boundary=300,
                unit="minutes",
            ),
            _above_violation(
                "late_cold_outdoor_minutes",
                float(late),
                soft_boundary=90,
                hard_boundary=120,
                unit="minutes",
            ),
        ]
    else:
        indoor_or_backup = _contains_any(text, INDOOR_MARKERS)
        violations = [
            _above_violation(
                "backup_weather_outdoor_minutes",
                float(total),
                soft_boundary=240,
                hard_boundary=300,
                unit="minutes",
            ),
            {
                "name": "backup_or_indoor_option_missing",
                "value": not indoor_or_backup,
                "severity": "minor"
                if not indoor_or_backup and total <= 300
                else "major"
                if not indoor_or_backup
                else "none",
            },
        ]

    severity = _worst_severity(violations)
    return _level_result(
        rule_id,
        severity,
        "weather exposure is handled for the active profile",
        "weather exposure has a profile violation",
        {
            "city_tags": sorted(tags),
            "max_midday_outdoor_minutes": midday,
            "max_outdoor_minutes": total,
            "max_late_outdoor_minutes": late,
        },
        thresholds=thresholds,
        threshold_source="human_rubric_with_db_outdoor_classification",
        violations=violations,
    )
