"""Shared helpers for deterministic soft-preference checks."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..utils import extract_entity_name, normalize_entity_name, slot_to_minutes
from .config import (
    DB_ATTRACTION_FIELDS,
    INDOOR_MARKERS,
    OUTDOOR_MARKERS,
    REST_MARKERS,
    SEVERITY_RANK,
    SEVERITY_SCORES,
    SOFT_PREFERENCE_FAMILIES,
)

_CSV_CACHE: Dict[tuple[str, str, int, int], List[Dict[str, str]]] = {}


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_average(
    values: List[float], default: Optional[float] = 1.0
) -> Optional[float]:
    return sum(values) / len(values) if values else default


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _rule_metadata_by_canonical(
    meta: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    user_profile = meta.get("user_profile") or {}
    metadata: Dict[str, List[Dict[str, Any]]] = {}
    for item in user_profile.get("rules") or []:
        if not isinstance(item, dict):
            continue
        rule_id = str(item.get("rule_id") or "").strip()
        if not rule_id:
            continue
        metadata.setdefault(rule_id, []).append(item)
    return metadata


def _source_rule_ids(
    input_rule_ids: Iterable[str], rule_metadata: Optional[List[Dict[str, Any]]] = None
) -> List[str]:
    values: List[str] = []
    for rule_id in input_rule_ids:
        rule_id = str(rule_id).strip()
        if rule_id and rule_id not in values:
            values.append(rule_id)
    return values


def _rule_variants(rule_metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    values: List[str] = []
    for item in rule_metadata or []:
        for key in ("preference_variant", "preference_variants", "interest_tags"):
            for value in _as_list(item.get(key)):
                text = str(value).strip()
                if text and text not in values:
                    values.append(text)
    return values


def _preference_family_for_rule(rule_id: str) -> str:
    for family, rule_ids in SOFT_PREFERENCE_FAMILIES.items():
        if rule_id in rule_ids:
            return family
    return "other"


def _severity_from_score(score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    if score >= 0.999:
        return "none"
    if score >= 0.5:
        return "minor"
    return "major"


def _score_for_severity(severity: str) -> float:
    return SEVERITY_SCORES.get(severity, 0.0)


def _worst_severity(violations: Iterable[Dict[str, Any]]) -> str:
    worst = "none"
    for item in violations:
        severity = str(item.get("severity") or "none")
        if SEVERITY_RANK.get(severity, 0) > SEVERITY_RANK[worst]:
            worst = severity
    return worst


def _above_violation(
    name: str,
    value: float,
    *,
    soft_boundary: float,
    hard_boundary: float,
    unit: str = "",
) -> Dict[str, Any]:
    if value <= soft_boundary:
        severity = "none"
    elif value <= hard_boundary:
        severity = "minor"
    else:
        severity = "major"
    return {
        "name": name,
        "value": round(value, 4),
        "soft_boundary": soft_boundary,
        "hard_boundary": hard_boundary,
        "severity": severity,
        "unit": unit,
    }


def _count_violation(
    name: str, value: int, *, minor_at: int = 1, major_at: int = 2
) -> Dict[str, Any]:
    if value < minor_at:
        severity = "none"
    elif value < major_at:
        severity = "minor"
    else:
        severity = "major"
    return {
        "name": name,
        "value": value,
        "minor_at": minor_at,
        "major_at": major_at,
        "severity": severity,
    }


def _level_result(
    rule_id: str,
    severity: str,
    message_none: str,
    message_violation: str,
    signals: Dict[str, Any],
    *,
    thresholds: Optional[Dict[str, Any]] = None,
    threshold_source: str = "not_threshold_based",
    threshold_sources: Optional[Dict[str, str]] = None,
    violations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    score = _score_for_severity(severity)
    signals = dict(signals)
    if violations is not None:
        signals["violations"] = violations
    return _result(
        rule_id,
        score,
        severity == "none",
        message_none if severity == "none" else message_violation,
        signals,
        thresholds=thresholds,
        threshold_source=threshold_source,
        threshold_sources=threshold_sources,
        severity=severity,
        violations=violations,
    )


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _transfer_count_from_details(details: Dict[str, Any]) -> int:
    value = None
    for key in ("transfers", "transfer_count", "connection_count", "layover_count"):
        if key in details:
            value = details.get(key)
            break
    if value in (None, ""):
        return 0
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return max(0, int(value))
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for key in ("count", "total"):
            parsed = _safe_float(value.get(key))
            if parsed is not None:
                return max(0, int(parsed))
        return len(value)
    text = str(value).strip().lower()
    if not text or text in {
        "none",
        "no",
        "false",
        "direct",
        "nonstop",
        "non-stop",
        "no transfer",
    }:
        return 0
    parsed = _safe_float(text)
    if parsed is not None:
        return max(0, int(parsed))
    return 1 if _contains_any(text, ("transfer", "connection", "layover")) else 0


def _database_path(database_dir: Optional[Path | str]) -> Optional[Path]:
    if database_dir in (None, ""):
        return None
    path = Path(database_dir)
    return path if path.exists() else None


def _read_db_rows(
    database_dir: Optional[Path | str], relative_path: str
) -> List[Dict[str, str]]:
    root = _database_path(database_dir)
    if root is None:
        return []
    path = root / relative_path
    try:
        stat = path.stat()
    except OSError:
        return []
    key = (str(path.resolve()), relative_path, int(stat.st_size), int(stat.st_mtime_ns))
    cached = _CSV_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        rows = []
    _CSV_CACHE[key] = rows
    return rows


def _quantile(values: List[float], q: float) -> Optional[float]:
    cleaned = sorted(value for value in values if math.isfinite(value))
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    pos = (len(cleaned) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return cleaned[lo]
    return cleaned[lo] + (cleaned[hi] - cleaned[lo]) * (pos - lo)


def _numeric_field_quantiles(
    database_dir: Optional[Path | str],
    relative_path: str,
    fields: Iterable[str],
    qs: Dict[str, float],
) -> tuple[Dict[str, float], str]:
    rows = _read_db_rows(database_dir, relative_path)
    values: List[float] = []
    for row in rows:
        for field in fields:
            value = _safe_float(row.get(field))
            if value is not None:
                values.append(value)
                break
    result: Dict[str, float] = {}
    for name, q in qs.items():
        value = _quantile(values, q)
        if value is not None:
            result[name] = round(value, 4)
    source = "sample_database_quantile" if result else "static_fallback"
    return result, source


def _record_index(
    database_dir: Optional[Path | str],
    relative_path: str,
    name_fields: Iterable[str],
) -> Dict[str, Dict[str, str]]:
    rows = _read_db_rows(database_dir, relative_path)
    index: Dict[str, Dict[str, str]] = {}
    for row in rows:
        for field in name_fields:
            name = str(row.get(field) or "").strip()
            if name:
                index[normalize_entity_name(name)] = row
    return index


def _db_record_for_activity(
    act: Dict[str, Any],
    database_dir: Optional[Path | str],
    *,
    kind: str,
) -> Optional[Dict[str, str]]:
    name = normalize_entity_name(extract_entity_name(act))
    if not name:
        return None
    if kind == "attraction":
        return _record_index(
            database_dir, "attractions/attractions.csv", ("attraction_name", "name")
        ).get(name)
    if kind == "restaurant":
        return _record_index(
            database_dir, "restaurants/restaurants.csv", ("restaurant_name", "name")
        ).get(name)
    return None


def _row_text(row: Optional[Dict[str, str]], fields: Iterable[str]) -> str:
    if not row:
        return ""
    return " ".join(str(row.get(field) or "") for field in fields).strip()


def _duration_minutes(act: Dict[str, Any]) -> int:
    start, end = slot_to_minutes(act.get("time_slot"))
    if start is not None and end is not None:
        return max(0, end - start)
    details = act.get("details") or {}
    duration = _safe_float(details.get("duration"))
    return int(duration or 0)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _contains_any(text: str, markers: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def _iter_activities(plan: Dict[str, Any]) -> Iterable[tuple[int, Dict[str, Any]]]:
    for day_idx, day in enumerate(plan.get("daily_plans", []) or [], start=1):
        for act in day.get("activities", []) or []:
            if isinstance(act, dict):
                yield day_idx, act


def _daily_activities(plan: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    return [
        [act for act in day.get("activities", []) or [] if isinstance(act, dict)]
        for day in plan.get("daily_plans", []) or []
        if isinstance(day, dict)
    ]


def _attraction_visit_count(activities: List[Dict[str, Any]]) -> int:
    return sum(1 for act in activities if act.get("type") == "attraction")


def _has_rest_block(activities: List[Dict[str, Any]]) -> bool:
    for act in activities:
        act_type = act.get("type")
        text = _json_text(act)
        duration = _duration_minutes(act)
        if act_type == "hotel" and duration >= 30 and _contains_any(text, REST_MARKERS):
            return True
        if act_type == "buffer" and duration >= 40:
            return True
    return False


def _day_span_minutes(activities: List[Dict[str, Any]]) -> int:
    starts: List[int] = []
    ends: List[int] = []
    for act in activities:
        start, end = slot_to_minutes(act.get("time_slot"))
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)
    if not starts or not ends:
        return 0
    return max(ends) - min(starts)


def _first_start_minutes(activities: List[Dict[str, Any]]) -> Optional[int]:
    starts = [slot_to_minutes(act.get("time_slot"))[0] for act in activities]
    starts = [value for value in starts if value is not None]
    return min(starts) if starts else None


def _last_end_minutes(activities: List[Dict[str, Any]]) -> Optional[int]:
    ends = [slot_to_minutes(act.get("time_slot"))[1] for act in activities]
    ends = [value for value in ends if value is not None]
    return max(ends) if ends else None


def _local_transfer_minutes(plan: Dict[str, Any]) -> List[int]:
    return [
        _duration_minutes(act)
        for _day, act in _iter_activities(plan)
        if act.get("type") == "travel_city"
    ]


def _is_outdoor_attraction(
    act: Dict[str, Any], database_dir: Optional[Path | str] = None
) -> bool:
    row = _db_record_for_activity(act, database_dir, kind="attraction")
    db_text = _row_text(row, DB_ATTRACTION_FIELDS)
    if db_text:
        if _contains_any(db_text, INDOOR_MARKERS):
            return False
        if _contains_any(db_text, OUTDOOR_MARKERS):
            return True
    text = _json_text(act)
    duration = _duration_minutes(act)
    if _contains_any(text, INDOOR_MARKERS):
        return False
    return _contains_any(text, OUTDOOR_MARKERS) or duration >= 150


def _outdoor_minutes_by_day(
    plan: Dict[str, Any],
    database_dir: Optional[Path | str] = None,
    *,
    midday_only: bool = False,
    late_only: bool = False,
) -> List[int]:
    totals: Dict[int, int] = {}
    for day_idx, act in _iter_activities(plan):
        if act.get("type") != "attraction":
            continue
        duration = _duration_minutes(act)
        if not _is_outdoor_attraction(act, database_dir):
            continue
        start, end = slot_to_minutes(act.get("time_slot"))
        if midday_only and start is not None and end is not None:
            overlap = max(0, min(end, 15 * 60) - max(start, 11 * 60))
            duration = overlap
        if late_only and end is not None:
            duration = max(0, end - max(start or end, 18 * 60))
        totals[day_idx] = totals.get(day_idx, 0) + max(0, duration)
    return list(totals.values())


def _city_tags(meta: Dict[str, Any]) -> set[str]:
    """Collect active city/environment tags from query metadata.

    Weather-sensitive checks rely on these tags to decide whether heat, cold,
    rain, or other environment-aware preferences are relevant for the sample.
    """
    tags: set[str] = set()
    for source in (
        meta.get("city_context") or {},
        meta.get("environment_reference") or {},
    ):
        if not isinstance(source, dict):
            continue
        for key in ("city_tags", "seasonal_advisories", "environment_tags"):
            value = source.get(key)
            if isinstance(value, list):
                tags.update(str(item) for item in value if str(item).strip())
    return tags


def _hard_ratio(hard_result: Dict[str, Any]) -> tuple[float, int, int]:
    """Compute a continuous hard-constraint ratio from per-constraint results.

    ``hard_result["score"]`` is often an all-pass style score. User alignment
    uses this ratio so one failed hard check does not hide the soft-preference
    diagnostics in the final requirement details.
    """
    constraints = hard_result.get("constraints") or {}
    if not constraints:
        return 1.0, 0, 0
    total = 0
    passed = 0
    for detail in constraints.values():
        if not isinstance(detail, dict):
            continue
        total += 1
        if detail.get("passed"):
            passed += 1
    if not total:
        return 1.0, 0, 0
    return passed / total, passed, total


def _evidence_basis_for_rule(rule_id: str) -> List[str]:
    if rule_id.startswith("transport_"):
        return [
            "travel_intercity_public.time_slot",
            "details.mode",
            "details.transfers",
        ]
    if rule_id.startswith("weather_"):
        return [
            "city_context.city_tags",
            "attraction.time_slot",
            "database.attraction_type/tags",
            "outdoor exposure",
        ]
    if rule_id.startswith("interest_"):
        return [
            "exact DB entity match",
            "database.attraction_type/tags",
            "database.restaurant.cuisine/tags",
        ]
    if rule_id == "hotel_value_first":
        return [
            "accommodation.price_per_night",
            "sample database hotel price quantiles",
            "computed_cost.accommodation",
        ]
    if rule_id.startswith("budget_") or rule_id == "meal_avoid_expensive":
        return [
            "computed_cost.total",
            "explicit budget constraint",
            "meal.details.cost",
            "sample database price quantiles",
        ]
    return ["daily attraction count", "travel_city.duration", "rest blocks"]


def _result(
    rule_id: str,
    score: Optional[float],
    passed: Optional[bool],
    message: str,
    signals: Dict[str, Any],
    *,
    thresholds: Optional[Dict[str, Any]] = None,
    threshold_source: str = "not_threshold_based",
    threshold_sources: Optional[Dict[str, str]] = None,
    applicable: bool = True,
    not_applicable_reason: Optional[str] = None,
    severity: Optional[str] = None,
    violations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    severity = severity if severity is not None else _severity_from_score(score)
    result = {
        "rule_id": rule_id,
        "applicable": applicable,
        "score": round(_clip01(score), 4) if score is not None else None,
        "passed": passed,
        "severity": severity,
        "violation_level": severity,
        "message": message,
        "signals": signals,
        "raw_signals": signals,
        "thresholds": thresholds or {},
        "threshold_source": threshold_source,
    }
    if violations is not None:
        result["violations"] = violations
    if threshold_sources:
        result["threshold_sources"] = threshold_sources
    if not_applicable_reason:
        result["not_applicable_reason"] = not_applicable_reason
    return result


def _not_applicable(
    rule_id: str, reason: str, signals: Dict[str, Any]
) -> Dict[str, Any]:
    return _result(
        rule_id,
        None,
        None,
        reason,
        signals,
        applicable=False,
        not_applicable_reason=reason,
        threshold_source="not_applicable",
    )
