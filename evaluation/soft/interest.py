"""Interest-matching soft-preference checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .common import (
    _contains_any,
    _db_record_for_activity,
    _iter_activities,
    _json_text,
    _result,
    _row_text,
    _rule_variants,
    _score_for_severity,
    _source_rule_ids,
)
from .config import (
    AMUSEMENT_MARKERS,
    ART_MARKERS,
    DB_ATTRACTION_FIELDS,
    DB_RESTAURANT_FIELDS,
    FOOD_MARKERS,
    HISTORY_MARKERS,
    LANDMARK_MARKERS,
    LOCAL_FOOD_DB_MARKERS,
    MUSEUM_MARKERS,
    NATURE_MARKERS,
    PARK_MARKERS,
    SHOPPING_MARKERS,
)


def _score_interest(
    plan: Dict[str, Any],
    rule_id: str,
    database_dir: Optional[Path | str] = None,
    source_rule_ids: Optional[Iterable[str]] = None,
    rule_metadata: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Score whether attractions or meals provide evidence for profile interests."""
    source_rule_ids = _source_rule_ids(source_rule_ids or [], rule_metadata)
    attraction_acts = [
        act for _day, act in _iter_activities(plan) if act.get("type") == "attraction"
    ]
    meal_acts = [
        act for _day, act in _iter_activities(plan) if act.get("type") == "meal"
    ]

    marker_map = {
        "interest_local_food": LOCAL_FOOD_DB_MARKERS,
        "interest_outdoor_nature": tuple(
            dict.fromkeys((*NATURE_MARKERS, *PARK_MARKERS))
        ),
        "interest_culture": tuple(dict.fromkeys((*HISTORY_MARKERS, *MUSEUM_MARKERS))),
        "interest_art": ART_MARKERS,
        "interest_shopping": SHOPPING_MARKERS,
        "interest_landmark": LANDMARK_MARKERS,
        "interest_amusement": AMUSEMENT_MARKERS,
    }
    markers = marker_map.get(rule_id, ())
    if rule_id == "interest_local_food":
        matched = 0
        exact_db_matches = 0
        fallback_matches = 0
        for act in meal_acts:
            row = _db_record_for_activity(act, database_dir, kind="restaurant")
            db_text = _row_text(row, DB_RESTAURANT_FIELDS)
            if db_text:
                exact_db_matches += 1
                if _contains_any(db_text, markers):
                    matched += 1
            elif database_dir is None and _contains_any(_json_text(act), FOOD_MARKERS):
                matched += 1
                fallback_matches += 1
        denominator = max(1, min(2, len(meal_acts)))
    else:
        matched = 0
        exact_db_matches = 0
        fallback_matches = 0
        for act in attraction_acts:
            row = _db_record_for_activity(act, database_dir, kind="attraction")
            db_text = _row_text(row, DB_ATTRACTION_FIELDS)
            if db_text:
                exact_db_matches += 1
                if _contains_any(db_text, markers):
                    matched += 1
            elif database_dir is None and _contains_any(_json_text(act), markers):
                matched += 1
                fallback_matches += 1
        denominator = max(1, min(2, len(attraction_acts)))
    if matched >= denominator:
        severity = "none"
    elif matched > 0:
        severity = "minor"
    else:
        severity = "major"
    score = _score_for_severity(severity)
    threshold_source = (
        "database_type_or_tags_exact_match"
        if database_dir
        else "plan_text_marker_fallback_no_database"
    )
    return _result(
        rule_id,
        score,
        severity == "none",
        "plan includes profile-matching interest content"
        if severity == "none"
        else "plan has weak evidence for this interest",
        {
            "matched_items": matched,
            "candidate_items": len(
                meal_acts if rule_id == "interest_local_food" else attraction_acts
            ),
            "exact_db_entity_matches": exact_db_matches,
            "fallback_marker_matches": fallback_matches,
            "markers": list(markers),
            "source_rule_ids": source_rule_ids,
            "preference_variants": _rule_variants(rule_metadata),
        },
        thresholds={"min_matching_items": denominator},
        threshold_source=threshold_source,
        severity=severity,
        violations=[
            {
                "name": "profile_interest_matched_items",
                "value": matched,
                "target": denominator,
                "severity": severity,
            }
        ],
    )
