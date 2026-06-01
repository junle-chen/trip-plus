"""Public visibility repair API for rendered initial-query text."""

from __future__ import annotations

import re
from typing import Any

from query_generation.common import date_text, safe_int
from query_generation.initial_query.visibility.repair import (
    _budget_phrase,
    _ensure_core_trip_fields_visible,
    _ensure_depart_date_visible,
    _ensure_exact_db_terms_visible,
    _ensure_return_date_visible,
    _ensure_selected_constraints_visible,
    _finalize_visible_query_quality,
    _harden_visible_hard_constraint_wording,
    _normalize_budget_mentions,
    normalize_query_punctuation,
    _remove_redundant_appended_hints,
    _remove_redundant_lingering_additions,
)


def ensure_visible_initial_hard_constraints(record: dict[str, Any]) -> None:
    query = normalize_query_punctuation(str(record.get("query", "")).strip())
    meta = record["meta_info"]
    org = meta.get("org")
    if org and org not in query:
        query = f"I want to depart from {org}. {query}"
    query = _ensure_depart_date_visible(query, meta)
    query = _ensure_return_date_visible(query, meta)
    query = _ensure_core_trip_fields_visible(query, meta)
    days = safe_int(meta.get("days"), 0)
    if days > 1:
        query = re.sub(rf"\b{days}\s*days?\s*\d+\s*nights?\b", f"{days} days {days - 1} nights", query, flags=re.IGNORECASE)
    query = _ensure_selected_constraints_visible(query, meta)
    query = _harden_visible_hard_constraint_wording(query, meta)

    budget = record["meta_info"].get("hard_constraints", {}).get("budget_constraint")
    if budget:
        query = _normalize_budget_mentions(query, budget)
        required_numbers = [
            str(budget[key])
            for key in ("min_budget", "max_budget")
            if key in budget
        ]
        if required_numbers and not all(number in query for number in required_numbers):
            phrase = _budget_phrase(budget)
            if phrase:
                query = re.sub(
                    r"budget[^.;,]*?\d+[^.;,]*?(?:to|-)[^.;,]*?\d+[^.;,]*",
                    phrase.rstrip("."),
                    query,
                    count=1,
                    flags=re.IGNORECASE,
                )
        if required_numbers and not all(number in query for number in required_numbers):
            phrase = _budget_phrase(budget)
            if phrase:
                record["query"] = normalize_query_punctuation(query.rstrip(". ") + ". " + phrase)
                return
    query = _ensure_selected_constraints_visible(query, meta)
    query = _ensure_exact_db_terms_visible(query, meta)
    query = _harden_visible_hard_constraint_wording(query, meta)
    query = _remove_redundant_appended_hints(query, meta)
    query = _remove_redundant_lingering_additions(query, meta)
    depart_date = str(meta.get("depart_date") or "")
    depart_visible = bool(depart_date and (depart_date in query or date_text(depart_date) in query))
    if depart_visible:
        query = re.sub(r"\bnext month\b", "", query, flags=re.IGNORECASE)
    query = _finalize_visible_query_quality(query, meta)
    record["query"] = normalize_query_punctuation(query)


def prune_initial_record_for_output(record: dict[str, Any]) -> dict[str, Any]:
    """Drop render-time/debug-only data while keeping planning/evaluation fields."""
    pruned = {
        "id": record["id"],
        "query": record["query"],
        "meta_info": dict(record["meta_info"]),
    }
    meta = pruned["meta_info"]
    for key in (
        "route_headers",
        "route_reference",
        "environmental_grounding",
        "query_category",
        "persona",
        "interaction_archetype_label",
        "t0_structure",
    ):
        meta.pop(key, None)

    environment_reference = meta.get("environment_reference")
    if isinstance(environment_reference, dict):
        environment_reference.pop("route_season", None)
        environment_reference.pop("route_manifest_id", None)
    return pruned
