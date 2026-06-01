"""Prompt construction for traveler experience simulation."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from .experience_trace import build_experience_trace

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "traveler_experience_simulation.md"


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _compact_evidence_list(items: Any, limit: int = 3) -> list[Any]:
    if not isinstance(items, list):
        return []
    compact_items = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            compact_items.append(deepcopy(item))
            continue
        compact_items.append(
            {
                key: item.get(key)
                for key in ("item_ref", "source", "claim")
                if item.get(key) not in (None, "", [], {})
            }
        )
    return compact_items


def _compact_trace_event(event: Any) -> Dict[str, Any]:
    if not isinstance(event, dict):
        return {}
    return {
        key: event.get(key)
        for key in (
            "item_ref",
            "day",
            "activity_index",
            "type",
            "name",
            "time_slot",
            "duration_minutes",
            "mode",
            "cost",
            "source",
        )
        if event.get(key) not in (None, "", [], {})
    }


def _compact_experience_facts(facts: Any) -> Dict[str, Any]:
    if not isinstance(facts, dict):
        return {}
    keep_keys = (
        "activity_type",
        "duration_minutes",
        "duration_bucket",
        "experience_flags",
        "cost",
        "budget_cost_relevance",
    )
    return {
        key: facts.get(key)
        for key in keep_keys
        if facts.get(key) not in (None, "", [], {})
    }


def _compact_user_model_for_prompt(user_model: Any) -> Dict[str, Any]:
    if not isinstance(user_model, dict):
        return {}

    party = user_model.get("party") if isinstance(user_model.get("party"), dict) else {}
    sensitivities = (
        user_model.get("sensitivities")
        if isinstance(user_model.get("sensitivities"), dict)
        else {}
    )
    positive_preferences = (
        user_model.get("positive_preferences")
        if isinstance(user_model.get("positive_preferences"), dict)
        else {}
    )
    negative_preferences = (
        user_model.get("negative_preferences")
        if isinstance(user_model.get("negative_preferences"), dict)
        else {}
    )

    return {
        key: value
        for key, value in {
            "party": {
                party_key: party.get(party_key)
                for party_key in ("has_child", "has_elder")
                if party.get(party_key) not in (None, "", [], {})
            },
            "comfort_sensitivities": {
                compact_key: sensitivities.get(raw_key)
                for compact_key, raw_key in (
                    ("fatigue", "fatigue"),
                    ("environment", "environment"),
                    ("schedule", "schedule_stress"),
                    ("budget", "budget_stress"),
                )
                if sensitivities.get(raw_key) not in (None, "", [], {})
            },
            "interest_preferences": {
                key: value
                for key, value in positive_preferences.items()
                if value is True
            },
            "sensitivity_flags": {
                key: value
                for key, value in negative_preferences.items()
                if value is True
            },
        }.items()
        if value not in (None, "", [], {})
    }


def _compact_activity_for_prompt(activity: Any) -> Dict[str, Any]:
    if not isinstance(activity, dict):
        return {}
    details = activity.get("details") if isinstance(activity.get("details"), dict) else {}
    compact_details = {
        key: details.get(key)
        for key in (
            "name",
            "from",
            "to",
            "mode",
            "transport_mode",
            "duration",
            "distance",
            "cost",
            "price",
            "ticket_price",
            "meal_type",
        )
        if details.get(key) not in (None, "", [], {})
    }
    return {
        key: value
        for key, value in {
            "type": activity.get("type"),
            "time_slot": activity.get("time_slot"),
            "details": compact_details,
        }.items()
        if value not in (None, "", [], {})
    }


def _compact_plan_for_prompt(plan: Dict[str, Any]) -> Dict[str, Any]:
    days = []
    for day in plan.get("daily_plans", []) or []:
        if not isinstance(day, dict):
            continue
        accommodation = day.get("accommodation")
        if isinstance(accommodation, dict):
            accommodation = {
                key: accommodation.get(key)
                for key in ("name", "price", "cost", "nights")
                if accommodation.get(key) not in (None, "", [], {})
            }
        compact_day = {
            key: value
            for key, value in {
                "day": day.get("day"),
                "date": day.get("date"),
                "current_city": day.get("current_city"),
                "accommodation": accommodation,
                "activities": [
                    item for item in (
                        _compact_activity_for_prompt(activity)
                        for activity in day.get("activities", []) or []
                    )
                    if item
                ],
            }.items()
            if value not in (None, "", [], {})
        }
        days.append(compact_day)
    return {
        key: value
        for key, value in {
            "daily_plans": days,
            "budget_summary": plan.get("budget_summary"),
        }.items()
        if value not in (None, "", [], {})
    }


def _compact_experience_trace_for_prompt(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only fields the LLM needs for subjective experience simulation."""

    user_model = trace.get("user_model") if isinstance(trace, dict) else {}
    if not isinstance(user_model, dict):
        user_model = {}

    compact_items = []
    for item in trace.get("activity_trace", []) or []:
        if not isinstance(item, dict):
            continue
        compact_items.append(
            {
                key: value
                for key, value in {
                    "item_ref": item.get("item_ref"),
                    "event": _compact_trace_event(item.get("event")),
                    "experience_facts": _compact_experience_facts(item.get("experience_facts")),
                }.items()
                if value not in (None, "", [], {})
            }
        )

    return {
        key: value
        for key, value in {
            "turn_context": trace.get("turn_context"),
            "user_model": _compact_user_model_for_prompt(user_model),
            "environment": trace.get("environment"),
            "budget": trace.get("budget"),
            "expected_activity_refs": trace.get("expected_activity_refs"),
            "activity_count": trace.get("activity_count"),
            "activity_trace": compact_items,
        }.items()
        if value not in (None, "", [], {})
    }


def build_user_simulator_messages(
    *,
    query_record: Dict[str, Any],
    plan: Dict[str, Any],
    turn_id: Optional[int] = None,
    evaluation_context: Optional[Dict[str, Any]] = None,
    experience_trace: Optional[Dict[str, Any]] = None,
) -> list[dict[str, str]]:
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    context = evaluation_context or {}
    trace = experience_trace or build_experience_trace(
        query_record=query_record,
        plan=plan,
        turn_id=turn_id,
        evaluation_context=context,
    )
    prompt_trace = _compact_experience_trace_for_prompt(trace)
    user_message = f"""Evaluate this sample using the system instructions. Return compact valid JSON only.

PLAN:
{_json_dumps(_compact_plan_for_prompt(plan))}

EXPERIENCE_TRACE:
{_json_dumps(prompt_trace)}
"""
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_message},
    ]
