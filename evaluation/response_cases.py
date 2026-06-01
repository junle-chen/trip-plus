"""Result builders for non-standard response modes."""

from __future__ import annotations

import re
from typing import Any, Dict, List


def _contains_any(text: str, terms: List[str]) -> bool:
    normalized = re.sub(r"\s+", "", text.lower())
    return any(re.sub(r"\s+", "", term.lower()) in normalized for term in terms if term)


def _unsat_reason_semantic_match(
    reason_text: str, unsat_reason: Dict[str, Any]
) -> bool:
    """Match natural-language no-solution explanations against structured causes."""
    if not reason_text.strip():
        return False
    oracle_text = " ".join(
        str(item)
        for item in [
            unsat_reason.get("reason", ""),
            *(unsat_reason.get("reason_keywords") or []),
            *(unsat_reason.get("blocking_constraints") or []),
        ]
    )
    budget_oracle = _contains_any(
        oracle_text,
        [
            "budget",
            "cost",
            "price",
            "minimum cost",
            "total budget",
            "fare",
            "upper limit",
            "per night",
            "per person",
        ],
    )
    if not budget_oracle:
        return False

    has_budget = _contains_any(
        reason_text,
        [
            "budget",
            "cost",
            "price",
            "total budget",
            "fare",
            "upper limit",
            "per night",
            "per person",
            "over budget",
            "exceed",
            "insufficient",
        ],
    )
    has_infeasible = _contains_any(
        reason_text,
        [
            "unsat",
            "infeasible",
            "no solution",
            "cannot satisfy",
            "not feasible",
            "impossible",
        ],
    )
    mentions_constraint = _contains_any(
        reason_text,
        [
            "hard constraint",
            "constraint",
            "preserve",
            "unchanged",
            "do not relax",
            "previous requirement",
            "requirement",
        ],
    )
    mentions_cost_evidence = _contains_any(
        reason_text,
        [
            "transport",
            "accommodation",
            "hotel",
            "flight",
            "train",
            "ticket",
            "meal",
            "restaurant",
            "only",
            "minimum",
        ],
    )
    return has_budget and has_infeasible and (mentions_constraint or mentions_cost_evidence)


def evaluate_unsat_case(plan: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Score a no-solution response against its expected infeasibility reason."""
    unsat_reason = meta.get("unsat_reason") or {}
    expected_keywords = [
        str(keyword).strip()
        for keyword in unsat_reason.get("reason_keywords", [])
        if str(keyword).strip()
    ]
    reason_text = " ".join(
        str(plan.get(field, "")).strip()
        for field in ("unsat_explanation", "reason", "message", "summary")
    )
    normalized_reason = re.sub(r"\s+", "", reason_text.lower())
    status = str(plan.get("status", "")).strip().lower()
    has_itinerary = bool(plan.get("daily_plans"))

    reason_ok = False
    matched_keywords: list[str] = []
    if expected_keywords:
        matched_keywords = [
            keyword
            for keyword in expected_keywords
            if keyword.lower().replace(" ", "") in normalized_reason
        ]
        reason_ok = len(matched_keywords) >= min(
            2, len(expected_keywords)
        ) or _unsat_reason_semantic_match(reason_text, unsat_reason)
    elif reason_text:
        reason_ok = True

    passed = True
    message = None
    if status != "unsat":
        passed = False
        message = "Expected an unsat response with status='unsat'."
    elif has_itinerary:
        passed = False
        message = "Unsat query should not return daily_plans."
    elif not reason_text:
        passed = False
        message = "Missing unsat explanation."
    elif not reason_ok:
        passed = False
        message = (
            f"Unsat explanation does not match ground-truth reason keywords. "
            f"Expected keywords: {expected_keywords}, matched: {matched_keywords}"
        )

    final_score = 1.0 if passed else 0.0
    return {
        "scores": {
            "feasibility_score": final_score,
            "strict_feasibility": final_score,
            "hard_constraint_score": final_score,
            "strict_hard_constraint": final_score,
            "requirement_score": final_score,
            "llm_user_simulation_score": None,
        },
        "hard_constraint_dimension_score": {
            "score": final_score,
            "constraints": {
                "unsat_reason_match": {
                    "passed": passed,
                    "message": message,
                }
            },
        },
    }


def build_unsat_result(sample_id: str, special_eval: Dict[str, Any]) -> Dict[str, Any]:
    """Build the score payload for a correct no-solution style response."""
    feasibility_score = special_eval["scores"]["feasibility_score"]
    requirement_score = special_eval["scores"]["hard_constraint_score"]
    requirement_details = special_eval["hard_constraint_dimension_score"]
    if isinstance(requirement_details, dict):
        requirement_details = {
            **requirement_details,
            "strict_hard_constraint": 1.0 if requirement_score >= 1.0 - 1e-9 else 0.0,
            "hard_constraint_score": requirement_score,
            "strict_soft_preference": None,
            "soft_preference_score": None,
        }
    return {
        "sample_id": sample_id,
        "evaluation_mode": "unsat_reasoning",
        "scores": {
            "feasibility_score": feasibility_score,
            "strict_feasibility": 1.0 if feasibility_score >= 1.0 - 1e-9 else 0.0,
            "strict_hard_constraint": 1.0 if requirement_score >= 1.0 - 1e-9 else 0.0,
            "hard_constraint_score": requirement_score,
            "strict_soft_preference": None,
            "soft_preference_score": None,
            "requirement_score": requirement_score,
            "llm_user_simulation_score": None,
        },
        "feasibility_details": {
            "score": feasibility_score,
            "strict_feasibility": 1.0 if feasibility_score >= 1.0 - 1e-9 else 0.0,
            "dimensions": {},
            "subdimensions": {},
            "checks": {},
        },
        "requirement_details": requirement_details,
        "diagnostics": {
            "delivered_itinerary": False,
        },
    }


def build_missing_itinerary_result(
    sample_id: str,
    plan: Dict[str, Any],
    hard_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the score payload when a satisfiable query returned no itinerary."""
    actual_mode = str(plan.get("status") or "missing").strip().lower()
    message = (
        f"Expected an executable itinerary with daily_plans for a satisfiable query, "
        f"but got {actual_mode}."
    )
    hard_constraints = (
        hard_result.get("constraints", {}) if isinstance(hard_result, dict) else {}
    )
    gated_hard_result = {
        **(hard_result if isinstance(hard_result, dict) else {}),
        "score": 0.0,
        "strict_hard_constraint": 0.0,
        "hard_constraint_score": 0.0,
        "strict_soft_preference": None,
        "soft_preference_score": None,
        "constraints": {
            **hard_constraints,
            "response_expectation": {
                "passed": False,
                "message": message,
                "details": {
                    "expected_mode": "plan",
                    "actual_status": actual_mode,
                    "has_daily_plans": False,
                },
            },
        },
    }
    return {
        "sample_id": sample_id,
        "evaluation_mode": "sat_query_missing_itinerary",
        "scores": {
            "feasibility_score": 0.0,
            "strict_feasibility": 0.0,
            "strict_hard_constraint": 0.0,
            "hard_constraint_score": 0.0,
            "strict_soft_preference": None,
            "soft_preference_score": None,
            "requirement_score": 0.0,
            "llm_user_simulation_score": None,
        },
        "feasibility_details": {
            "score": 0.0,
            "strict_feasibility": 0.0,
            "passed_checks": 0,
            "total_checks": 1,
            "dimensions": {},
            "subdimensions": {},
            "checks": {
                "response_expectation": {
                    "passed": False,
                    "message": message,
                }
            },
        },
        "requirement_details": gated_hard_result,
        "hard_constraint_details": hard_result,
        "diagnostics": {
            "delivered_itinerary": False,
        },
    }
