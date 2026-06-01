"""Response-mode checks: plan, clarification, and no-solution."""

from __future__ import annotations

from typing import Any, Dict


def _status_is_unsat(plan: Dict[str, Any]) -> bool:
    return str(plan.get("status") or "").strip().lower() == "unsat" and not bool(
        plan.get("daily_plans")
    )


def _status_is_clarification(plan: Dict[str, Any]) -> bool:
    return str(
        plan.get("status") or ""
    ).strip().lower() == "clarification" and not bool(plan.get("daily_plans"))


def _response_mode(plan: Dict[str, Any]) -> str:
    status = str(plan.get("status") or "").strip().lower()
    has_daily_plans = bool(plan.get("daily_plans"))
    if status == "unsat" and not has_daily_plans:
        return "no_solution"
    if status == "clarification" and not has_daily_plans:
        return "clarification"
    if has_daily_plans and status not in {"unsat", "clarification"}:
        return "plan"
    return "invalid"


def _response_expectation_result(
    plan: Dict[str, Any], response_expectation: str
) -> Dict[str, Any]:
    expected = str(response_expectation or "plan").strip().lower()
    actual = _response_mode(plan)
    if expected == "conflict_resolution":
        passed = actual == "clarification"
        expected_mode = "clarification"
    elif expected in {"infeasible", "no_solution"}:
        passed = actual == "no_solution"
        expected_mode = "no_solution"
    elif expected == "clarification":
        passed = actual == "clarification"
        expected_mode = "clarification"
    else:
        passed = actual == "plan"
        expected_mode = "plan"
    return {
        "name": "response_expectation",
        "passed": passed,
        "message": None
        if passed
        else f"Expected {expected_mode} response for response_expectation={expected}, got {actual}.",
        "details": {
            "response_expectation": expected,
            "expected_mode": expected_mode,
            "actual_mode": actual,
            "actual_status": str(plan.get("status") or "").strip().lower(),
            "has_daily_plans": bool(plan.get("daily_plans")),
        },
    }


def build_response_mode_result(
    sample_id: str,
    plan: Dict[str, Any],
    response_expectation: str,
    *,
    evaluation_mode: str,
    feasibility_score: float | None,
    strict_feasibility: float | None,
    requirement_score: float,
) -> Dict[str, Any]:
    check = _response_expectation_result(plan, response_expectation)
    strict_requirement = 1.0 if requirement_score >= 1.0 - 1e-9 else 0.0
    return {
        "sample_id": sample_id,
        "evaluation_mode": evaluation_mode,
        "scores": {
            "feasibility_score": feasibility_score,
            "strict_feasibility": strict_feasibility,
            "strict_hard_constraint": strict_requirement,
            "hard_constraint_score": requirement_score,
            "strict_soft_preference": None,
            "soft_preference_score": None,
            "requirement_score": requirement_score,
            "llm_user_simulation_score": None,
        },
        "response_expectation_details": check,
    }
