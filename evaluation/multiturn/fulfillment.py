"""Fulfillment and preserve checks for multi-turn turns."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils import normalize_entity_name
from .constants import EVALUATED_MUST_UPDATE_KEYS, NON_ITINERARY_MUST_UPDATE_KEYS
from .preserve import build_preserve_checks
from .response_mode import (
    _response_expectation_result,
    _status_is_clarification,
    _status_is_unsat,
)
from .update_rules import (
    _dietary_update_result,
    _duration_update_result,
    _environment_update_result,
    _extract_plan_attractions,
    _family_rule_ids,
    _generated_constraint_results,
    _late_start_update_result,
    _non_itinerary_resolution_result,
    _party_update_result,
    _profile_rule_ids_for_turn,
    _resolved_pacing_limit_result,
    _restaurant_slot_result,
    _schedule_update_result,
    _soft_check_result,
)


def _ratio_from_bools(values: List[bool]) -> Optional[float]:
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def evaluate_turn_alignment(
    plan: Dict[str, Any],
    meta: Dict[str, Any],
    hard_result: Dict[str, Any],
    feasibility_results: Dict[str, Tuple[bool, Optional[str]]] | None = None,
    database_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate turn-level goal and context-retention signals."""
    ground_truth = meta.get("turn_ground_truth") or {}
    must_preserve = [str(item) for item in ground_truth.get("must_preserve") or []]
    explicit_must_update = [
        str(item) for item in ground_truth.get("must_update") or []
    ]
    inferred_must_update = [
        str(item) for item in ground_truth.get("inferred_must_update") or []
    ]
    must_update = list(dict.fromkeys([*explicit_must_update, *inferred_must_update]))
    generated_constraints = ground_truth.get("generated_hard_constraints") or {}
    current_generated_constraints = (
        ground_truth.get("current_generated_hard_constraints") or {}
    )

    hard_constraints = (
        hard_result.get("constraints", {}) if isinstance(hard_result, dict) else {}
    )
    expected_status = str(
        ground_truth.get("expected_solution_status")
        or meta.get("solution_status")
        or "sat"
    ).lower()
    response_expectation = str(
        ground_truth.get("response_expectation")
        or meta.get("response_expectation")
        or ("clarification" if expected_status == "unsat" else "plan")
    ).lower()
    expects_non_itinerary = response_expectation in {
        "clarification",
        "conflict_resolution",
        "infeasible",
        "no_solution",
    }
    response_check = _response_expectation_result(plan, response_expectation)
    retention_checks = build_preserve_checks(
        must_preserve,
        expected_status=expected_status,
        expects_non_itinerary=expects_non_itinerary,
        hard_constraints=hard_constraints,
    )

    update_checks: List[Dict[str, Any]] = []
    plan_attractions = _extract_plan_attractions(plan)

    if "initial_plan" in must_update:
        update_checks.append(
            {
                "name": "initial_plan",
                "passed": bool(plan.get("daily_plans"))
                if expected_status != "unsat"
                else _status_is_unsat(plan),
                "message": None,
            }
        )

    update_checks.append(response_check)

    for update_name in sorted(NON_ITINERARY_MUST_UPDATE_KEYS):
        if update_name in must_update:
            update_checks.append(_non_itinerary_resolution_result(update_name, plan))

    active_added_constraints = {
        key: value
        for key, value in generated_constraints.items()
        if str(key).startswith("attraction_")
    }
    current_added_constraints = {
        key: value
        for key, value in current_generated_constraints.items()
        if str(key).startswith("attraction_")
    }
    added_constraints = current_added_constraints or (
        active_added_constraints if "add_attraction" in explicit_must_update else {}
    )
    if "add_attraction" in must_update:
        for key, constraint in added_constraints.items():
            names = [
                str(name).strip()
                for name in constraint.get("attraction_names", [])
                if str(name).strip()
            ]
            passed = all(
                normalize_entity_name(name) in plan_attractions for name in names
            )
            hard_detail = hard_constraints.get(key)
            if hard_detail is not None:
                passed = passed and bool(hard_detail.get("passed"))
            update_checks.append(
                {
                    "name": key,
                    "passed": passed,
                    "message": None
                    if passed
                    else f"Missing added attraction(s): {names}",
                }
            )

    if "party_update" in must_update:
        update_checks.append(
            _party_update_result(plan, ground_truth, feasibility_results)
        )

    if "restaurant_requirement" in must_update:
        results = _generated_constraint_results(
            current_generated_constraints or generated_constraints,
            hard_constraints,
            "restaurant_",
            "Added restaurant requirement not satisfied",
        )
        update_checks.extend(
            results
            or [
                {
                    "name": "restaurant_requirement",
                    "passed": False,
                    "message": "No generated restaurant hard constraint found for restaurant_requirement",
                }
            ]
        )
        for key, constraint in (
            current_generated_constraints or generated_constraints
        ).items():
            if not str(key).startswith("restaurant_") or not isinstance(
                constraint, dict
            ):
                continue
            slot_result = _restaurant_slot_result(plan, key, constraint)
            if slot_result is not None:
                update_checks.append(slot_result)

    if "hotel_requirement" in must_update:
        results = _generated_constraint_results(
            current_generated_constraints or generated_constraints,
            hard_constraints,
            "hotel_",
            "Added hotel requirement not satisfied",
        )
        update_checks.extend(
            results
            or [
                {
                    "name": "hotel_requirement",
                    "passed": False,
                    "message": "No generated hotel hard constraint found for hotel_requirement",
                }
            ]
        )

    if "budget_update" in must_update:
        detail = hard_constraints.get("budget_constraint")
        passed = bool(detail and detail.get("passed"))
        update_checks.append(
            {
                "name": "budget_update",
                "passed": passed,
                "message": None
                if passed
                else (detail or {}).get(
                    "message", "Updated budget constraint not satisfied"
                ),
            }
        )

    if "late_start_request" in must_update:
        update_checks.append(_late_start_update_result(plan, ground_truth))

    if "schedule_update" in must_update:
        update_checks.append(_schedule_update_result(plan, ground_truth))

    if "duration_update" in must_update:
        update_checks.append(_duration_update_result(plan, ground_truth))

    if "dietary_update" in must_update:
        update_checks.append(_dietary_update_result(plan, ground_truth))

    if (
        "explain_unsolved" in must_update
        or "infeasibility_detection" in must_update
        or expected_status == "unsat"
    ):
        expects_no_solution = expected_status == "unsat" or response_expectation in {
            "infeasible",
            "no_solution",
        }
        passed = (
            _status_is_unsat(plan)
            if expects_no_solution
            else _status_is_clarification(plan)
        )
        update_checks.append(
            {
                "name": "unsat_response"
                if expects_no_solution
                else "clarification_for_blocking_constraint",
                "passed": passed,
                "message": None
                if passed
                else (
                    "Expected no-solution response without daily_plans"
                    if expects_no_solution
                    else "Expected a clarification response without daily_plans"
                ),
            }
        )

    if "profile_preference" in must_update:
        update_checks.append(
            _soft_check_result(
                "profile_preference",
                plan,
                meta,
                _profile_rule_ids_for_turn(ground_truth),
                database_dir=database_dir,
            )
        )

    if "environment_aware_replanning" in must_update:
        update_checks.append(
            _environment_update_result(
                "environment_aware_replanning", plan, ground_truth
            )
        )

    if "resolved_priority" in must_update:
        update_checks.append(
            _soft_check_result(
                "resolved_priority",
                plan,
                meta,
                _family_rule_ids(meta, "route"),
                threshold=0.70,
                database_dir=database_dir,
            )
        )

    if "resolved_pacing_limit" in must_update:
        update_checks.append(_resolved_pacing_limit_result(plan, ground_truth))

    if "final_integrated_plan" in must_update:
        update_checks.append(
            _soft_check_result(
                "final_integrated_plan",
                plan,
                meta,
                None,
                threshold=0.70,
                database_dir=database_dir,
            )
        )

    if "route_preference" in must_update:
        update_checks.append(
            _soft_check_result(
                "route_preference",
                plan,
                meta,
                _family_rule_ids(meta, "route"),
                database_dir=database_dir,
            )
        )

    if "transport_preference" in must_update:
        update_checks.append(
            _soft_check_result(
                "transport_preference",
                plan,
                meta,
                _family_rule_ids(meta, "transport"),
                database_dir=database_dir,
            )
        )

    if "hotel_preference" in must_update:
        update_checks.append(
            _soft_check_result(
                "hotel_preference",
                plan,
                meta,
                _family_rule_ids(meta, "hotel"),
                database_dir=database_dir,
            )
        )

    if (
        "restaurant_preference" in must_update
        or "resolved_restaurant_preference" in must_update
    ):
        update_name = (
            "resolved_restaurant_preference"
            if "resolved_restaurant_preference" in must_update
            else "restaurant_preference"
        )
        update_checks.append(
            _soft_check_result(
                update_name,
                plan,
                meta,
                _family_rule_ids(meta, "restaurant"),
                threshold=0.60,
                database_dir=database_dir,
            )
        )

    if "apply_relaxed_constraint" in must_update:
        passed = bool(hard_result.get("score"))
        update_checks.append(
            {
                "name": "apply_relaxed_constraint",
                "passed": passed,
                "message": None
                if passed
                else "Relaxed turn still fails active hard constraints",
            }
        )

    # Keep explicitly listed soft updates in the report, but do not force a
    # binary score until a dedicated soft-preference judge/simulator is wired.
    evaluated_update_names = {check["name"] for check in update_checks}
    for update_name in must_update:
        if update_name in EVALUATED_MUST_UPDATE_KEYS:
            continue
        if update_name not in evaluated_update_names:
            update_checks.append(
                {
                    "name": update_name,
                    "passed": None,
                    "message": "Update recorded in ground truth; use response-level reflection scoring for this update",
                }
            )

    scored_update = [
        bool(check["passed"])
        for check in update_checks
        if check.get("passed") is not None
    ]
    retention_bools = [
        bool(check["passed"])
        for check in retention_checks
        if check.get("passed") is not None
    ]
    preserve_error_count = sum(
        1 for check in retention_checks if check.get("passed") is False
    )
    fulfillment_score = _ratio_from_bools(scored_update)
    preserve_score = _ratio_from_bools(retention_bools)
    preserve_check_count = len(retention_bools)
    preserve_error_rate = (
        preserve_error_count / preserve_check_count
        if preserve_check_count
        else None
    )
    family_breakdowns: Dict[str, Optional[float]] = {
        "user_state_update_success": None,
        "request_resolution_success": None,
        "environment_adaptation_success": None,
    }
    if any(
        name in must_update
        for name in {
            "profile_preference",
            "party_update",
            "budget_update",
            "late_start_request",
            "schedule_update",
            "duration_update",
            "dietary_update",
        }
    ):
        family_breakdowns["user_state_update_success"] = fulfillment_score
    if any(
        name in must_update
        for name in NON_ITINERARY_MUST_UPDATE_KEYS
        | {"resolved_priority", "resolved_pacing_limit", "apply_relaxed_constraint"}
    ):
        family_breakdowns["request_resolution_success"] = fulfillment_score
    if "environment_aware_replanning" in must_update:
        family_breakdowns["environment_adaptation_success"] = fulfillment_score

    return {
        "fulfillment_score": fulfillment_score,
        "preserve_score": preserve_score,
        "preserve_check_count": preserve_check_count,
        "preserve_error_count": preserve_error_count,
        "preserve_error_rate": preserve_error_rate,
        "fulfillment_checks": update_checks,
        "preserve_checks": retention_checks,
        "breakdowns": family_breakdowns,
    }
