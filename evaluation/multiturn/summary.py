"""Summary metrics for multi-turn evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..runtime import aggregate_runtime_totals

PLAN_SCOPED_SCORE_NAMES = {
    "feasibility_score",
    "strict_feasibility",
    "strict_hard_constraint",
    "hard_constraint_score",
    "strict_soft_preference",
    "soft_preference_score",
    "requirement_score",
}


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _response_expectation_check_from_turn(
    turn: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    evaluation_result = turn.get("evaluation_result") or {}
    turn_alignment = evaluation_result.get("turn_alignment") or {}
    checks = (
        turn_alignment.get("fulfillment_checks")
        or []
    )
    for check in checks:
        if isinstance(check, dict) and check.get("name") == "response_expectation":
            return check
    detail = evaluation_result.get("response_expectation_details")
    return detail if isinstance(detail, dict) else None


def _response_expectation_check_details(
    check: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(check, dict):
        return {}
    details = check.get("details")
    if isinstance(details, dict):
        return details
    return check


def _expected_mode_from_response_expectation(response_expectation: object) -> str:
    expected = str(response_expectation or "plan").strip().lower()
    if expected == "conflict_resolution":
        return "clarification"
    if expected in {"infeasible", "no_solution"}:
        return "no_solution"
    if expected == "clarification":
        return expected
    return "plan"


def _turn_response_modes(turn: Dict[str, Any]) -> Dict[str, str]:
    check = _response_expectation_check_from_turn(turn)
    details = _response_expectation_check_details(check)
    expected_mode = str(details.get("expected_mode") or "").strip().lower()
    if not expected_mode:
        expected_mode = _expected_mode_from_response_expectation(
            details.get("response_expectation") or details.get("expected")
        )
    actual_mode = str(details.get("actual_mode") or "").strip().lower()
    return {
        "expected_mode": expected_mode or "plan",
        "actual_mode": actual_mode or "invalid",
    }


def _soft_preference_failure_count(evaluation_result: Dict[str, Any]) -> int:
    requirement_details = evaluation_result.get("requirement_details") or {}
    soft_preferences = requirement_details.get("soft_preferences") or {}
    failures = 0
    for check in soft_preferences.get("checks") or []:
        if not isinstance(check, dict):
            continue
        if not check.get("applicable", True):
            continue
        try:
            score = float(check.get("score"))
        except (TypeError, ValueError):
            continue
        if score < 1.0 - 1e-9:
            failures += 1
    return failures


def _score_value(turn: Dict[str, Any], score_name: str) -> Optional[float]:
    evaluation_result = turn.get("evaluation_result") or {}
    scores = evaluation_result.get("scores") or {}
    value = scores.get(score_name)
    if value is None and score_name == "strict_feasibility":
        feasibility_details = evaluation_result.get("feasibility_details") or {}
        value = feasibility_details.get("strict_feasibility")
        if value is None:
            checks = feasibility_details.get("checks") or {}
            check_values = [
                bool(check.get("passed"))
                for check in checks.values()
                if isinstance(check, dict) and check.get("passed") is not None
            ]
            if check_values:
                value = 1.0 if all(check_values) else 0.0
        if value is None and scores.get("feasibility_score") is not None:
            try:
                value = 1.0 if float(scores["feasibility_score"]) >= 1.0 - 1e-9 else 0.0
            except (TypeError, ValueError):
                value = None
    if value is None and score_name in {
        "strict_hard_constraint",
        "hard_constraint_score",
        "strict_soft_preference",
        "soft_preference_score",
    }:
        requirement_details = evaluation_result.get("requirement_details") or {}
        if score_name == "strict_hard_constraint":
            value = requirement_details.get("strict_hard_constraint")
            if value is None:
                value = requirement_details.get("hard_constraint_score")
        elif score_name == "hard_constraint_score":
            value = requirement_details.get("hard_constraint_score")
            if (
                requirement_details.get("strict_hard_constraint") is None
                and requirement_details.get("hard_constraint_ratio") is not None
            ):
                value = requirement_details.get("hard_constraint_ratio")
        elif score_name == "strict_soft_preference":
            value = requirement_details.get("strict_soft_preference")
            if value is None:
                soft_preferences = requirement_details.get("soft_preferences") or {}
                check_values = [
                    float(check["score"]) >= 1.0 - 1e-9
                    for check in soft_preferences.get("checks") or []
                    if isinstance(check, dict)
                    and check.get("applicable", True)
                    and check.get("score") is not None
                ]
                if check_values:
                    value = 1.0 if all(check_values) else 0.0
            if (
                value is None
                and requirement_details.get("soft_preference_score") is not None
            ):
                try:
                    value = (
                        1.0
                        if float(requirement_details["soft_preference_score"])
                        >= 1.0 - 1e-9
                        else 0.0
                    )
                except (TypeError, ValueError):
                    value = None
        else:
            value = requirement_details.get(score_name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _successful_plan_turns(
    evaluated_turns: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    successful: List[Dict[str, Any]] = []
    for turn in evaluated_turns:
        modes = _turn_response_modes(turn)
        if modes["expected_mode"] == "plan" and modes["actual_mode"] == "plan":
            successful.append(turn)
    return successful


def _mean_score(turns: List[Dict[str, Any]], score_name: str) -> Optional[float]:
    values = [
        value
        for turn in turns
        for value in [_score_value(turn, score_name)]
        if value is not None
    ]
    return _mean(values)


def _zero_imputed_mean_score(
    turns: List[Dict[str, Any]], score_name: str
) -> Optional[float]:
    if not turns:
        return None
    return sum(float(_score_value(turn, score_name) or 0.0) for turn in turns) / len(
        turns
    )


def _score_count(turns: List[Dict[str, Any]], score_name: str) -> int:
    return sum(1 for turn in turns if _score_value(turn, score_name) is not None)


def _turn_result_sort_key(turn: Dict[str, Any]) -> tuple[int, str]:
    raw = turn.get("turn_id")
    try:
        return (0, f"{int(str(raw)):06d}")
    except (TypeError, ValueError):
        return (1, str(raw))


def _final_successful_turns(
    records: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    final_turns: List[Dict[str, Any]] = []
    for record in records or []:
        successful_turns = [
            turn
            for turn in record.get("turn_results", [])
            if turn.get("success") and turn.get("evaluation_result")
        ]
        if successful_turns:
            final_turns.append(max(successful_turns, key=_turn_result_sort_key))
    return final_turns


def _expected_plan_turns(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        turn for turn in turns if _turn_response_modes(turn)["expected_mode"] == "plan"
    ]


PLAN_SCOPED_SCORE_NAMES = {
    "feasibility_score",
    "strict_feasibility",
    "strict_hard_constraint",
    "hard_constraint_score",
    "strict_soft_preference",
    "soft_preference_score",
    "requirement_score",
}


def _build_metric_groups(
    evaluated_turns: List[Dict[str, Any]],
    metrics: Dict[str, Optional[float]],
    records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    successful_plans = _successful_plan_turns(evaluated_turns)
    expected_plan_turns = _expected_plan_turns(evaluated_turns)
    final_expected_plan_turns = _expected_plan_turns(_final_successful_turns(records))
    final_successful_plan_turns = _successful_plan_turns(final_expected_plan_turns)
    score_names = [
        "feasibility_score",
        "strict_feasibility",
        "strict_hard_constraint",
        "hard_constraint_score",
        "strict_soft_preference",
        "soft_preference_score",
        "requirement_score",
        "llm_user_simulation_score",
    ]
    all_response_scores = {name: metrics.get(name) for name in score_names}
    for name in PLAN_SCOPED_SCORE_NAMES:
        all_response_scores[name] = _zero_imputed_mean_score(evaluated_turns, name)
    successful_plan_scores = {
        name: _mean_score(successful_plans, name) for name in score_names
    }

    if final_expected_plan_turns:
        all_response_scores["llm_user_simulation_score"] = _mean_score(
            final_expected_plan_turns,
            "llm_user_simulation_score",
        )
    if final_successful_plan_turns:
        successful_plan_scores["llm_user_simulation_score"] = _mean_score(
            final_successful_plan_turns,
            "llm_user_simulation_score",
        )

    return {
        "all_responses": {
            **all_response_scores,
            "evaluated_turn_count": len(evaluated_turns),
            "score_counts": {
                name: _score_count(
                    evaluated_turns,
                    name,
                )
                for name in score_names
            },
            "score_denominators": {
                name: len(evaluated_turns)
                if name in PLAN_SCOPED_SCORE_NAMES
                else _score_count(evaluated_turns, name)
                for name in score_names
            },
            "plan_requirement_denominator": len(expected_plan_turns),
            "llm_user_simulation_denominator": len(final_expected_plan_turns),
            "definition": (
                "Coverage over all evaluated responses. Plan-scoped score averages "
                "impute missing, non-plan, or inapplicable turn scores as 0 over all "
                "evaluated turns, so this group reflects end-to-end response coverage. "
                "Score counts report observed non-null scores; score denominators report "
                "the averaging denominator."
            ),
        },
        "successful_plans": {
            **successful_plan_scores,
            "evaluated_turn_count": len(successful_plans),
            "score_counts": {
                name: _score_count(successful_plans, name) for name in score_names
            },
            "llm_user_simulation_denominator": len(final_successful_plan_turns),
            "definition": (
                "Conditional average over turns where the benchmark expected a plan "
                "and the model delivered a parseable plan. Missing or inapplicable "
                "scores are not imputed as 0."
            ),
        },
    }


def _hard_all_pass_value(evaluation_result: Dict[str, Any]) -> Optional[float]:
    requirement_details = evaluation_result.get("requirement_details") or {}
    strict = requirement_details.get("strict_hard_constraint")
    if strict is not None:
        try:
            return 1.0 if float(strict) >= 1.0 - 1e-9 else 0.0
        except (TypeError, ValueError):
            pass
    passed = requirement_details.get("hard_constraints_passed")
    total = requirement_details.get("hard_constraints_total")
    if passed is not None and total is not None:
        try:
            return 1.0 if int(passed) >= int(total) else 0.0
        except (TypeError, ValueError):
            pass

    ratio = requirement_details.get("hard_constraint_ratio")
    if ratio is not None:
        try:
            return 1.0 if float(ratio) >= 1.0 - 1e-9 else 0.0
        except (TypeError, ValueError):
            return None
    return None


def _strict_requirement_value(evaluation_result: Dict[str, Any]) -> Optional[float]:
    hard_all_pass = _hard_all_pass_value(evaluation_result)
    if hard_all_pass is None:
        return None

    requirement_details = evaluation_result.get("requirement_details") or {}
    strict_soft = requirement_details.get("strict_soft_preference")
    if strict_soft is None:
        soft_preferences = requirement_details.get("soft_preferences") or {}
        check_values = [
            float(check["score"]) >= 1.0 - 1e-9
            for check in soft_preferences.get("checks") or []
            if isinstance(check, dict)
            and check.get("applicable", True)
            and check.get("score") is not None
        ]
        if check_values:
            strict_soft = 1.0 if all(check_values) else 0.0
    if (
        strict_soft is None
        and requirement_details.get("soft_preference_score") is not None
    ):
        try:
            strict_soft = (
                1.0
                if float(requirement_details["soft_preference_score"]) >= 1.0 - 1e-9
                else 0.0
            )
        except (TypeError, ValueError):
            strict_soft = None
    if strict_soft is None:
        return hard_all_pass

    try:
        return (
            1.0
            if hard_all_pass >= 1.0 - 1e-9 and float(strict_soft) >= 1.0 - 1e-9
            else 0.0
        )
    except (TypeError, ValueError):
        return hard_all_pass


def _reporting_metric_bucket() -> Dict[str, Any]:
    return {
        "response_mode_success_values": [],
        "fulfillment_score_values": [],
        "preserve_score_values": [],
        "plan_feasibility_values": [],
        "cumulative_requirement_values": [],
        "end_to_end_requirement_strict_values": [],
        "hard_all_pass_filtered_values": [],
        "preserve_error_count": 0,
        "preserve_check_count": 0,
        "turn_count": 0,
        "expected_plan_turn_count": 0,
        "delivered_plan_turn_count": 0,
        "cumulative_requirement_evaluated_count": 0,
        "strict_requirement_evaluated_count": 0,
        "soft_preference_failure_count": 0,
        "expected_response_mode_counts": {},
        "actual_response_mode_counts": {},
    }


def _add_reporting_metrics(bucket: Dict[str, Any], turn: Dict[str, Any]) -> None:
    evaluation_result = turn.get("evaluation_result") or {}
    scores = evaluation_result.get("scores") or {}
    turn_alignment = evaluation_result.get("turn_alignment") or {}
    modes = _turn_response_modes(turn)
    expected_plan = modes["expected_mode"] == "plan"
    delivered_plan = modes["actual_mode"] == "plan"

    bucket["turn_count"] += 1
    bucket["expected_response_mode_counts"][modes["expected_mode"]] = (
        bucket["expected_response_mode_counts"].get(modes["expected_mode"], 0) + 1
    )
    bucket["actual_response_mode_counts"][modes["actual_mode"]] = (
        bucket["actual_response_mode_counts"].get(modes["actual_mode"], 0) + 1
    )
    if expected_plan:
        bucket["expected_plan_turn_count"] += 1
    if delivered_plan:
        bucket["delivered_plan_turn_count"] += 1

    response_check = _response_expectation_check_from_turn(turn)
    if isinstance(response_check, dict) and response_check.get("passed") is not None:
        bucket["response_mode_success_values"].append(
            1.0 if bool(response_check.get("passed")) else 0.0
        )

    if turn_alignment.get("fulfillment_score") is not None:
        bucket["fulfillment_score_values"].append(
            float(turn_alignment["fulfillment_score"])
        )

    if turn_alignment.get("preserve_score") is not None:
        bucket["preserve_score_values"].append(
            float(turn_alignment["preserve_score"])
        )

    if expected_plan and delivered_plan and scores.get("feasibility_score") is not None:
        bucket["plan_feasibility_values"].append(float(scores["feasibility_score"]))

    if expected_plan and delivered_plan and scores.get("requirement_score") is not None:
        bucket["cumulative_requirement_values"].append(
            float(scores["requirement_score"])
        )
        bucket["cumulative_requirement_evaluated_count"] += 1

    if expected_plan:
        strict_requirement = 0.0
        if delivered_plan:
            hard_all_pass = _hard_all_pass_value(evaluation_result)
            if hard_all_pass is not None:
                bucket["hard_all_pass_filtered_values"].append(hard_all_pass)
            strict_requirement = _strict_requirement_value(evaluation_result) or 0.0
        bucket["end_to_end_requirement_strict_values"].append(strict_requirement)
        bucket["strict_requirement_evaluated_count"] += 1

    bucket["preserve_error_count"] += int(turn_alignment.get("preserve_error_count") or 0)
    if turn_alignment.get("preserve_check_count") is not None:
        bucket["preserve_check_count"] += int(
            turn_alignment.get("preserve_check_count") or 0
        )
    else:
        bucket["preserve_check_count"] += sum(
            1
            for check in turn_alignment.get("preserve_checks") or []
            if isinstance(check, dict) and check.get("passed") is not None
        )
    bucket["soft_preference_failure_count"] += (
        _soft_preference_failure_count(evaluation_result) if delivered_plan else 0
    )


def _finalize_reporting_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    hard_all_pass_count = sum(bucket["hard_all_pass_filtered_values"])
    hard_all_pass_rate = (
        hard_all_pass_count / bucket["expected_plan_turn_count"]
        if bucket["expected_plan_turn_count"]
        else None
    )
    return {
        "turn_count": bucket["turn_count"],
        "expected_response_mode_counts": dict(
            sorted(bucket["expected_response_mode_counts"].items())
        ),
        "actual_response_mode_counts": dict(
            sorted(bucket["actual_response_mode_counts"].items())
        ),
        "expected_plan_turn_count": bucket["expected_plan_turn_count"],
        "delivered_plan_turn_count": bucket["delivered_plan_turn_count"],
        "response_mode_accuracy": _mean(bucket["response_mode_success_values"]),
        "fulfillment_score": _mean(bucket["fulfillment_score_values"]),
        "preserve_score": _mean(bucket["preserve_score_values"]),
        "plan_feasibility": _mean(bucket["plan_feasibility_values"]),
        "cumulative_requirement_satisfaction": _mean(
            bucket["cumulative_requirement_values"]
        ),
        "hard_all_pass_rate": hard_all_pass_rate,
        "hard_all_pass_rate_filtered": _mean(bucket["hard_all_pass_filtered_values"]),
        "end_to_end_requirement_strict": _mean(
            bucket["end_to_end_requirement_strict_values"]
        ),
        "hard_all_pass_count": hard_all_pass_count,
        "preserve_error_count": bucket["preserve_error_count"],
        "preserve_check_count": bucket["preserve_check_count"],
        "preserve_error_rate": (
            bucket["preserve_error_count"] / bucket["preserve_check_count"]
            if bucket["preserve_check_count"]
            else None
        ),
        "soft_preference_failure_count": bucket["soft_preference_failure_count"],
        "plan_feasibility_evaluated_count": len(bucket["plan_feasibility_values"]),
        "cumulative_requirement_evaluated_count": bucket[
            "cumulative_requirement_evaluated_count"
        ],
        "strict_requirement_evaluated_count": bucket[
            "strict_requirement_evaluated_count"
        ],
    }


def _build_reporting_metrics(evaluated_turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    overall = _reporting_metric_bucket()
    by_turn: Dict[str, Dict[str, Any]] = {}
    by_turn_and_type: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for turn in evaluated_turns:
        turn_key = f"T{turn.get('turn_id')}"
        type_key = str(turn.get("_interaction_type") or "unknown")

        _add_reporting_metrics(overall, turn)
        _add_reporting_metrics(
            by_turn.setdefault(turn_key, _reporting_metric_bucket()), turn
        )
        type_bucket = by_turn_and_type.setdefault(turn_key, {})
        _add_reporting_metrics(
            type_bucket.setdefault(type_key, _reporting_metric_bucket()), turn
        )

    def turn_sort_key(item: tuple[str, Any]) -> tuple[int, str]:
        key = item[0]
        try:
            return (0, f"{int(key.lstrip('T')):06d}")
        except ValueError:
            return (1, key)

    return {
        "overall": _finalize_reporting_bucket(overall),
        "by_turn": {
            key: _finalize_reporting_bucket(bucket)
            for key, bucket in sorted(by_turn.items(), key=turn_sort_key)
        },
        "by_turn_and_type": {
            turn_key: {
                type_key: _finalize_reporting_bucket(type_bucket)
                for type_key, type_bucket in sorted(type_buckets.items())
            }
            for turn_key, type_buckets in sorted(
                by_turn_and_type.items(), key=turn_sort_key
            )
        },
        "definitions": {
            "response_mode_accuracy": "Whether each turn uses the expected response mode: plan, clarification, or no-solution.",
            "fulfillment_score": "Whether the current turn's requested update or response requirement is handled correctly.",
            "preserve_score": "Whether historical hard constraints listed in must_preserve remain satisfied.",
            "plan_feasibility": "Feasibility averaged only over turns where a plan is expected and a plan is delivered.",
            "cumulative_requirement_satisfaction": "Active hard constraints plus active soft preferences, averaged only over delivered expected-plan turns.",
            "hard_all_pass_rate": "Share of expected-plan turns where a delivered plan satisfies every active hard constraint; missing or non-plan responses count as failures.",
            "hard_all_pass_rate_filtered": "Reference metric: share of delivered expected-plan turns where every active hard constraint passes.",
            "end_to_end_requirement_strict": "Strict requirement score over all expected-plan turns. Missing/non-plan responses score 0; delivered plans average hard_all_pass and active soft-preference score.",
            "preserve_error_count": "Failed preserved historical hard constraints. Active soft preferences are included in cumulative_requirement_satisfaction.",
        },
    }


def summarize_multiturn_results(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    evaluated_turns = [
        {**turn, "_interaction_type": record.get("interaction_type")}
        for record in records
        for turn in record.get("turn_results", [])
        if turn.get("success") and turn.get("evaluation_result")
    ]
    missing_turns = [
        {"sample_id": record.get("sample_id"), "turn_id": turn.get("turn_id")}
        for record in records
        for turn in record.get("turn_results", [])
        if not turn.get("success")
    ]
    score_names = [
        "feasibility_score",
        "strict_feasibility",
        "strict_hard_constraint",
        "hard_constraint_score",
        "strict_soft_preference",
        "soft_preference_score",
        "requirement_score",
        "llm_user_simulation_score",
    ]
    successful_plans = _successful_plan_turns(evaluated_turns)
    metrics = {}
    for score_name in score_names:
        score_turns = (
            successful_plans
            if score_name in PLAN_SCOPED_SCORE_NAMES
            else evaluated_turns
        )
        values = [
            value
            for turn in score_turns
            for value in [_score_value(turn, score_name)]
            if value is not None
        ]
        metrics[score_name] = _mean(values)
    for name in ("fulfillment_score", "preserve_score"):
        values = [
            float(turn["evaluation_result"]["turn_alignment"][name])
            for turn in evaluated_turns
            if turn["evaluation_result"]["turn_alignment"].get(name) is not None
        ]
        metrics[name] = _mean(values)
    preserve_check_count = 0
    preserve_error_count = 0
    for turn in evaluated_turns:
        turn_alignment = turn["evaluation_result"].get("turn_alignment") or {}
        preserve_error_count += int(turn_alignment.get("preserve_error_count") or 0)
        if turn_alignment.get("preserve_check_count") is not None:
            preserve_check_count += int(
                turn_alignment.get("preserve_check_count") or 0
            )
        else:
            preserve_check_count += sum(
                1
                for check in turn_alignment.get("preserve_checks") or []
                if isinstance(check, dict) and check.get("passed") is not None
            )
    metrics["preserve_error_rate"] = (
        preserve_error_count / preserve_check_count
        if preserve_check_count
        else None
    )
    response_pass_values: List[float] = []
    response_by_turn: Dict[str, List[float]] = {}
    response_by_type: Dict[str, List[float]] = {}
    response_confusion: Dict[str, int] = {}
    for turn in evaluated_turns:
        check = _response_expectation_check_from_turn(turn)
        if not isinstance(check, dict) or check.get("passed") is None:
            continue
        passed = 1.0 if bool(check.get("passed")) else 0.0
        response_pass_values.append(passed)
        turn_key = f"T{turn.get('turn_id')}"
        response_by_turn.setdefault(turn_key, []).append(passed)
        type_key = str(turn.get("_interaction_type") or "unknown")
        response_by_type.setdefault(type_key, []).append(passed)
        if not passed:
            details = _response_expectation_check_details(check)
            expected = (
                details.get("response_expectation")
                or details.get("expected_mode")
                or "unknown"
            )
            actual = details.get("actual_mode") or "unknown"
            confusion_key = f"expected_{expected}__got_{actual}"
            response_confusion[confusion_key] = (
                response_confusion.get(confusion_key, 0) + 1
            )
    metrics["response_mode_accuracy"] = _mean(response_pass_values)
    breakdowns: Dict[str, Optional[float]] = {}
    for name in (
        "user_state_update_success",
        "request_resolution_success",
        "environment_adaptation_success",
    ):
        values = [
            float(
                turn["evaluation_result"]["turn_alignment"]
                .get("breakdowns", {})
                .get(name)
            )
            for turn in evaluated_turns
            if turn["evaluation_result"]["turn_alignment"]
            .get("breakdowns", {})
            .get(name)
            is not None
        ]
        breakdowns[name] = _mean(values)
    llm_user_sim_scores = []
    llm_user_sim_success_count = 0
    llm_user_sim_failed_count = 0
    for turn in evaluated_turns:
        llm_user_sim = turn["evaluation_result"].get("llm_user_simulation")
        if not isinstance(llm_user_sim, dict):
            continue
        if llm_user_sim.get("status") == "ok":
            llm_user_sim_success_count += 1
            if llm_user_sim.get("score") not in (None, ""):
                try:
                    llm_user_sim_scores.append(float(llm_user_sim["score"]))
                except (TypeError, ValueError):
                    pass
        elif llm_user_sim.get("status") == "failed":
            llm_user_sim_failed_count += 1
    if metrics.get("llm_user_simulation_score") is None:
        metrics["llm_user_simulation_score"] = _mean(llm_user_sim_scores)
    diagnostics = {
        "llm_user_simulation_success_count": llm_user_sim_success_count,
        "llm_user_simulation_failed_count": llm_user_sim_failed_count,
        "llm_user_simulation_scope": "final_turn_only",
        "runtime_totals": aggregate_runtime_totals(
            [turn["evaluation_result"].get("runtime") or {} for turn in evaluated_turns]
        ),
        "llm_user_simulation_runtime_totals": aggregate_runtime_totals(
            [
                (turn["evaluation_result"].get("llm_user_simulation") or {}).get(
                    "runtime"
                )
                or {}
                for turn in evaluated_turns
                if isinstance(
                    turn["evaluation_result"].get("llm_user_simulation"), dict
                )
            ]
        ),
        "preserve_error_count": preserve_error_count,
        "preserve_check_count": preserve_check_count,
        "preserve_error_rate": metrics["preserve_error_rate"],
        "missing_turn_plans": len(missing_turns),
        "response_mode_accuracy_by_turn": {
            key: _mean(values) for key, values in sorted(response_by_turn.items())
        },
        "response_mode_accuracy_by_type": {
            key: _mean(values) for key, values in sorted(response_by_type.items())
        },
        "response_mode_confusion": dict(sorted(response_confusion.items())),
    }
    first_mismatch_distribution: Dict[str, int] = {}
    for record in records:
        first_key = "none"
        successful_turns = [
            turn
            for turn in record.get("turn_results", [])
            if turn.get("success") and turn.get("evaluation_result")
        ]

        def _turn_sort_key(item: Dict[str, Any]) -> tuple[int, str]:
            raw = item.get("turn_id")
            try:
                return (0, f"{int(str(raw)):06d}")
            except (TypeError, ValueError):
                return (1, str(raw))

        for turn in sorted(successful_turns, key=_turn_sort_key):
            check = _response_expectation_check_from_turn(turn)
            if isinstance(check, dict) and check.get("passed") is False:
                first_key = f"T{turn.get('turn_id')}"
                break
        if successful_turns:
            first_mismatch_distribution[first_key] = (
                first_mismatch_distribution.get(first_key, 0) + 1
            )
    diagnostics["first_response_mode_mismatch_turn_distribution"] = dict(
        sorted(first_mismatch_distribution.items())
    )
    metric_groups = _build_metric_groups(evaluated_turns, metrics, records)
    reporting_metrics = _build_reporting_metrics(evaluated_turns)
    return {
        "total_samples": len(records),
        "evaluated_turns": len(evaluated_turns),
        "missing_turn_plans": len(missing_turns),
        "metrics": metrics,
        "metric_groups": metric_groups,
        "reporting_metrics": reporting_metrics,
        "breakdowns": breakdowns,
        "diagnostics": diagnostics,
        "missing_turns": missing_turns,
        "results": records,
    }
