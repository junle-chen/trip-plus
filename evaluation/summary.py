"""Summary aggregation for single-turn evaluation runs."""

from __future__ import annotations

from typing import Any, Dict, List

from .feasibility.config import FEASIBILITY_DIMENSIONS
from .runtime import aggregate_runtime_totals


def build_evaluation_summary(
    total_test_samples: int,
    plan_files_found: int,
    results: List[Dict[str, Any]],
    success_count: int,
    failed_count: int,
    elapsed_time: float,
    workers: int,
) -> Dict[str, Any]:
    """Aggregate per-sample score payloads into the run-level summary."""
    valid_results = [
        r for r in results if r.get("success") and r.get("evaluation_result")
    ]
    dimension_stats = {
        dim_name: {
            "weight": dim_cfg["weight"],
            "sum_score": 0.0,
            "count": 0,
        }
        for dim_name, dim_cfg in FEASIBILITY_DIMENSIONS.items()
    }
    metrics_acc = {
        "feasibility_score": 0.0,
        "strict_feasibility": 0.0,
        "requirement_score": 0.0,
    }
    requirement_breakdown_acc = {
        "strict_hard_constraint": 0.0,
        "hard_constraint_score": 0.0,
        "strict_soft_preference": 0.0,
        "soft_preference_score": 0.0,
    }
    requirement_breakdown_counts = {key: 0 for key in requirement_breakdown_acc}
    llm_user_sim_count = 0
    llm_user_sim_score_sum = 0.0
    llm_user_sim_fail_count = 0
    delivered_itinerary_count = 0
    error_stats: Dict[str, Dict[str, Any]] = {}

    for result in valid_results:
        sample_id = result.get("sample_id", "unknown")
        evaluation = result["evaluation_result"]
        scores = evaluation["scores"]
        for metric_name in metrics_acc:
            value = scores.get(metric_name)
            if value is None:
                if (
                    metric_name == "strict_feasibility"
                    and scores.get("feasibility_score") is not None
                ):
                    feasibility_details = evaluation.get("feasibility_details") or {}
                    checks = feasibility_details.get("checks") or {}
                    check_values = [
                        bool(check.get("passed"))
                        for check in checks.values()
                        if isinstance(check, dict) and check.get("passed") is not None
                    ]
                    if check_values:
                        value = 1.0 if all(check_values) else 0.0
                    else:
                        value = (
                            1.0
                            if float(scores.get("feasibility_score") or 0.0)
                            >= 1.0 - 1e-9
                            else 0.0
                        )
                else:
                    continue
            try:
                metrics_acc[metric_name] += float(value)
            except (TypeError, ValueError):
                continue

        requirement_details = evaluation.get("requirement_details") or {}
        for breakdown_name in requirement_breakdown_acc:
            if breakdown_name == "strict_hard_constraint":
                value = requirement_details.get("strict_hard_constraint")
                if value is None:
                    value = requirement_details.get("hard_constraint_score")
                if value is None:
                    value = scores.get("strict_hard_constraint")
            elif breakdown_name == "hard_constraint_score":
                value = requirement_details.get("hard_constraint_score")
                if (
                    requirement_details.get("strict_hard_constraint") is None
                    and requirement_details.get("hard_constraint_ratio") is not None
                ):
                    value = requirement_details.get("hard_constraint_ratio")
                if value is None:
                    value = scores.get("hard_constraint_score")
            elif breakdown_name == "strict_soft_preference":
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
                    value = (
                        1.0
                        if float(
                            requirement_details.get("soft_preference_score") or 0.0
                        )
                        >= 1.0 - 1e-9
                        else 0.0
                    )
                if value is None:
                    value = scores.get("strict_soft_preference")
            else:
                value = requirement_details.get(breakdown_name)
                if value is None:
                    value = scores.get(breakdown_name)
            if value is None:
                continue
            try:
                requirement_breakdown_acc[breakdown_name] += float(value)
                requirement_breakdown_counts[breakdown_name] += 1
            except (TypeError, ValueError):
                continue

        diagnostics = evaluation.get("diagnostics") or {}
        if diagnostics.get("delivered_itinerary"):
            delivered_itinerary_count += 1
        llm_user_sim = evaluation.get("llm_user_simulation")
        if isinstance(llm_user_sim, dict):
            if llm_user_sim.get("status") == "ok":
                llm_user_sim_count += 1
                try:
                    llm_user_sim_score_sum += float(llm_user_sim.get("score") or 0.0)
                except (TypeError, ValueError):
                    pass
            elif llm_user_sim.get("status") == "failed":
                llm_user_sim_fail_count += 1

        for dim_name, dim_detail in (
            evaluation.get("feasibility_details", {}).get("dimensions", {}).items()
        ):
            if dim_name in dimension_stats:
                dimension_stats[dim_name]["sum_score"] += dim_detail.get("score", 0.0)
                dimension_stats[dim_name]["count"] += 1

        for check_name, check_detail in (
            evaluation.get("feasibility_details", {}).get("checks", {}).items()
        ):
            if not check_detail.get("passed") and check_detail.get("message"):
                error_type = f"[Feasibility] {check_name}"
                bucket = error_stats.setdefault(
                    error_type, {"count": 0, "samples": [], "messages": []}
                )
                bucket["count"] += 1
                bucket["samples"].append(sample_id)
                bucket["messages"].append(check_detail["message"])

        for constraint_name, constraint_detail in (
            evaluation.get("requirement_details", {}).get("constraints", {}).items()
        ):
            if not constraint_detail.get("passed") and constraint_detail.get("message"):
                error_type = f"[Hard] {constraint_name}"
                bucket = error_stats.setdefault(
                    error_type, {"count": 0, "samples": [], "messages": []}
                )
                bucket["count"] += 1
                bucket["samples"].append(sample_id)
                bucket["messages"].append(constraint_detail["message"])

    delivery_rate = (
        delivered_itinerary_count / total_test_samples
        if total_test_samples > 0
        else 0.0
    )
    llm_user_simulation_score = (
        llm_user_sim_score_sum / llm_user_sim_count if llm_user_sim_count else None
    )
    summary_metrics = {
        "feasibility_score": metrics_acc["feasibility_score"] / total_test_samples
        if total_test_samples
        else 0.0,
        "strict_feasibility": metrics_acc["strict_feasibility"] / total_test_samples
        if total_test_samples
        else 0.0,
        "requirement_score": metrics_acc["requirement_score"] / total_test_samples
        if total_test_samples
        else 0.0,
        "llm_user_simulation_score": llm_user_simulation_score,
        "feasibility_dimensions": {
            dim_name: {
                "weight": stats["weight"],
                "score": (stats["sum_score"] / stats["count"])
                if stats["count"]
                else 0.0,
            }
            for dim_name, stats in dimension_stats.items()
        },
    }
    requirement_breakdowns = {
        name: (
            requirement_breakdown_acc[name] / requirement_breakdown_counts[name]
            if requirement_breakdown_counts[name]
            else None
        )
        for name in requirement_breakdown_acc
    }
    summary_metrics.update(
        {
            "strict_hard_constraint": requirement_breakdowns.get(
                "strict_hard_constraint"
            ),
            "hard_constraint_score": requirement_breakdowns.get(
                "hard_constraint_score"
            ),
            "strict_soft_preference": requirement_breakdowns.get(
                "strict_soft_preference"
            ),
            "soft_preference_score": requirement_breakdowns.get(
                "soft_preference_score"
            ),
        }
    )
    diagnostics = {
        "delivery_rate": delivery_rate,
        "converted_file_count": plan_files_found,
        "delivered_itinerary_count": delivered_itinerary_count,
        "missing_plan_count": max(0, total_test_samples - delivered_itinerary_count),
        "runtime_totals": aggregate_runtime_totals(
            [
                result["evaluation_result"].get("runtime") or {}
                for result in valid_results
            ]
        ),
        "llm_user_simulation_runtime_totals": aggregate_runtime_totals(
            [
                (result["evaluation_result"].get("llm_user_simulation") or {}).get(
                    "runtime"
                )
                or {}
                for result in valid_results
                if isinstance(
                    result["evaluation_result"].get("llm_user_simulation"), dict
                )
            ]
        ),
        "llm_user_simulation_success_count": llm_user_sim_count,
        "llm_user_simulation_failed_count": llm_user_sim_fail_count,
    }

    sorted_errors = sorted(
        error_stats.items(), key=lambda item: item[1]["count"], reverse=True
    )
    error_stats_serializable = [
        {
            "rank": index + 1,
            "error_type": error_type,
            "count": info["count"],
            "affected_samples": info["samples"],
            "sample_messages": info["messages"],
        }
        for index, (error_type, info) in enumerate(sorted_errors)
    ]

    return {
        "total_test_samples": total_test_samples,
        "plan_files_found": plan_files_found,
        "evaluation_success_count": success_count,
        "evaluation_failed_count": failed_count,
        "elapsed_time": elapsed_time,
        "max_workers": workers,
        "metrics": summary_metrics,
        "breakdowns": {
            "requirement_details": requirement_breakdowns,
        },
        "diagnostics": diagnostics,
        "error_statistics": error_stats_serializable,
        "results": [r["evaluation_result"] for r in valid_results],
    }
