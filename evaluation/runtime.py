"""Runtime and source-file metadata helpers for evaluation outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_runtime_stats_for_plan_file(plan_file: Path) -> Dict[str, Any]:
    trajectory_stem = plan_file.stem.replace("_converted", "")
    trajectory_stems = [trajectory_stem]
    if trajectory_stem.startswith("id_"):
        trajectory_stems.append(trajectory_stem[3:])
    else:
        trajectory_stems.append(f"id_{trajectory_stem}")
    trajectory_path = next(
        (
            plan_file.parent.parent / "trajectories" / f"{stem}.json"
            for stem in trajectory_stems
            if (plan_file.parent.parent / "trajectories" / f"{stem}.json").exists()
        ),
        plan_file.parent.parent / "trajectories" / f"{trajectory_stem}.json",
    )
    if not trajectory_path.exists():
        return {}
    try:
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    runtime = trajectory.get("runtime_stats") or {}
    if not isinstance(runtime, dict):
        runtime = {}
    if not runtime:
        messages = trajectory.get("messages") or []
        tool_errors = 0
        tool_calls = 0
        duplicate_tool_calls = 0
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            tool_calls += 1
            try:
                payload = json.loads(message.get("content") or "")
            except Exception:
                continue
            if isinstance(payload, dict) and payload.get("duplicate") is True:
                duplicate_tool_calls += 1
            if isinstance(payload, dict) and payload.get("error"):
                tool_errors += 1
        runtime = {
            "llm_calls": sum(1 for message in messages if isinstance(message, dict) and message.get("role") == "assistant"),
            "tool_calls": tool_calls,
            "tool_executions": max(0, tool_calls - duplicate_tool_calls),
            "duplicate_tool_calls": duplicate_tool_calls,
            "tool_errors": tool_errors,
            "token_usage": {},
            "token_usage_status": "unavailable_missing_runtime_stats",
            "status": "derived_from_trajectory",
        }
    token_usage = runtime.get("token_usage") if isinstance(runtime.get("token_usage"), dict) else {}
    token_usage_status = runtime.get("token_usage_status")
    if not token_usage_status:
        token_usage_status = "available" if token_usage else "unavailable_from_provider_or_trajectory"
    return {
        "trajectory_file": str(trajectory_path),
        "elapsed_time": trajectory.get("elapsed_time"),
        "llm_calls": runtime.get("llm_calls"),
        "tool_calls": runtime.get("tool_calls"),
        "tool_executions": runtime.get("tool_executions"),
        "duplicate_tool_calls": runtime.get("duplicate_tool_calls"),
        "tool_errors": runtime.get("tool_errors"),
        "token_usage": token_usage,
        "token_usage_status": token_usage_status,
        "status": runtime.get("status"),
    }


def aggregate_runtime_totals(runtime_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    token_totals: Dict[str, int] = {}
    token_available = 0
    token_missing = 0
    llm_calls = 0
    tool_calls = 0
    tool_executions = 0
    duplicate_tool_calls = 0
    tool_errors = 0
    for runtime in runtime_records:
        if not isinstance(runtime, dict):
            continue
        runtime_tool_calls = int(runtime.get("tool_calls") or 0)
        runtime_duplicate_tool_calls = int(runtime.get("duplicate_tool_calls") or 0)
        runtime_tool_executions = (
            int(runtime.get("tool_executions") or 0)
            if runtime.get("tool_executions") is not None
            else max(0, runtime_tool_calls - runtime_duplicate_tool_calls)
        )
        llm_calls += int(runtime.get("llm_calls") or 0)
        tool_calls += runtime_tool_calls
        tool_executions += runtime_tool_executions
        duplicate_tool_calls += runtime_duplicate_tool_calls
        tool_errors += int(runtime.get("tool_errors") or 0)
        usage = runtime.get("token_usage")
        if isinstance(usage, dict) and any(isinstance(value, (int, float)) for value in usage.values()):
            token_available += 1
            for key, value in usage.items():
                if isinstance(value, (int, float)):
                    token_totals[key] = token_totals.get(key, 0) + int(value)
        else:
            token_missing += 1
    if token_available == 0:
        token_status = "unavailable"
    elif token_missing:
        token_status = "partial"
    else:
        token_status = "available"
    return {
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "tool_executions": tool_executions,
        "duplicate_tool_calls": duplicate_tool_calls,
        "tool_errors": tool_errors,
        "token_usage": token_totals if token_available else None,
        "token_usage_status": token_status,
        "token_usage_available_count": token_available,
        "token_usage_missing_count": token_missing,
    }


def attach_evaluation_metadata(
    evaluation_result: Dict[str, Any],
    *,
    sample_record: Optional[Dict[str, Any]],
    query: str,
    plan_file: Path,
    runtime_stats: Dict[str, Any],
) -> None:
    artifact_stem = plan_file.stem.replace("_converted", "")
    report_stems = [artifact_stem]
    if artifact_stem.startswith("id_"):
        report_stems.append(artifact_stem[3:])
    else:
        report_stems.append(f"id_{artifact_stem}")
    report_path = next(
        (
            plan_file.parent.parent / "reports" / f"{stem}.txt"
            for stem in report_stems
            if (plan_file.parent.parent / "reports" / f"{stem}.txt").exists()
        ),
        plan_file.parent.parent / "reports" / f"{artifact_stem}.txt",
    )
    evaluation_result["query"] = query
    evaluation_result["source_files"] = {
        "converted_plan": str(plan_file),
        **({"report": str(report_path)} if report_path.exists() else {}),
        **({"trajectory": runtime_stats.get("trajectory_file")} if runtime_stats.get("trajectory_file") else {}),
    }
    evaluation_result["runtime"] = {
        key: value
        for key, value in runtime_stats.items()
        if key != "trajectory_file" and value is not None
    }
    if sample_record:
        evaluation_result["query_id"] = sample_record.get("id")
