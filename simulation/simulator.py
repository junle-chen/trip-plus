"""Run LLM-based traveler experience simulation for one itinerary."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional
from pathlib import Path

from agent.call_llm import call_llm

from .chunking import (
    filter_experience_trace,
    filter_plan,
    merge_chunk_payloads,
    split_activity_refs,
)
from .experience_trace import build_experience_trace, plan_activity_refs
from .prompting import build_user_simulator_messages
from .scoring import (
    assert_simulation_checks_pass,
    computed_overall_scores,
    normalize_user_simulation_output,
)


def _extract_json_object(text: str) -> Dict[str, Any]:
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("LLM user simulator returned empty content")

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S)
    if fence:
        stripped = fence.group(1).strip()
    else:
        tag = re.search(r"<JSON>\s*(\{.*?\})\s*</JSON>", stripped, flags=re.S | re.I)
        if tag:
            stripped = tag.group(1).strip()

    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in LLM user simulator output")
        stripped = stripped[start : end + 1]

    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("LLM user simulator output must be a JSON object")
    return parsed


def _simulation_request_overrides(request_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    overrides = {
        "temperature": float(os.environ.get("USER_SIMULATION_TEMPERATURE", "0.0")),
        "top_p": float(os.environ.get("USER_SIMULATION_TOP_P", "1.0")),
        "max_tokens": int(os.environ.get("USER_SIMULATION_MAX_TOKENS", "8192")),
    }
    if request_overrides:
        overrides.update(request_overrides)
    return overrides


def _response_usage_to_dict(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, dict):
        data = dict(usage)
    elif hasattr(usage, "model_dump"):
        data = usage.model_dump()
    elif hasattr(usage, "dict"):
        data = usage.dict()
    else:
        data = {
            key: getattr(usage, key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if getattr(usage, key, None) is not None
        }
    return data if isinstance(data, dict) else {}


def _merge_token_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    for key, value in usage.items():
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value


def _llm_runtime_payload(
    model: str,
    call_usages: list[Dict[str, Any]],
    *,
    call_count: Optional[int] = None,
) -> Dict[str, Any]:
    token_usage: Dict[str, Any] = {}
    for usage in call_usages:
        _merge_token_usage(token_usage, usage)
    effective_call_count = int(call_count if call_count is not None else len(call_usages))
    return {
        "llm_calls": effective_call_count,
        "tool_calls": 0,
        "tool_errors": 0,
        "token_usage": token_usage,
        "token_usage_status": "available" if token_usage else "unavailable_from_provider",
        "llm_call_usage": [
            {"call_index": index, **usage}
            for index, usage in enumerate(call_usages, start=1)
        ],
        "model": model,
        "status": "completed",
    }


def run_simulation(
    *,
    model: str,
    query_record: Dict[str, Any],
    plan: Dict[str, Any],
    turn_id: Optional[int] = None,
    evaluation_context: Optional[Dict[str, Any]] = None,
    request_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    experience_trace = build_experience_trace(
        query_record=query_record,
        plan=plan,
        turn_id=turn_id,
        evaluation_context=evaluation_context,
    )
    expected_activity_refs = experience_trace.get("expected_activity_refs") or plan_activity_refs(plan)
    messages = build_user_simulator_messages(
        query_record=query_record,
        plan=plan,
        turn_id=turn_id,
        evaluation_context=evaluation_context,
        experience_trace=experience_trace,
    )
    response = call_llm(model, messages, request_overrides=_simulation_request_overrides(request_overrides))
    usage = _response_usage_to_dict(response)
    payload = normalize_user_simulation_output(
        _extract_json_object(response.choices[0].message.content or ""),
        input_activity_count=len(expected_activity_refs),
        expected_activity_refs=expected_activity_refs,
        experience_trace=experience_trace,
    )
    assert_simulation_checks_pass(payload)
    payload.setdefault("input_experience_trace", experience_trace)
    payload.setdefault("simulator_model", model)
    payload.setdefault("turn_id", turn_id)
    payload["runtime"] = _llm_runtime_payload(model, [usage] if usage else [], call_count=1)
    return payload


def run_chunked_simulation(
    *,
    model: str,
    query_record: Dict[str, Any],
    plan: Dict[str, Any],
    turn_id: Optional[int] = None,
    evaluation_context: Optional[Dict[str, Any]] = None,
    request_overrides: Optional[Dict[str, Any]] = None,
    chunk_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the simulator in bounded activity chunks, then validate globally."""

    experience_trace = build_experience_trace(
        query_record=query_record,
        plan=plan,
        turn_id=turn_id,
        evaluation_context=evaluation_context,
    )
    expected_activity_refs = experience_trace.get("expected_activity_refs") or plan_activity_refs(plan)
    effective_chunk_size = int(chunk_size or os.environ.get("USER_SIMULATION_CHUNK_SIZE", "8"))
    if len(expected_activity_refs) <= effective_chunk_size:
        return run_simulation(
            model=model,
            query_record=query_record,
            plan=plan,
            turn_id=turn_id,
            evaluation_context=evaluation_context,
            request_overrides=request_overrides,
        )

    chunks = split_activity_refs(expected_activity_refs, effective_chunk_size)
    context = dict(evaluation_context or {})
    context.setdefault("chunked_simulation", {})
    context["chunked_simulation"].update(
        {
            "enabled": True,
            "chunk_size": effective_chunk_size,
            "full_activity_count": len(expected_activity_refs),
        }
    )
    overrides = _simulation_request_overrides(request_overrides)

    chunk_payloads = []
    activity_simulations = []
    call_usages: list[Dict[str, Any]] = []
    for chunk_index, chunk_refs in enumerate(chunks, start=1):
        chunk_trace = filter_experience_trace(experience_trace, chunk_refs)
        chunk_plan = filter_plan(plan, chunk_refs)
        chunk_context = dict(context)
        chunk_context["chunked_simulation"] = dict(context["chunked_simulation"])
        chunk_context["chunked_simulation"].update(
            {
                "chunk_index": chunk_index,
                "chunk_count": len(chunks),
                "chunk_activity_refs": chunk_refs,
            }
        )
        messages = build_user_simulator_messages(
            query_record=query_record,
            plan=chunk_plan,
            turn_id=turn_id,
            evaluation_context=chunk_context,
            experience_trace=chunk_trace,
        )
        response = call_llm(model, messages, request_overrides=overrides)
        usage = _response_usage_to_dict(response)
        if usage:
            call_usages.append(usage)
        chunk_payload = normalize_user_simulation_output(
            _extract_json_object(response.choices[0].message.content or ""),
            input_activity_count=len(chunk_refs),
            expected_activity_refs=chunk_refs,
            experience_trace=chunk_trace,
        )
        assert_simulation_checks_pass(chunk_payload)
        chunk_payload["chunk_index"] = chunk_index
        chunk_payload["chunk_activity_refs"] = chunk_refs
        chunk_payloads.append(chunk_payload)
        activity_simulations.extend(chunk_payload.get("activity_simulations") or [])

    merged = merge_chunk_payloads(
        chunk_payloads=chunk_payloads,
        activity_simulations=activity_simulations,
        expected_activity_refs=expected_activity_refs,
        experience_trace=experience_trace,
        model=model,
        turn_id=turn_id,
        chunk_size=effective_chunk_size,
    )
    merged["runtime"] = _llm_runtime_payload(model, call_usages, call_count=len(chunk_payloads))
    assert_simulation_checks_pass(merged)
    return merged


def write_simulation_artifact(
    *,
    artifact_dir: Path,
    sample_id: str,
    simulation: Dict[str, Any],
    turn_id: Optional[int] = None,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_turn_{turn_id}" if turn_id is not None else ""
    sample_label = str(sample_id)
    if not sample_label.startswith("id_"):
        sample_label = f"id_{sample_label}"
    path = artifact_dir / f"{sample_label}{suffix}_user_simulation.json"
    path.write_text(json.dumps(simulation, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def summarize_simulation(simulation: Dict[str, Any], artifact_path: Optional[Path] = None) -> Dict[str, Any]:
    return {
        "status": "ok",
        "artifact_path": str(artifact_path) if artifact_path else None,
        "score": computed_overall_scores(simulation).get("recomputed_score"),
        "score_1_5": computed_overall_scores(simulation).get("recomputed_score_1_5"),
        "llm_reported_overall": simulation.get("llm_reported_overall"),
        "score_recalculation": simulation.get("score_recalculation"),
        "activity_count_check": simulation.get("activity_count_check") or {},
        "trace_faithfulness_check": simulation.get("trace_faithfulness_check") or {},
        "simulator_model": simulation.get("simulator_model"),
        "runtime": simulation.get("runtime"),
    }


def build_failed_summary(error: Exception) -> Dict[str, Any]:
    return {
        "status": "failed",
        "error": f"{type(error).__name__}: {error}",
    }
