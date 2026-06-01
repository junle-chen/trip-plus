"""Evaluation artifact writers and compact summary helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEBUG_SCORE_DIR_NAME = "debug"
MULTITURN_MAIN_SUMMARY_FILE_NAME = "_summary.json"
MULTITURN_FULL_SUMMARY_FILE_NAME = "multiturn_summary_full.json"

DEBUG_ONLY_TOP_LEVEL_KEYS = {
    "ground_truth",
    "hard_constraint_details",
    "plan_file",
    "source_files",
}
DEBUG_ONLY_REQUIREMENT_KEYS = {"hard_constraints"}
DEBUG_ONLY_FEASIBILITY_KEYS = {"subdimensions"}
DEBUG_ONLY_SOFT_CHECK_KEYS = {
    "signals",
    "raw_signals",
    "thresholds",
    "threshold_source",
    "threshold_sources",
    "canonical_rule_id",
    "input_rule_ids",
    "source",
    "score_type",
    "evidence_basis",
}
DEBUG_ONLY_DIAGNOSTIC_NESTED_KEYS = {
    "signals",
    "raw_signals",
    "thresholds",
    "threshold_source",
    "threshold_sources",
    "references",
}
COMPACT_TOP_LEVEL_KEYS = (
    "sample_id",
    "parent_id",
    "turn_id",
    "evaluation_mode",
    "success",
    "error",
    "scores",
    "feasibility_details",
    "requirement_details",
    "diagnostics",
    "turn_alignment",
    "llm_user_simulation",
    "query",
    "runtime",
    "query_id",
)


def _copy_payload(value: Any) -> Any:
    return copy.deepcopy(value)


def _compact_soft_check(check: Any) -> Any:
    if not isinstance(check, dict):
        return _copy_payload(check)
    return {
        key: _copy_payload(value)
        for key, value in check.items()
        if key not in DEBUG_ONLY_SOFT_CHECK_KEYS
    }


def _compact_soft_preferences(details: Any) -> Any:
    if not isinstance(details, dict):
        return _copy_payload(details)
    compact = {}
    for key, value in details.items():
        if key == "checks" and isinstance(value, list):
            compact[key] = [_compact_soft_check(check) for check in value]
        else:
            compact[key] = _copy_payload(value)
    return compact


def _compact_feasibility_details(details: Any) -> Any:
    if not isinstance(details, dict):
        return _copy_payload(details)
    return {
        key: _copy_payload(value)
        for key, value in details.items()
        if key not in DEBUG_ONLY_FEASIBILITY_KEYS
    }


def _compact_requirement_details(details: Any) -> Any:
    if not isinstance(details, dict):
        return _copy_payload(details)
    compact = {}
    for key, value in details.items():
        if key in DEBUG_ONLY_REQUIREMENT_KEYS:
            continue
        if key == "soft_preferences":
            compact[key] = _compact_soft_preferences(value)
        else:
            compact[key] = _copy_payload(value)
    return compact


def _compact_diagnostics(details: Any) -> Any:
    if isinstance(details, dict):
        return {
            key: _compact_diagnostics(value)
            for key, value in details.items()
            if key not in DEBUG_ONLY_DIAGNOSTIC_NESTED_KEYS
        }
    if isinstance(details, list):
        return [_compact_diagnostics(item) for item in details]
    return _copy_payload(details)


def compact_score_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return compact score JSON for summaries and manual review."""
    if not isinstance(payload, dict):
        return {"payload": _copy_payload(payload)}

    compact: Dict[str, Any] = {}
    for key in COMPACT_TOP_LEVEL_KEYS:
        if key not in payload or key in DEBUG_ONLY_TOP_LEVEL_KEYS:
            continue
        value = payload[key]
        if key == "feasibility_details":
            compact[key] = _compact_feasibility_details(value)
        elif key == "requirement_details":
            compact[key] = _compact_requirement_details(value)
        elif key == "diagnostics":
            compact[key] = _compact_diagnostics(value)
        else:
            compact[key] = _copy_payload(value)

    for key, value in payload.items():
        if (
            key in compact
            or key in COMPACT_TOP_LEVEL_KEYS
            or key in DEBUG_ONLY_TOP_LEVEL_KEYS
        ):
            continue
        compact[key] = _copy_payload(value)
    return compact


def score_debug_path(output_file: Path) -> Path:
    if output_file.name.endswith("_score.json"):
        debug_name = output_file.name.replace("_score.json", "_score_debug.json")
    else:
        debug_name = f"{output_file.stem}_debug{output_file.suffix}"
    return output_file.parent / DEBUG_SCORE_DIR_NAME / debug_name


def write_score_payload(payload: Dict[str, Any], output_file: Path) -> None:
    """Write compact score JSON and full debug score JSON side by side."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(compact_score_payload(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    debug_file = score_debug_path(output_file)
    debug_file.parent.mkdir(parents=True, exist_ok=True)
    debug_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def multiturn_main_summary_path(output_dir: Path) -> Path:
    return output_dir / MULTITURN_MAIN_SUMMARY_FILE_NAME


def multiturn_full_summary_path(output_dir: Path) -> Path:
    return output_dir / DEBUG_SCORE_DIR_NAME / MULTITURN_FULL_SUMMARY_FILE_NAME


def _select_keys(data: Any, keys: Tuple[str, ...]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    return {key: _copy_payload(data[key]) for key in keys if key in data}


def _compact_multiturn_diagnostics(diagnostics: Any) -> Dict[str, Any]:
    return _select_keys(
        diagnostics,
        ("delivered_itinerary",),
    )


def _compact_turn_alignment_summary(turn_alignment: Any) -> Dict[str, Any]:
    return _select_keys(
        turn_alignment,
        (
            "fulfillment_score",
            "preserve_score",
            "preserve_check_count",
            "preserve_error_count",
            "preserve_error_rate",
            "breakdowns",
        ),
    )


def _compact_llm_user_simulation_summary(
    llm_user_simulation: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(llm_user_simulation, dict):
        return None
    return _select_keys(
        llm_user_simulation,
        (
            "status",
            "score",
            "score_1_5",
            "simulator_model",
            "artifact_path",
            "runtime",
        ),
    )


def compact_multiturn_summary(
    results: Dict[str, Any], output_dir: Path
) -> Dict[str, Any]:
    """Return the small top-level multi-turn summary kept in evaluation/."""
    compact: Dict[str, Any] = {
        key: _copy_payload(value) for key, value in results.items() if key != "results"
    }
    compact["summary_files"] = {
        "full_debug_summary": str(multiturn_full_summary_path(output_dir)),
        "per_turn_score_pattern": str(
            output_dir / "{sample_id}_turn_{turn_id}_score.json"
        ),
        "per_turn_debug_score_pattern": str(
            output_dir
            / DEBUG_SCORE_DIR_NAME
            / "{sample_id}_turn_{turn_id}_score_debug.json"
        ),
    }

    compact_results: List[Dict[str, Any]] = []
    for record in results.get("results", []) or []:
        if not isinstance(record, dict):
            continue
        sample_id = str(record.get("sample_id") or record.get("id") or "").strip()
        compact_record = _select_keys(
            record,
            ("sample_id", "base_query_id", "interaction_type", "database_path"),
        )
        compact_turns: List[Dict[str, Any]] = []
        for turn in record.get("turn_results", []) or []:
            if not isinstance(turn, dict):
                continue
            turn_id = turn.get("turn_id")
            evaluation_result = (
                turn.get("evaluation_result")
                if isinstance(turn.get("evaluation_result"), dict)
                else {}
            )
            turn_sample_id = str(
                evaluation_result.get("sample_id") or f"{sample_id}_turn_{turn_id}"
            )
            score_file = output_dir / f"{sample_id}_turn_{turn_id}_score.json"
            compact_eval = {
                "sample_id": turn_sample_id,
                **_select_keys(
                    evaluation_result,
                    ("query_id", "evaluation_mode", "scores", "runtime"),
                ),
            }
            response_expectation = _select_keys(
                evaluation_result.get("response_expectation_details"),
                ("passed", "expected", "actual_mode", "actual_status", "message"),
            )
            if response_expectation:
                compact_eval["response_expectation_details"] = response_expectation
            diagnostics = _compact_multiturn_diagnostics(
                evaluation_result.get("diagnostics")
            )
            if diagnostics:
                compact_eval["diagnostics"] = diagnostics
            turn_alignment = _compact_turn_alignment_summary(
                evaluation_result.get("turn_alignment")
            )
            if turn_alignment:
                compact_eval["turn_alignment"] = turn_alignment
            llm_user_simulation = _compact_llm_user_simulation_summary(
                evaluation_result.get("llm_user_simulation")
            )
            if llm_user_simulation:
                compact_eval["llm_user_simulation"] = llm_user_simulation

            compact_turn = _select_keys(turn, ("turn_id", "success", "query"))
            compact_turn["score_file"] = str(score_file)
            compact_turn["debug_score_file"] = str(score_debug_path(score_file))
            compact_turn["evaluation_result"] = compact_eval
            compact_turns.append(compact_turn)
        compact_record["turn_results"] = compact_turns
        compact_results.append(compact_record)
    compact["results"] = compact_results
    return compact


def write_multiturn_summary_files(
    results: Dict[str, Any], output_dir: Path
) -> Tuple[Path, Path]:
    """Write a slim main multi-turn summary plus the full debug summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = multiturn_full_summary_path(output_dir)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    main_path = multiturn_main_summary_path(output_dir)
    main_summary = compact_multiturn_summary(results, output_dir)
    main_path.write_text(
        json.dumps(main_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return main_path, full_path


def test_data_has_multiturn_records(test_data_path: Path) -> bool:
    try:
        data = json.loads(test_data_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return False
    return any(
        isinstance(sample, dict) and isinstance(sample.get("turns"), list)
        for sample in data
    )


def write_multiturn_score_files(results: Dict[str, Any], output_dir: Path) -> int:
    """Write one per-turn score JSON beside the multi-turn summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for record in results.get("results", []) or []:
        if not isinstance(record, dict):
            continue
        sample_id = str(record.get("sample_id") or record.get("id") or "").strip()
        if not sample_id:
            continue
        for turn in record.get("turn_results", []) or []:
            if not isinstance(turn, dict):
                continue
            turn_id = str(turn.get("turn_id"))
            evaluation_result = turn.get("evaluation_result")
            if isinstance(evaluation_result, dict):
                payload = evaluation_result
            else:
                payload = {
                    "sample_id": f"{sample_id}_turn_{turn_id}",
                    "parent_id": sample_id,
                    "turn_id": turn.get("turn_id"),
                    "query": turn.get("query") or "",
                    "success": False,
                    "error": turn.get("error") or "missing_evaluation_result",
                    "ground_truth": turn.get("ground_truth"),
                }
            output_file = output_dir / f"{sample_id}_turn_{turn_id}_score.json"
            write_score_payload(payload, output_file)
            written += 1
    return written
