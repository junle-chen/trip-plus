"""Batch orchestration for completed-run user simulation."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.call_llm import load_model_config
from evaluation.multiturn.files import find_turn_plan
from evaluation.multiturn.ground_truth import derive_turn_ground_truth
from simulation.experience_trace import build_experience_trace, count_plan_activities
from simulation.simulator import (
    build_failed_summary,
    run_chunked_simulation,
    run_simulation,
    summarize_simulation,
    write_simulation_artifact,
)
from simulation.models import slug

REUSE_SIMULATIONS_ENV = "USER_SIMULATION_REUSE_EXISTING"
CHUNKED_SIMULATOR_ENV = "USER_SIMULATION_CHUNKED"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _normalize_result_dir(path: Path) -> Path:
    path = resolve_repo_path(path)
    if path.name == "converted_plans":
        return path.parent
    if path.name == "evaluation":
        return path.parent
    return path


def summary_path(result_dir: Path) -> Path | None:
    evaluation_dir = result_dir / "evaluation"
    for name in ("_summary.json", "multiturn_summary.json"):
        candidate = evaluation_dir / name
        if candidate.exists():
            return candidate
    return None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def display_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    diagnostics = (
        summary.get("diagnostics")
        if isinstance(summary.get("diagnostics"), dict)
        else {}
    )
    metric_groups = (
        summary.get("metric_groups")
        if isinstance(summary.get("metric_groups"), dict)
        else {}
    )
    successful = (
        metric_groups.get("successful_plans")
        if isinstance(metric_groups.get("successful_plans"), dict)
        else {}
    )
    return {
        "all_response_user_sim": metrics.get("llm_user_simulation_score"),
        "successful_plan_user_sim": successful.get("llm_user_simulation_score"),
        "user_sim_success_count": diagnostics.get("llm_user_simulation_success_count"),
        "user_sim_failed_count": diagnostics.get("llm_user_simulation_failed_count"),
        "runtime_totals": diagnostics.get("llm_user_simulation_runtime_totals"),
    }


def relative_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)


def _response_mode(plan: dict[str, Any]) -> str:
    status = str(plan.get("status") or "").strip().lower()
    has_daily_plans = bool(plan.get("daily_plans"))
    if status == "unsat" and not has_daily_plans:
        return "no_solution"
    if status == "clarification" and not has_daily_plans:
        return "clarification"
    if has_daily_plans and status not in {"unsat", "clarification"}:
        return "plan"
    return "invalid"


def _expected_mode(response_expectation: object) -> str:
    expected = str(response_expectation or "plan").strip().lower()
    if expected == "conflict_resolution":
        return "clarification"
    if expected in {"infeasible", "no_solution"}:
        return "no_solution"
    if expected == "clarification":
        return "clarification"
    return "plan"


def _turn_id_for_simulation(turn_id: object) -> int | None:
    try:
        return int(str(turn_id))
    except (TypeError, ValueError):
        return None


def _same_turn_id(left: object, right: object) -> bool:
    return str(left) == str(right)


def _last_turn_id_for_record(record: dict[str, Any]) -> object:
    turns = [turn for turn in record.get("turns", []) or [] if isinstance(turn, dict)]
    if not turns:
        return None

    def sort_key(turn: dict[str, Any]) -> tuple[int, str]:
        raw = turn.get("turn_id")
        try:
            return (0, f"{int(str(raw)):06d}")
        except (TypeError, ValueError):
            return (1, str(raw))

    return max(turns, key=sort_key).get("turn_id")


def _user_simulation_artifact_path(
    artifact_dir: Path, sample_id: str, turn_id: int | None
) -> Path:
    suffix = f"_turn_{turn_id}" if turn_id is not None else ""
    sample_label = str(sample_id)
    if not sample_label.startswith("id_"):
        sample_label = f"id_{sample_label}"
    return artifact_dir / f"{sample_label}{suffix}_user_simulation.json"


def _simulation_trace_matches(
    simulation: dict[str, Any], expected_trace: dict[str, Any]
) -> bool:
    artifact_trace = simulation.get("input_experience_trace")
    if not isinstance(artifact_trace, dict):
        return False
    if artifact_trace.get("expected_activity_refs") != expected_trace.get(
        "expected_activity_refs"
    ):
        return False
    if artifact_trace.get("activity_count") != expected_trace.get("activity_count"):
        return False
    artifact_source = artifact_trace.get("profile_source") or {}
    expected_source = expected_trace.get("profile_source") or {}
    if not isinstance(artifact_source, dict) or not isinstance(expected_source, dict):
        return False
    for key in (
        "used_observable_profile",
        "used_user_profile",
        "used_oracle_state_after_turn",
    ):
        if bool(artifact_source.get(key)) != bool(expected_source.get(key)):
            return False
    return True


def _load_existing_user_simulation_summary(
    artifact_dir: Path,
    sample_id: str,
    turn_id: int | None,
    expected_trace: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if os.environ.get(REUSE_SIMULATIONS_ENV, "1").strip().lower() in {
        "0",
        "false",
        "no",
    }:
        return None
    artifact_path = _user_simulation_artifact_path(artifact_dir, sample_id, turn_id)
    if not artifact_path.exists():
        return None
    try:
        simulation = load_json(artifact_path)
    except Exception:
        return None
    if expected_trace is not None and not _simulation_trace_matches(
        simulation, expected_trace
    ):
        return None
    summary = summarize_simulation(simulation, artifact_path)
    summary["reused_existing_artifact"] = True
    return summary


def _load_plan(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Converted plan must be a JSON object: {path}")
    return payload


def _final_turn_for_record(record: dict[str, Any]) -> dict[str, Any] | None:
    final_turn_id = _last_turn_id_for_record(record)
    for turn in record.get("turns", []) or []:
        if isinstance(turn, dict) and _same_turn_id(turn.get("turn_id"), final_turn_id):
            return turn
    return None


def _merge_token_usage(total: dict[str, Any], usage: dict[str, Any]) -> None:
    for key, value in usage.items():
        if isinstance(value, (int, float)):
            total[key] = int(total.get(key, 0)) + int(value)


def _aggregate_runtime_totals(results: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "llm_calls": 0,
        "tool_calls": 0,
        "tool_errors": 0,
        "token_usage": {},
        "token_usage_available_count": 0,
        "token_usage_missing_count": 0,
    }
    for item in results:
        runtime = item.get("runtime") if isinstance(item.get("runtime"), dict) else {}
        totals["llm_calls"] += int(runtime.get("llm_calls") or 0)
        totals["tool_calls"] += int(runtime.get("tool_calls") or 0)
        totals["tool_errors"] += int(runtime.get("tool_errors") or 0)
        usage = runtime.get("token_usage")
        if isinstance(usage, dict) and any(
            isinstance(value, (int, float)) for value in usage.values()
        ):
            totals["token_usage_available_count"] += 1
            _merge_token_usage(totals["token_usage"], usage)
        elif item.get("status") == "ok":
            totals["token_usage_missing_count"] += 1
    if totals["token_usage_available_count"] == 0:
        totals["token_usage_status"] = "unavailable"
        totals["token_usage"] = None
    elif totals["token_usage_missing_count"]:
        totals["token_usage_status"] = "partial"
    else:
        totals["token_usage_status"] = "available"
    return totals


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def parse_model_filter(values: list[str] | None) -> set[str]:
    if not values:
        return set()
    items: set[str] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                items.add(slug(part))
    return items


def discover_runs(
    *,
    result_root: Path,
    language: str,
    run_pattern: str | None,
    model_filter: set[str],
    latest_per_model: bool,
) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    if not result_root.exists():
        return runs

    suffix = f"_{language}"
    for plans_dir in result_root.glob("**/converted_plans"):
        result_dir = plans_dir.parent
        if not any(plans_dir.glob("id_*_converted.json")):
            continue
        if suffix and not result_dir.name.endswith(suffix):
            continue
        if run_pattern and run_pattern not in str(result_dir):
            continue

        display_name = result_dir.name.removesuffix(suffix)
        if model_filter:
            model_keys = {
                slug(display_name),
                slug(result_dir.name.removesuffix(suffix)),
                slug(result_dir.parent.name),
                slug(result_dir.parent.parent.name) if result_dir.parent.parent else "",
            }
            if model_keys.isdisjoint(model_filter):
                continue
        runs.append((display_name, result_dir))

    runs.sort(key=lambda item: item[1].stat().st_mtime, reverse=True)
    if not latest_per_model:
        return list(reversed(runs))

    selected: dict[str, tuple[str, Path]] = {}
    for display_name, result_dir in runs:
        selected.setdefault(slug(display_name), (display_name, result_dir))
    return sorted(selected.values(), key=lambda item: item[0].lower())


def custom_runs(result_dirs: list[Path]) -> list[tuple[str, Path]]:
    runs = []
    for path in result_dirs:
        result_dir = _normalize_result_dir(path)
        runs.append((result_dir.name.removesuffix("_en"), result_dir))
    return runs


def validate_result_dir(name: str, result_dir: Path) -> None:
    plans_dir = result_dir / "converted_plans"
    if not plans_dir.exists():
        raise FileNotFoundError(f"{name}: missing converted plans dir: {plans_dir}")
    if not any(plans_dir.glob("id_*_converted.json")):
        raise FileNotFoundError(
            f"{name}: no converted plan files found in: {plans_dir}"
        )


def check_simulator_config(model: str) -> dict[str, Any]:
    config = load_model_config(model)
    base_url = str(config.get("base_url") or "").rstrip("/")
    api_key_env = str(config.get("api_key_env") or "")
    model_type = str(config.get("model_type") or "openai").strip().lower()
    if model_type != "openai":
        raise ValueError(
            f"{model} must use an OpenAI-compatible config, got model_type={model_type!r}"
        )
    if not base_url:
        base_url_env = str(config.get("base_url_env") or "")
        if base_url_env:
            raise RuntimeError(
                f"Missing {base_url_env}. Set it in the environment or .env before running."
            )
        raise ValueError(f"{model} must define base_url in models_config.json.")
    if not api_key_env:
        raise ValueError(f"{model} must define api_key_env in models_config.json.")
    if not os.environ.get(api_key_env):
        raise RuntimeError(
            f"Missing {api_key_env}. Set it in the environment or .env before running."
        )
    return config


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def target_id_filter(
    records: list[dict[str, Any]], target_ids: list[str] | None
) -> list[dict[str, Any]]:
    if not target_ids:
        return records
    wanted = {
        str(item).strip().removeprefix("id_")
        for item in target_ids
        if str(item).strip()
    }
    return [
        record
        for record in records
        if str(record.get("id") or "").removeprefix("id_") in wanted
        or str(record.get("base_query_id") or "").removeprefix("id_") in wanted
    ]


def _prepare_user_simulation_item(
    record: dict[str, Any], result_dir: Path
) -> dict[str, Any]:
    sample_id = str(record.get("id") or "").strip()
    final_turn = _final_turn_for_record(record)
    if not sample_id:
        return {"status": "skipped", "skip_reason": "missing_sample_id"}
    if final_turn is None:
        return {
            "sample_id": sample_id,
            "status": "skipped",
            "skip_reason": "missing_final_turn",
        }

    turn_id = final_turn.get("turn_id")
    plan_path = find_turn_plan(
        result_dir / "converted_plans", sample_id, turn_id, record.get("base_query_id")
    )
    if plan_path is None:
        return {
            "sample_id": sample_id,
            "turn_id": turn_id,
            "status": "skipped",
            "skip_reason": "missing_final_turn_plan",
        }

    plan = _load_plan(plan_path)
    turn_gt = derive_turn_ground_truth(record, final_turn)
    meta = turn_gt["meta_info"] if isinstance(turn_gt.get("meta_info"), dict) else {}
    solution_status = str(meta.get("solution_status", "sat")).strip().lower()
    response_expectation = (
        str(meta.get("response_expectation") or "plan").strip().lower()
    )
    expected_mode = _expected_mode(response_expectation)
    actual_mode = _response_mode(plan)

    if solution_status == "unsat" or expected_mode != "plan":
        return {
            "sample_id": sample_id,
            "turn_id": turn_id,
            "plan_file": relative_path(plan_path),
            "status": "skipped",
            "skip_reason": "final_turn_not_plan_expected",
            "solution_status": solution_status,
            "response_expectation": response_expectation,
            "expected_mode": expected_mode,
            "actual_mode": actual_mode,
        }
    if actual_mode != "plan":
        return {
            "sample_id": sample_id,
            "turn_id": turn_id,
            "plan_file": relative_path(plan_path),
            "status": "skipped",
            "skip_reason": "final_turn_output_not_plan",
            "solution_status": solution_status,
            "response_expectation": response_expectation,
            "expected_mode": expected_mode,
            "actual_mode": actual_mode,
        }

    turn_for_simulation = _turn_id_for_simulation(turn_id)
    return {
        "sample_id": sample_id,
        "turn_id": turn_id,
        "turn_for_simulation": turn_for_simulation,
        "plan_file": relative_path(plan_path),
        "status": "eligible",
        "record": record,
        "plan": plan,
        "activity_count": count_plan_activities(plan),
    }


def _run_user_simulation_item(
    item: dict[str, Any],
    *,
    simulator_model: str,
    simulation_dir: Path,
    require_user_simulator: bool,
    reuse_existing: bool,
) -> dict[str, Any]:
    if item.get("status") != "eligible":
        return item

    sample_id = str(item["sample_id"])
    turn_for_simulation = item.get("turn_for_simulation")
    record = item["record"]
    plan = item["plan"]
    result = {
        key: value for key, value in item.items() if key not in {"record", "plan"}
    }

    evaluation_context: dict[str, Any] = {}
    if reuse_existing:
        expected_trace = build_experience_trace(
            query_record=record,
            plan=plan,
            turn_id=turn_for_simulation,
            evaluation_context=evaluation_context,
        )
        existing_summary = _load_existing_user_simulation_summary(
            simulation_dir,
            sample_id,
            turn_for_simulation,
            expected_trace=expected_trace,
        )
        if existing_summary is not None:
            result.update(
                {
                    "status": "ok",
                    "reused_existing_artifact": True,
                    "summary": existing_summary,
                    "score": existing_summary.get("score"),
                    "score_1_5": existing_summary.get("score_1_5"),
                    "artifact_path": existing_summary.get("artifact_path"),
                    "runtime": existing_summary.get("runtime"),
                }
            )
            return result

    try:
        use_chunked_simulator = os.environ.get(CHUNKED_SIMULATOR_ENV, "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        simulator_fn = (
            run_chunked_simulation
            if use_chunked_simulator
            else run_simulation
        )
        simulation = simulator_fn(
            model=simulator_model,
            query_record=record,
            plan=plan,
            turn_id=turn_for_simulation,
            evaluation_context=evaluation_context,
        )
        artifact_path = write_simulation_artifact(
            artifact_dir=simulation_dir,
            sample_id=sample_id,
            simulation=simulation,
            turn_id=turn_for_simulation,
        )
        summary = summarize_simulation(simulation, artifact_path)
        result.update(
            {
                "status": "ok",
                "summary": summary,
                "score": summary.get("score"),
                "score_1_5": summary.get("score_1_5"),
                "artifact_path": str(artifact_path),
                "runtime": summary.get("runtime"),
            }
        )
    except Exception as error:
        if require_user_simulator:
            raise
        failed = build_failed_summary(error)
        result.update(
            {
                "status": "failed",
                "error": failed.get("error"),
            }
        )
    return result


def run_user_simulations_only(
    *,
    display_name: str,
    result_dir: Path,
    records: list[dict[str, Any]],
    simulator_model: str,
    output_dir: Path,
    workers: int,
    require_user_simulator: bool,
    reuse_existing: bool,
) -> dict[str, Any]:
    simulation_dir = output_dir / "user_simulations"
    prepared = [_prepare_user_simulation_item(record, result_dir) for record in records]
    eligible = [item for item in prepared if item.get("status") == "eligible"]
    skipped = [item for item in prepared if item.get("status") != "eligible"]
    results: list[dict[str, Any]] = []

    print(f"  eligible final-turn plans: {len(eligible)} / {len(prepared)}")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(
                _run_user_simulation_item,
                item,
                simulator_model=simulator_model,
                simulation_dir=simulation_dir,
                require_user_simulator=require_user_simulator,
                reuse_existing=reuse_existing,
            )
            for item in eligible
        ]
        for finished, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if result.get("status") == "ok":
                print(
                    f"    [{finished}/{len(eligible)}] ok "
                    f"{result.get('sample_id')}_turn_{result.get('turn_id')} "
                    f"score={result.get('score')}"
                )
            else:
                print(
                    f"    [{finished}/{len(eligible)}] {result.get('status')} "
                    f"{result.get('sample_id')}_turn_{result.get('turn_id')}: {result.get('error')}"
                )

    results.extend(skipped)
    results.sort(
        key=lambda item: (
            str(item.get("sample_id") or ""),
            str(item.get("turn_id") or ""),
        )
    )
    ok_scores = [
        float(item["score"])
        for item in results
        if item.get("status") == "ok" and item.get("score") is not None
    ]
    ok_scores_1_5 = [
        float(item["score_1_5"])
        for item in results
        if item.get("status") == "ok" and item.get("score_1_5") is not None
    ]

    counts: dict[str, int] = {}
    skip_reasons: dict[str, int] = {}
    for item in results:
        status = str(item.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
        if status == "skipped":
            reason = str(item.get("skip_reason") or "unknown")
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    compact_results = [
        {key: value for key, value in item.items() if key not in {"summary", "runtime"}}
        for item in results
    ]

    summary = {
        "mode": "user_simulation_only",
        "display_name": display_name,
        "source_result_dir": relative_path(result_dir),
        "simulator_model": simulator_model,
        "user_simulation_dir": relative_path(simulation_dir),
        "counts": {
            "records": len(prepared),
            "eligible": len(eligible),
            "ok": counts.get("ok", 0),
            "failed": counts.get("failed", 0),
            "skipped": counts.get("skipped", 0),
            "skip_reasons": skip_reasons,
        },
        "scores": {
            "mean_score": _mean(ok_scores),
            "mean_score_1_5": _mean(ok_scores_1_5),
        },
        "runtime_totals": _aggregate_runtime_totals(results),
        "results": compact_results,
    }
    write_json(output_dir / "user_simulation_summary.json", summary)
    return summary
