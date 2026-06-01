"""Batch inference runner for benchmark query files."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from .input_processing import (
    build_multiturn_chat_user_message,
    build_multiturn_turn_query,
    is_multiturn_sample,
    planner_meta_from_sample,
)
from .prompts import get_system_prompt
from .planner import TravelPlannerAgent
from tools.sample_db_resolver import resolve_sample_database_path_with_query


REQUIRED_SAMPLE_DB_FILES = (
    "locations/locations_coords.csv",
    "restaurants/restaurants.csv",
    "transportation/distance_matrix.csv",
)


def failed_inference_report(error: str) -> str:
    """Return a parseable, zero-credit report for failed inference."""
    message = f"Model inference failed and did not produce an executable itinerary: {error}"
    return f"<plan>\n{message}\n</plan>"


def _verify_sample_database(sample_id: object, sample_db_path: Path) -> None:
    if not sample_db_path.exists():
        raise RuntimeError(
            f"Sample DB preflight failed for {sample_id}: missing sample DB directory: {sample_db_path}"
        )

    missing = [relative for relative in REQUIRED_SAMPLE_DB_FILES if not (sample_db_path / relative).exists()]
    if missing:
        raise RuntimeError(
            f"Sample DB preflight failed for {sample_id}: missing required files: {', '.join(missing)}"
        )


def _preflight_sample_databases(
    samples: List[Dict[str, Any]],
    *,
    database_root: Path,
    language: str,
    query_file: Path,
) -> Dict[str, Any]:
    checked = 0
    failures: List[str] = []
    print(f"  🔎 Preflight sample DBs: resolving/verifying {len(samples)} samples before inference")
    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample.get("id", "")).strip()
        if not sample_id:
            failures.append(f"sample at index {index} has no id")
            continue
        try:
            sample_db_path = resolve_sample_database_path_with_query(
                sample_id=sample_id,
                database_root=database_root,
                language=language,
                query_file=query_file,
            )
            _verify_sample_database(sample_id, sample_db_path)
            checked += 1
        except Exception as exc:
            failures.append(f"{sample_id}: {exc}")
        if index == 1 or index % 50 == 0 or index == len(samples):
            print(f"     sample DB preflight {index}/{len(samples)}")

    if failures:
        preview = "\n".join(f"     - {failure}" for failure in failures[:10])
        raise RuntimeError(
            f"Sample DB preflight failed for {len(failures)}/{len(samples)} samples:\n{preview}"
        )
    return {"checked": checked}


def _merge_runtime_stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "llm_calls": 0,
        "tool_calls": 0,
        "tool_executions": 0,
        "duplicate_tool_calls": 0,
        "tool_errors": 0,
        "token_usage": {},
    }
    calls: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        merged["llm_calls"] += int(item.get("llm_calls") or 0)
        tool_calls = int(item.get("tool_calls") or 0)
        duplicate_tool_calls = int(item.get("duplicate_tool_calls") or 0)
        tool_executions = (
            int(item.get("tool_executions") or 0)
            if "tool_executions" in item
            else max(0, tool_calls - duplicate_tool_calls)
        )
        merged["tool_calls"] += tool_calls
        merged["tool_executions"] += tool_executions
        merged["duplicate_tool_calls"] += duplicate_tool_calls
        merged["tool_errors"] += int(item.get("tool_errors") or 0)
        for key, value in (item.get("token_usage") or {}).items():
            if isinstance(value, (int, float)):
                merged["token_usage"][key] = merged["token_usage"].get(key, 0) + value
        for call in item.get("llm_call_usage") or []:
            if isinstance(call, dict):
                calls.append(call)
    if calls:
        merged["llm_call_usage"] = calls
    return merged


def run_agent_inference(
    model: str,
    language: str,
    test_data_path: Path,
    database_dir: Path,
    tool_schema_path: Path,
    output_dir: Path,
    workers: int = 10,
    max_llm_calls: int = 100,
    verbose: bool = False,
    rerun_ids: Optional[List[int]] = None,
    query_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run agent inference over a benchmark query file."""
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if rerun_ids is not None:
        rerun_ids_set = set(str(id) for id in rerun_ids)
        original_count = len(test_data)
        filtered_data = []
        for sample in test_data:
            sample_id = str(sample.get("id"))
            if sample_id in rerun_ids_set:
                filtered_data.append(sample)
                continue
            turn_ids = {
                f"{sample_id}_turn_{turn.get('turn_id', turn_index)}"
                for turn_index, turn in enumerate(sample.get("turns", []) or [])
                if isinstance(turn, dict)
            }
            if turn_ids & rerun_ids_set:
                filtered_data.append(sample)
        test_data = filtered_data
        print(f"  🔄 Filtered {original_count} samples to {len(test_data)} samples for rerun")
        if len(test_data) == 0:
            print("  ⚠️  Warning: No samples found matching the specified IDs")
            return {"total": 0, "success": 0, "failed": 0, "elapsed_time": 0, "results": []}

    print(f"\n{'=' * 80}")
    print("Agent Inference")
    print(f"{'=' * 80}")
    print(f"Model: {model}")
    print(f"Language: {language}")
    print(f"Samples: {len(test_data)}")
    print(f"Workers: {workers}")
    print(f"{'=' * 80}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trajectories").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)
    sample_db_preflight = _preflight_sample_databases(
        test_data,
        database_root=database_dir,
        language=language,
        query_file=test_data_path,
    )

    print_lock = Lock()
    results = []

    def process_sample(sample):
        sample_id_raw = sample.get("id", "unknown")
        sample_id = f"id_{sample_id_raw}" if str(sample_id_raw).isdigit() else str(sample_id_raw)
        query = sample.get("query", "")
        if query_overrides:
            query = query_overrides.get(str(sample_id_raw), query)

        try:
            with print_lock:
                print(f"\n🚀 Processing sample: {sample_id}")

            if is_multiturn_sample(sample):
                prompt_meta = planner_meta_from_sample(sample)
                turn_results: List[Dict[str, Any]] = []
                chat_history: List[Dict[str, str]] = []
                prior_turn_failed = False

                for turn_index, turn in enumerate([t for t in sample.get("turns", []) if isinstance(t, dict)]):
                    turn_id = str(turn.get("turn_id", turn_index))
                    turn_sample_id = f"{sample_id}_turn_{turn_id}"
                    turn_query = build_multiturn_chat_user_message(sample, turn_index, language)
                    if not turn_query:
                        turn_query = build_multiturn_turn_query(sample, turn_index, language)
                    prior_chat_messages = list(chat_history)

                    with print_lock:
                        print(f"   🔁 Processing turn: {turn_sample_id}")

                    if prior_turn_failed:
                        error_message = "Skipped because a previous turn in this multi-turn sample failed."
                        final_plan = failed_inference_report(error_message)
                        serialized_messages = [
                            *prior_chat_messages,
                            {"role": "user", "content": turn_query},
                            {"role": "assistant", "content": final_plan},
                        ]
                        result = {
                            "id": turn_sample_id,
                            "parent_id": sample_id,
                            "turn_id": turn_id,
                            "query": turn_query,
                            "model": model,
                            "language": language,
                            "context_mode": "chat_history",
                            "prior_chat_messages": prior_chat_messages,
                            "final_plan": final_plan,
                            "messages": serialized_messages,
                            "elapsed_time": 0.0,
                            "runtime_stats": {"status": "skipped_after_failed_turn"},
                            "success": True,
                            "inference_failed": True,
                            "error": error_message,
                        }
                        turn_results.append(result)
                        _write_result_files(result, output_dir, turn_sample_id, final_plan)
                        with print_lock:
                            print(f"⚠️  Sample {turn_sample_id}: {error_message}")
                        continue

                    agent = TravelPlannerAgent(
                        model=model,
                        sample_id=sample_id_raw,
                        database_base_path=database_dir,
                        test_data_path=str(test_data_path),
                        tool_schema_path=str(tool_schema_path),
                        language=language,
                        verbose=verbose,
                    )

                    system_prompt = get_system_prompt(language, prompt_meta)
                    start_time = time.time()
                    try:
                        final_plan, full_messages = agent.run(
                            user_query=turn_query,
                            system_prompt=system_prompt,
                            max_llm_calls=max_llm_calls,
                            initial_messages=prior_chat_messages,
                        )
                    except Exception as exc:
                        final_plan = failed_inference_report(str(exc))
                        full_messages = [
                            *prior_chat_messages,
                            {"role": "user", "content": turn_query},
                            {"role": "assistant", "content": final_plan},
                        ]
                        prior_turn_failed = True
                    elapsed = time.time() - start_time
                    serialized_messages = agent._serialize_messages(full_messages)
                    plan_extracted = bool(final_plan and final_plan.strip())
                    if not plan_extracted:
                        final_plan = failed_inference_report("No plan extracted from assistant output")
                        full_messages = [
                            *prior_chat_messages,
                            {"role": "user", "content": turn_query},
                            {"role": "assistant", "content": final_plan},
                        ]
                        serialized_messages = agent._serialize_messages(full_messages)
                        prior_turn_failed = True

                    result = {
                        "id": turn_sample_id,
                        "parent_id": sample_id,
                        "turn_id": turn_id,
                        "query": turn_query,
                        "model": model,
                        "language": language,
                        "context_mode": "chat_history",
                        "prior_chat_messages": prior_chat_messages,
                        "final_plan": final_plan,
                        "messages": serialized_messages,
                        "elapsed_time": elapsed,
                        "runtime_stats": agent.runtime_stats,
                        "success": True,
                    }
                    if prior_turn_failed or not plan_extracted:
                        result["inference_failed"] = True
                        result["error"] = "No plan extracted from assistant output" if not plan_extracted else final_plan
                    turn_results.append(result)
                    _write_result_files(result, output_dir, turn_sample_id, final_plan)
                    if not plan_extracted:
                        with print_lock:
                            print(f"⚠️  Sample {turn_sample_id}: No plan extracted")
                    with print_lock:
                        print(f"✅ Sample {turn_sample_id} completed in {elapsed:.2f}s")

                    chat_history.append({"role": "user", "content": turn_query})
                    if plan_extracted:
                        chat_history.append({"role": "assistant", "content": final_plan})

                return {
                    "id": sample_id,
                    "query": sample.get("base_query") or query,
                    "model": model,
                    "language": language,
                    "success": all((turn.get("final_plan") or "").strip() for turn in turn_results),
                    "turns": turn_results,
                    "runtime_stats": _merge_runtime_stats([item.get("runtime_stats") or {} for item in turn_results]),
                    "elapsed_time": sum(float(item.get("elapsed_time") or 0.0) for item in turn_results),
                }

            agent = TravelPlannerAgent(
                model=model,
                sample_id=sample_id_raw,
                database_base_path=database_dir,
                test_data_path=str(test_data_path),
                tool_schema_path=str(tool_schema_path),
                language=language,
                verbose=verbose,
            )
            system_prompt = get_system_prompt(language, planner_meta_from_sample(sample))
            start_time = time.time()
            final_plan, full_messages = agent.run(
                user_query=query,
                system_prompt=system_prompt,
                max_llm_calls=max_llm_calls,
            )
            elapsed = time.time() - start_time
            serialized_messages = agent._serialize_messages(full_messages)
            plan_extracted = bool(final_plan and final_plan.strip())
            if not plan_extracted:
                final_plan = failed_inference_report("No plan extracted from assistant output")
                full_messages = [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": final_plan},
                ]
                serialized_messages = agent._serialize_messages(full_messages)

            result = {
                "id": sample_id,
                "query": query,
                "model": model,
                "language": language,
                "final_plan": final_plan,
                "messages": serialized_messages,
                "elapsed_time": elapsed,
                "runtime_stats": agent.runtime_stats,
                "success": True,
            }
            if not plan_extracted:
                result["inference_failed"] = True
                result["error"] = "No plan extracted from assistant output"
            _write_result_files(result, output_dir, sample_id, final_plan)
            if not plan_extracted:
                with print_lock:
                    print(f"⚠️  Sample {sample_id}: No plan extracted")
            with print_lock:
                print(f"✅ Sample {sample_id} completed in {elapsed:.2f}s")
            return result

        except Exception as e:
            final_plan = failed_inference_report(str(e))
            result = {
                "id": sample_id,
                "query": query,
                "model": model,
                "language": language,
                "final_plan": final_plan,
                "messages": [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": final_plan},
                ],
                "elapsed_time": 0.0,
                "runtime_stats": {"status": "inference_exception"},
                "success": True,
                "inference_failed": True,
                "error": str(e),
            }
            try:
                _write_result_files(result, output_dir, sample_id, final_plan)
            except Exception as write_exc:
                result["success"] = False
                result["write_error"] = str(write_exc)
            with print_lock:
                print(f"❌ Sample {sample_id} failed: {e}")
            return result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_sample, sample) for sample in test_data]
        for future in as_completed(futures):
            results.append(future.result())

    success_count = sum(1 for result in results if result["success"])
    return {
        "total": len(results),
        "success": success_count,
        "failed": len(results) - success_count,
        "sample_db_preflight": sample_db_preflight,
        "results": results,
    }


def _write_result_files(result: Dict[str, Any], output_dir: Path, sample_id: str, final_plan: str) -> None:
    trajectory_file = output_dir / "trajectories" / f"{sample_id}.json"
    with open(trajectory_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    plan_file = output_dir / "reports" / f"{sample_id}.txt"
    with open(plan_file, "w", encoding="utf-8") as f:
        f.write(final_plan)
