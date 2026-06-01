"""Single-turn evaluation runner for structured travel plans."""

import json
import re
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from .feasibility.runner import (
    calculate_feasibility,
    eval_itinerary_feasibility,
)
from .hard import calculate_hard_score, eval_hard
from .soft import calculate_user_alignment
from .output import (
    test_data_has_multiturn_records,
    write_multiturn_score_files,
    write_multiturn_summary_files,
    write_score_payload,
)
from .runtime import (
    attach_evaluation_metadata,
    load_runtime_stats_for_plan_file,
)
from .response_cases import (
    build_missing_itinerary_result,
    build_unsat_result,
    evaluate_unsat_case,
)
from .summary import build_evaluation_summary

from tools.sample_db_resolver import resolve_sample_database_path_with_query


def _build_evaluation_result(
    sample_id: str,
    plan: Dict[str, Any],
    meta: Dict[str, Any],
    feasibility_results: Dict[str, Tuple[bool, Optional[str]]],
    hard_result: Dict[str, Any],
    database_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    feasibility_details = calculate_feasibility(feasibility_results)
    expected_status = str(meta.get("solution_status", "sat")).strip().lower()
    plan_status = str(plan.get("status", "")).strip().lower()
    has_daily_plans = bool(plan.get("daily_plans"))
    if expected_status != "unsat" and plan_status == "unsat":
        feasibility_details.setdefault("checks", {})[
            "unexpected_no_solution_for_sat_query"
        ] = {
            "passed": False,
            "message": "The query is labeled satisfiable, but the agent returned <no_solution> instead of an executable itinerary.",
        }
        feasibility_details["strict_feasibility"] = 0.0
        hard_constraints = (
            hard_result.get("constraints", {}) if isinstance(hard_result, dict) else {}
        )
        hard_constraints = {
            **hard_constraints,
            "unexpected_no_solution_for_sat_query": {
                "passed": False,
                "message": "Expected a satisfiable itinerary, but the converted plan status is unsat.",
            },
        }
        gated_hard_result = {
            **(hard_result if isinstance(hard_result, dict) else {}),
            "score": 0.0,
            "strict_hard_constraint": 0.0,
            "hard_constraint_score": 0.0,
            "strict_soft_preference": None,
            "soft_preference_score": None,
            "constraints": hard_constraints,
        }
        return {
            "sample_id": sample_id,
            "evaluation_mode": "sat_query_unexpected_no_solution",
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
            "feasibility_details": feasibility_details,
            "requirement_details": gated_hard_result,
            "diagnostics": {
                "delivered_itinerary": False,
            },
        }

    if expected_status != "unsat" and not has_daily_plans:
        return build_missing_itinerary_result(sample_id, plan, hard_result)

    requirement_details = calculate_user_alignment(
        plan, meta, hard_result, database_dir=database_dir
    )

    feasibility_score = feasibility_details["score"]
    strict_feasibility = feasibility_details.get("strict_feasibility")
    requirement_score = requirement_details["score"]
    strict_hard_constraint = requirement_details.get("strict_hard_constraint")
    hard_constraint_score = requirement_details.get("hard_constraint_score")
    strict_soft_preference = requirement_details.get("strict_soft_preference")
    soft_preference_score = requirement_details.get("soft_preference_score")

    return {
        "sample_id": sample_id,
        "scores": {
            "feasibility_score": feasibility_score,
            "strict_feasibility": strict_feasibility,
            "strict_hard_constraint": strict_hard_constraint,
            "hard_constraint_score": hard_constraint_score,
            "strict_soft_preference": strict_soft_preference,
            "soft_preference_score": soft_preference_score,
            "requirement_score": requirement_score,
            "llm_user_simulation_score": None,
        },
        "feasibility_details": feasibility_details,
        "requirement_details": requirement_details,
        "hard_constraint_details": hard_result,
        "diagnostics": {
            "delivered_itinerary": True,
        },
    }


def process_single_evaluation(
    plan_file: Path,
    test_samples: List[Dict],
    test_data_path: Path,
    output_dir: Path,
    database_dir: Path,
    print_lock: Lock,
    language: str,
) -> Dict[str, Any]:
    """Evaluate one converted plan file and write its score artifact."""
    sample_id = None
    try:
        match = re.match(r"id_(\d+)_converted\.json", plan_file.name)
        sample_id = (
            match.group(1)
            if match
            else plan_file.stem.replace("_converted", "").replace("id_", "")
        )

        with print_lock:
            print(f"\n{'=' * 80}")
            print(f"Evaluating sample {sample_id}")
            print(f"   Plan file: {plan_file.name}")
            print(f"{'=' * 80}")

        sample_record = next(
            (
                sample
                for sample in test_samples
                if str(sample.get("id")) == str(sample_id)
            ),
            None,
        )
        if sample_record is None:
            raise ValueError(f"meta_info not found for sample {sample_id}")
        meta = sample_record.get("meta_info", {})
        sample_database_path = resolve_sample_database_path_with_query(
            sample_id=sample_id,
            database_root=database_dir,
            language=language,
            query_file=test_data_path,
        )

        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        runtime_stats = load_runtime_stats_for_plan_file(plan_file)
        query_text = str(
            sample_record.get("query") or sample_record.get("base_query") or ""
        )
        response_expectation = (
            str(meta.get("response_expectation") or "").strip().lower()
        )
        solution_status = str(meta.get("solution_status", "sat")).strip().lower()

        if response_expectation in {"infeasible", "no_solution"} or (
            solution_status == "unsat"
            and response_expectation not in {"clarification", "conflict_resolution"}
        ):
            special_eval = evaluate_unsat_case(plan, meta)
            evaluation_result = build_unsat_result(sample_id, special_eval)
            hard_result = special_eval["hard_constraint_dimension_score"]
        elif response_expectation in {"clarification", "conflict_resolution"}:
            expected_mode = "clarification"
            plan_status = str(plan.get("status") or "").strip().lower()
            has_daily_plans = bool(plan.get("daily_plans"))
            passed = plan_status == expected_mode and not has_daily_plans
            score = 1.0 if passed else 0.0
            hard_result = {
                "score": score,
                "constraints": {
                    "response_expectation": {
                        "passed": passed,
                        "message": None
                        if passed
                        else f"Expected {expected_mode} response, got status={plan_status or 'missing'} with daily_plans={has_daily_plans}.",
                        "details": {
                            "response_expectation": response_expectation,
                            "expected_mode": expected_mode,
                            "actual_status": plan_status,
                            "has_daily_plans": has_daily_plans,
                        },
                    }
                },
            }
            evaluation_result = {
                "sample_id": sample_id,
                "evaluation_mode": "clarification_response",
                "scores": {
                    "feasibility_score": score,
                    "strict_feasibility": score,
                    "strict_hard_constraint": score,
                    "hard_constraint_score": score,
                    "strict_soft_preference": None,
                    "soft_preference_score": None,
                    "requirement_score": score,
                    "llm_user_simulation_score": None,
                },
                "feasibility_details": {
                    "score": score,
                    "strict_feasibility": score,
                    "dimensions": {},
                    "subdimensions": {},
                    "checks": {},
                },
                "requirement_details": hard_result,
                "response_expectation_details": hard_result["constraints"][
                    "response_expectation"
                ],
                "diagnostics": {"delivered_itinerary": False},
            }
        elif not bool(plan.get("daily_plans")):
            hard_result = calculate_hard_score(eval_hard(plan, meta))
            evaluation_result = build_missing_itinerary_result(
                sample_id, plan, hard_result
            )
        else:
            feasibility = eval_itinerary_feasibility(
                plan, meta, database_dir=sample_database_path
            )
            hard_result = calculate_hard_score(eval_hard(plan, meta))
            evaluation_result = _build_evaluation_result(
                sample_id=sample_id,
                plan=plan,
                meta=meta,
                feasibility_results=feasibility,
                hard_result=hard_result,
                database_dir=sample_database_path,
            )

        attach_evaluation_metadata(
            evaluation_result,
            sample_record=sample_record,
            query=query_text,
            plan_file=plan_file,
            runtime_stats=runtime_stats,
        )
        output_file = output_dir / f"id_{sample_id}_score.json"
        write_score_payload(evaluation_result, output_file)

        scores = evaluation_result.get("scores") or {}
        feasibility_score = float(scores.get("feasibility_score") or 0.0)
        requirement_score = float(scores.get("requirement_score") or 0.0)
        constraints = (evaluation_result.get("requirement_details") or {}).get(
            "constraints"
        ) or {}
        hard_passed = sum(
            1
            for item in constraints.values()
            if isinstance(item, dict) and item.get("passed")
        )
        hard_total = len(constraints)
        with print_lock:
            print(f"Sample {sample_id} evaluation completed")
            print(f"   Feasibility: {feasibility_score:.2%}")
            print(
                f"   Requirement: {requirement_score:.2%} ({hard_passed}/{hard_total} hard constraints)"
            )
            print(f"   Output file: {output_file.name}\n")

        return {
            "success": True,
            "sample_id": sample_id,
            "scores": scores,
            "diagnostics": evaluation_result.get("diagnostics", {}),
            "runtime": evaluation_result.get("runtime", {}),
            "evaluation_result": evaluation_result,
        }
    except Exception as exc:
        with print_lock:
            print(f"Sample {sample_id or plan_file.name} evaluation failed: {exc}\n")
        return {
            "success": False,
            "sample_id": sample_id or plan_file.name,
            "error": str(exc),
        }


def evaluate_plans(
    result_dir: Path,
    test_data_path: Path,
    database_dir: Path,
    language: str,
    output_dir: Optional[Path] = None,
    workers: int = 10,
    target_ids: Optional[List[str]] = None,
) -> Dict:
    """
    Evaluate multiple converted plans

    Args:
        result_dir: Result directory containing 'converted_plans' subdirectory
        test_data_path: Path to test data JSON (contains meta_info)
        database_dir: Database root directory
        language: Dataset language. This release supports English only.
        output_dir: Optional output directory for evaluation artifacts
        workers: Number of concurrent workers
        target_ids: Optional list of sample IDs to evaluate (filters test data results)

    Returns:
        dict: Evaluation summary with statistics
    """
    # Set plans_dir and output_dir based on result_dir
    plans_dir = result_dir / "converted_plans"
    output_dir = output_dir or (result_dir / "evaluation")
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read test data
    print(f"\nLoading test data from {test_data_path}")
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Ensure test_data is a list
    if isinstance(test_data, dict):
        test_samples = [test_data]
    elif isinstance(test_data, list):
        test_samples = test_data
    else:
        raise ValueError("test_data.json format error: should be a dict or list")

    # Filter test samples if target_ids is provided
    if target_ids is not None:
        target_ids_set = set(str(tid) for tid in target_ids)
        test_samples = [s for s in test_samples if str(s.get("id")) in target_ids_set]
        if not test_samples:
            print(
                f"⚠️  Warning: No samples found in test data matching target_ids: {target_ids}"
            )

    total_test_samples = len(test_samples)

    # Create a set of valid sample IDs
    valid_sample_ids = set(str(sample.get("id")) for sample in test_samples)

    # Find all plan files
    all_plan_files = list(plans_dir.glob("id_*_converted.json"))
    if not all_plan_files:
        # Try without id_ prefix
        all_plan_files = list(plans_dir.glob("*_converted.json"))

    if not all_plan_files:
        print(f"⚠️  No plan files found in {plans_dir}")
        return {"total": 0, "success": 0, "failed": 0, "results": []}

    # Filter plan files to only include those in the test data
    plan_files = []
    skipped_files = []
    for plan_file in all_plan_files:
        # Extract sample_id from filename
        match = re.match(r"id_(\d+)_converted\.json", plan_file.name)
        if match:
            sample_id = match.group(1)
        else:
            sample_id = plan_file.stem.replace("_converted", "").replace("id_", "")

        if sample_id in valid_sample_ids:
            plan_files.append(plan_file)
        else:
            skipped_files.append((plan_file.name, sample_id))

    if not plan_files:
        print(f"⚠️  No plan files match the test data samples in {plans_dir}")
        return {"total": 0, "success": 0, "failed": 0, "results": []}

    print(f"\n{'=' * 80}")
    print(f"📊 Evaluation Overview:")
    print(f"   - Total samples in test data: {total_test_samples}")
    print(f"   - Total plan files found: {len(all_plan_files)}")
    print(f"   - Plan files to evaluate (in test data): {len(plan_files)}")
    if skipped_files:
        print(f"   - Plan files skipped (not in test data): {len(skipped_files)}")
    print(f"   - Converted file coverage: {len(plan_files) / total_test_samples:.2%}")
    print(f"🚀 Using {workers} threads for parallel processing")
    print(f"📂 Input Directory: {plans_dir}")
    print(f"📂 Output Directory: {output_dir}")
    print(f"📂 Database Directory: {database_dir}")
    print(f"📋 Test Data File: {test_data_path.name}")
    print(f"{'=' * 80}\n")

    # Create print lock
    print_lock = Lock()

    # Record start time
    start_time = time.time()

    # Use thread pool for parallel processing
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_file = {}
        for plan_file in plan_files:
            future = executor.submit(
                process_single_evaluation,
                plan_file,
                test_samples,
                test_data_path,
                output_dir,
                database_dir,
                print_lock,
                language,
            )
            future_to_file[future] = plan_file

        # Collect results (in completion order)
        for future in as_completed(future_to_file):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                plan_file = future_to_file[future]
                with print_lock:
                    print(
                        f"❌ File {plan_file.name} encountered uncaught exception: {e}\n"
                    )
                results.append(
                    {"success": False, "sample_id": plan_file.name, "error": str(e)}
                )

    elapsed_time = time.time() - start_time
    success_count = sum(1 for result in results if result["success"])
    failed_count = len(results) - success_count

    summary_data = build_evaluation_summary(
        total_test_samples=total_test_samples,
        plan_files_found=len(plan_files),
        results=results,
        success_count=success_count,
        failed_count=failed_count,
        elapsed_time=elapsed_time,
        workers=workers,
    )
    metrics = summary_data.get("metrics", {})
    diagnostics = summary_data.get("diagnostics", {})
    error_stats = summary_data.get("error_statistics", [])

    print(f"\n{'=' * 80}")
    print("All samples evaluated")
    print(f"{'=' * 80}")
    print("Statistics:")
    print(f"   - Total samples in test data: {total_test_samples}")
    print(f"   - Plan files evaluated: {len(plan_files)}")
    print(f"   - Evaluation success: {success_count}")
    print(f"   - Evaluation failed: {failed_count}")
    print(f"   - Total time: {elapsed_time:.2f} seconds")
    if plan_files:
        print(f"   - Average time: {elapsed_time / len(plan_files):.2f} seconds/sample")
    print("\nEvaluation metrics:")
    print(f"   Delivery rate: {diagnostics.get('delivery_rate', 0.0):.2%}")
    print(f"   Feasibility score: {metrics.get('feasibility_score', 0.0):.2%}")
    print(f"   Strict feasibility: {metrics.get('strict_feasibility', 0.0):.2%}")
    print(f"   Requirement score: {metrics.get('requirement_score', 0.0):.2%}")
    print(
        f"   Hard constraint score: {(metrics.get('hard_constraint_score') or 0.0):.2%}"
    )
    soft_score = metrics.get("soft_preference_score")
    if soft_score is not None:
        print(f"   Soft preference score: {float(soft_score):.2%}")

    feasibility_dimensions = metrics.get("feasibility_dimensions") or {}
    if feasibility_dimensions:
        print("\nFeasibility by dimension:")
        for dim_name, dim_metrics in feasibility_dimensions.items():
            print(
                f"   - {dim_name} (weight={dim_metrics.get('weight', 0.0):.0%}): {dim_metrics.get('score', 0.0):.2%}"
            )

    if error_stats:
        print("\nError statistics:")
        for item in error_stats[:10]:
            print(f"{item['rank']}. {item['error_type']}")
            print(f"   Occurrences: {item['count']}")
            print(
                f"   Affected samples: {', '.join(str(sample) for sample in item['affected_samples'][:5])}"
            )
            if item.get("sample_messages"):
                sample_msg = str(item["sample_messages"][0])
                print(
                    f"   Example message: {sample_msg[:100]}{'...' if len(sample_msg) > 100 else ''}"
                )
    else:
        print("\nNo constraint violations found")

    print(f"\nOutput directory: {output_dir}")
    print(f"{'=' * 80}\n")

    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Evaluation summary saved to: {summary_path}\n")

    return {
        "total": len(plan_files),
        "success": success_count,
        "failed": failed_count,
        "average_score": metrics.get("requirement_score"),
        "results": results,
        "metrics": metrics,
        "breakdowns": summary_data.get("breakdowns", {}),
        "diagnostics": diagnostics,
        "elapsed_time": elapsed_time,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate converted travel plans")
    parser.add_argument(
        "--plans-dir",
        type=Path,
        required=True,
        help="Directory containing converted plan JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        required=True,
        help="Test data JSON file (contains meta_info)",
    )
    parser.add_argument(
        "--database-dir", type=Path, required=True, help="Database root directory"
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["en"],
        help="Dataset language (English-only)",
    )
    parser.add_argument(
        "--workers", type=int, default=10, help="Number of concurrent workers"
    )
    args = parser.parse_args()

    if test_data_has_multiturn_records(args.test_data):
        from .multiturn.runner import evaluate_multiturn_plans

        result = evaluate_multiturn_plans(
            query_file=args.test_data,
            plans_dir=args.plans_dir,
            database_root=args.database_dir,
            language=args.language,
            workers=args.workers,
        )
        detail_count = write_multiturn_score_files(result, args.output_dir)
        summary_path, full_summary_path = write_multiturn_summary_files(
            result, args.output_dir
        )

        print(
            f"Multi-turn evaluation completed: {result.get('evaluated_turns', 0)} turns evaluated"
        )
        print(f"missing turn plans: {result.get('missing_turn_plans', 0)}")
        print(f"per-turn score files: {detail_count}")
        print(f"summary: {summary_path}")
        print(f"full debug summary: {full_summary_path}")
        metrics = result.get("metrics") or {}
        print(f"feasibility score: {metrics.get('feasibility_score', 0.0):.2%}")
        print(f"requirement score: {metrics.get('requirement_score', 0.0):.2%}")
        raise SystemExit(0)

    result = evaluate_plans(
        result_dir=args.plans_dir.parent,
        test_data_path=args.test_data,
        database_dir=args.database_dir,
        language=args.language,
        output_dir=args.output_dir,
        workers=args.workers,
    )

    print(f"Evaluation completed: {result['success']}/{result['total']} succeeded")
    metrics = result.get("metrics") or {}
    print(f"feasibility score: {metrics.get('feasibility_score', 0.0):.2%}")
    print(f"requirement score: {metrics.get('requirement_score', 0.0):.2%}")
