"""Integrated Trip-Plus runner for inference, conversion, and evaluation.

Example:
    python run.py --model qwen3.6-27b-vllm --workers 2
"""

import sys
import time
from pathlib import Path

# Allow direct `python run.py` execution from the repository root.
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.conversion import convert_reports
from evaluation.single_turn import evaluate_plans
from evaluation.output import write_multiturn_summary_files
from evaluation.multiturn.runner import evaluate_multiturn_plans
from agent.batch_runner import run_agent_inference
from runner.config import (
    effective_workers_for_model,
    parse_args,
    print_config,
    resolve_stage_models,
    setup_paths,
)
from runner.ids import (
    conversion_output_ids_for_rerun_ids,
    detect_missing_converted_outputs,
    detect_missing_report_parent_ids,
    expected_converted_output_ids,
    is_multiturn_test_data,
    load_test_sample_ids,
    parse_id_list,
)
from runner.reporting import print_final_summary, write_multiturn_evaluation_details


def run_step_inference(args):
    """Step 1: Run agent inference to generate trajectories"""
    print("\n" + "=" * 80)
    print("STEP 1: Agent Inference")
    print("=" * 80)

    # Auto-detect missing reports if not explicitly specifying rerun_ids
    rerun_ids = None
    if args.rerun_ids:
        # User explicitly specified IDs to rerun
        rerun_ids = parse_id_list(args.rerun_ids)
        print(f"  🔄 User-specified IDs to rerun: {rerun_ids}")
        print(f"  📝 Total IDs: {len(rerun_ids)}")
    else:
        # Auto-detect missing reports
        reports_dir = args.output_dir / "reports"
        expected_ids = load_test_sample_ids(args.test_data)
        missing_ids = detect_missing_report_parent_ids(reports_dir, args.test_data)

        if missing_ids:
            rerun_ids = missing_ids
            print(f"  🔍 Auto-detected missing reports")
            print(
                f"  📝 Missing IDs ({len(missing_ids)}): {missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}"
            )
            print(f"  🔄 Will regenerate reports for these IDs")
        else:
            print(f"  ✅ All reports already exist")
            print(f"  ⏭️  Skipping inference step")
            return True, {
                "total": len(expected_ids),
                "success": len(expected_ids),
                "failed": 0,
                "cached": len(expected_ids),
                "processed": 0,
            }

    start_time = time.time()

    try:
        # Dynamically select the appropriate agent
        results = run_agent_inference(
            model=args.inference_model,
            language=args.language,
            test_data_path=args.test_data,
            database_dir=args.database_dir,
            tool_schema_path=args.tool_schema_path,
            output_dir=args.output_dir,
            workers=effective_workers_for_model(
                args.workers,
                args.inference_model,
                "Inference",
                getattr(args, "local_vllm_worker_cap", 4),
            ),
            max_llm_calls=args.max_llm_calls,
            verbose=args.verbose,
            rerun_ids=rerun_ids,  # Pass rerun_ids parameter
        )

        elapsed = time.time() - start_time

        print(f"\n✅ Inference completed in {elapsed:.2f}s")
        print(f"   Total samples: {results['total']}")
        print(f"   Success: {results['success']}")
        print(f"   Failed: {results['failed']}")
        if "cached" in results and results.get("cached", 0) > 0:
            print(f"   Cached: {results['cached']}")
            print(f"   Newly processed: {results.get('processed', 0)}")
        if int(results.get("failed") or 0) > 0:
            print(
                "   ⚠️  Inference output is incomplete; stopping before conversion/evaluation."
            )
            print("      Rerun the missing reports after fixing the inference errors.")
            return False, results

        return True, results

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Inference failed after {elapsed:.2f}s: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return False, None


def run_step_conversion(args):
    """Step 2: Convert reports to standardized plan format"""
    print("\n" + "=" * 80)
    print("STEP 2: Plan Conversion")
    print("=" * 80)

    reconvert_ids = parse_id_list(args.rerun_ids) if args.rerun_ids else None

    if reconvert_ids:
        for sample_id in conversion_output_ids_for_rerun_ids(
            args.test_data, reconvert_ids
        ):
            converted_file = (
                args.output_dir / "converted_plans" / f"id_{sample_id}_converted.json"
            )
            if converted_file.exists():
                converted_file.unlink()

    # Auto-detect missing converted plans
    converted_plans_dir = args.output_dir / "converted_plans"
    expected_ids = expected_converted_output_ids(args.test_data)
    missing_ids = detect_missing_converted_outputs(converted_plans_dir, expected_ids)

    if missing_ids:
        print(f"  🔍 Auto-detected missing converted plans")
        print(
            f"  📝 Missing IDs ({len(missing_ids)}): {missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}"
        )
        print(f"  🔄 Will convert reports for these IDs")
    else:
        print(f"  ✅ All converted plans already exist")
        print(f"  ⏭️  Skipping conversion step")
        return True, {
            "total": args.total_ids,
            "converted": 0,
            "skipped": args.total_ids,
        }

    start_time = time.time()

    try:
        # Always use skip_existing=True to only convert missing files
        results = convert_reports(
            result_dir=args.output_dir,
            language=args.language,
            conversion_model=args.conversion_model,
            workers=effective_workers_for_model(
                args.workers,
                args.conversion_model,
                "Conversion",
                getattr(args, "local_vllm_worker_cap", 4),
            ),
            skip_existing=True,
            database_dir=getattr(args, "database_dir", None),
            query_file=getattr(args, "test_data", None),
        )

        elapsed = time.time() - start_time

        print(f"\n✅ Conversion completed in {elapsed:.2f}s")
        print(f"   Total files: {results['total']}")
        print(f"   Converted: {results['converted']}")
        print(f"   Skipped: {results['skipped']}")

        return True, results

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Conversion failed after {elapsed:.2f}s: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return False, None


def run_step_evaluation(args):
    """Step 3: Evaluate converted plans"""
    print("\n" + "=" * 80)
    print("STEP 3: Plan Evaluation")
    print("=" * 80)
    print(f"  Model label: {args.evaluation_model}")
    print(f"  Language: {args.language}")
    print(f"  Database directory: {args.database_dir}")
    print(f"  Test data: {args.test_data}")

    target_ids = None
    if args.rerun_ids:
        target_ids = [str(i) for i in parse_id_list(args.rerun_ids)]
        print(
            f"  🎯 Evaluating user-specified rerun-ids subset: {len(target_ids)} samples"
        )
    else:
        print(
            f"  📊 Note: Will evaluate all reachable plans from the provided test data"
        )
    evaluation_workers = max(1, int(args.workers or 1))
    print(f"  Workers: {evaluation_workers}")
    print()

    start_time = time.time()

    try:
        if is_multiturn_test_data(args.test_data):
            output_dir = args.output_dir / "evaluation"
            output_dir.mkdir(parents=True, exist_ok=True)
            results = evaluate_multiturn_plans(
                query_file=args.test_data,
                plans_dir=args.output_dir / "converted_plans",
                database_root=args.database_dir,
                language=args.language,
                workers=evaluation_workers,
                target_ids=target_ids,
            )
            detail_count = write_multiturn_evaluation_details(results, output_dir)
            summary_path, full_summary_path = write_multiturn_summary_files(
                results, output_dir
            )

            elapsed = time.time() - start_time
            metrics = results.get("metrics", {})
            evaluated_turns = int(results.get("evaluated_turns") or 0)
            missing_turns = int(results.get("missing_turn_plans") or 0)

            print(f"\n✅ Multi-turn evaluation completed in {elapsed:.2f}s")
            print(f"   Evaluated turns: {evaluated_turns}")
            print(f"   Missing turn plans: {missing_turns}")
            print(f"   Per-turn score files: {detail_count}")
            print(f"   Summary: {summary_path}")
            print(f"   Full debug summary: {full_summary_path}")
            print("   Multi-turn metrics:")
            for name, label in (
                ("response_mode_accuracy", "Response mode accuracy"),
                ("fulfillment_score", "Fulfillment score"),
                ("preserve_score", "Preserve score"),
            ):
                if metrics.get(name) is not None:
                    print(f"     {label}: {metrics.get(name):.2f}")
            print("   Core plan metrics:")
            metric_labels = (
                ("feasibility_score", "Plan feasibility"),
                ("hard_constraint_score", "Hard constraint score"),
                ("soft_preference_score", "Soft preference score"),
                ("requirement_score", "Requirement score"),
                ("strict_hard_constraint", "Strict hard pass rate"),
                ("strict_soft_preference", "Strict soft pass rate"),
                ("llm_user_simulation_score", "LLM user simulation score"),
            )
            for name, label in metric_labels:
                if metrics.get(name) is not None:
                    print(f"     {label}: {metrics.get(name):.2f}")

            if evaluated_turns == 0:
                print(
                    "   ⚠️  No converted turn plans found. Evaluation cannot produce summary metrics."
                )
                return False, results
            return True, results

        results = evaluate_plans(
            result_dir=args.output_dir,
            test_data_path=args.test_data,
            database_dir=args.database_dir,
            language=args.language,
            workers=evaluation_workers,
            target_ids=target_ids,
        )

        elapsed = time.time() - start_time

        print(f"\n✅ Evaluation completed in {elapsed:.2f}s")
        total_plans = results.get("total", 0)
        print(f"   Total plans: {total_plans}")

        # If no converted plans are available, fail with a clear message instead of KeyError.
        if total_plans == 0:
            print(
                "   ⚠️  No converted plans found. Evaluation cannot produce summary metrics."
            )
            return False, results

        metrics = results.get("metrics") or {}
        if metrics:
            print("   Core metrics:")
            for name, label in (
                ("feasibility_score", "Plan feasibility"),
                ("hard_constraint_score", "Hard constraint score"),
                ("soft_preference_score", "Soft preference score"),
                ("requirement_score", "Requirement score"),
                ("strict_hard_constraint", "Strict hard pass rate"),
                ("strict_soft_preference", "Strict soft pass rate"),
                ("llm_user_simulation_score", "LLM user simulation score"),
            ):
                if metrics.get(name) is not None:
                    print(f"     {label}: {metrics.get(name):.2f}")

        return True, results

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Evaluation failed after {elapsed:.2f}s: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return False, None


def run_single_language(args, language):
    """Run pipeline for a single language"""
    # Update args with specific language
    args.language = language
    args = setup_paths(args)

    print_config(args)

    lang_start_time = time.time()

    inference_results = None
    conversion_results = None
    eval_results = None
    # Step 1: Inference
    if args.start_from == "inference":
        success, inference_results = run_step_inference(args)
        if not success:
            print("\n⚠️  Inference failed, skipping subsequent steps")
            return False, None, None, None

    # Step 2: Conversion
    if args.start_from in ["inference", "conversion"]:
        success, conversion_results = run_step_conversion(args)
        if not success:
            print("\n⚠️  Conversion failed, skipping evaluation")
            return False, inference_results, None, None

    # Step 3: Evaluation
    if args.start_from in ["inference", "conversion", "evaluation"]:
        success, eval_results = run_step_evaluation(args)
        if not success:
            print("\n⚠️  Evaluation failed")
            return False, inference_results, conversion_results, None

    # Print summary for this language
    lang_elapsed = time.time() - lang_start_time
    print_final_summary(args, inference_results, conversion_results, eval_results)
    print(
        f"\n✅ Model '{args.inference_model}' | Language '{language}' completed in {lang_elapsed:.2f}s ({lang_elapsed / 60:.1f} minutes)"
    )

    return True, inference_results, conversion_results, eval_results


def main():
    """Main execution function"""
    args = parse_args()
    args = resolve_stage_models(args)

    # Save user-specified output_dir (if any) before it gets modified
    # This allows language-specific directories to be generated for multi-language runs
    args._user_output_dir = args.output_dir
    args._user_database_dir = args.database_dir

    overall_start_time = time.time()

    # English-only release.
    languages = ["en"]

    # Run for each language
    all_success = True
    for idx, lang in enumerate(languages):
        if len(languages) > 1:
            print("\n" + "=" * 80)
            print(f"LANGUAGE {idx + 1}/{len(languages)}: {lang.upper()}")
            print("=" * 80)
            print()

        success, inf_res, conv_res, eval_res = run_single_language(args, lang)

        if not success:
            all_success = False
            print(f"\n❌ Pipeline failed for language '{lang}'")
            if len(languages) > 1 and idx < len(languages) - 1:
                print(f"Continuing with next language...\n")
                continue
            else:
                sys.exit(1)

    # Print overall summary
    overall_elapsed = time.time() - overall_start_time
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Languages run: {', '.join(languages)}")
    print(f"Total time: {overall_elapsed:.2f}s ({overall_elapsed / 60:.1f} minutes)")
    print("=" * 80)

    if all_success:
        print("\n✅ All pipelines completed successfully!")
    else:
        print("\n⚠️  Some pipelines failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
