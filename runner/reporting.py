"""Pipeline score-file and console-summary helpers."""

from __future__ import annotations

from pathlib import Path

from evaluation.output import write_score_payload


def write_multiturn_evaluation_details(results: dict, output_dir: Path) -> int:
    """Write one score file per multi-turn query turn for auditability."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for record in results.get("results", []) or []:
        sample_id = str(record.get("sample_id") or record.get("id") or "").strip()
        for turn in record.get("turn_results", []) or []:
            turn_id = str(turn.get("turn_id"))
            evaluation_result = turn.get("evaluation_result")
            payload = evaluation_result or {
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


def print_final_summary(args, inference_results, conversion_results, eval_results):
    """Print final summary of all executed pipeline steps."""
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    if inference_results:
        print(
            f"Inference:  {inference_results['success']}/{inference_results['total']} succeeded"
        )

    if conversion_results:
        print(
            f"Conversion: {conversion_results['converted']}/{conversion_results['total']} converted"
        )

    if eval_results:
        if "average_score" in eval_results:
            metrics = eval_results.get("metrics") or {}
            if metrics:
                llm_user_sim_text = (
                    f"{metrics['llm_user_simulation_score']:.2f}"
                    if metrics.get("llm_user_simulation_score") is not None
                    else "n/a"
                )
                print(
                    "Evaluation: "
                    f"feasibility={metrics.get('feasibility_score', 0.0):.2f}, "
                    f"requirement={metrics.get('requirement_score', 0.0):.2f}, "
                    f"llm_user_simulation={llm_user_sim_text}"
                )
            else:
                print(
                    f"Evaluation: requirement score = {eval_results.get('average_score', 0.0):.2f}"
                )
        else:
            _print_multiturn_summary(eval_results)

    print(f"\nResults saved to: {args.output_dir}")
    print("=" * 80)


def _print_multiturn_summary(eval_results):
    metrics = eval_results.get("metrics", {})
    print(f"Evaluation: Evaluated turns = {eval_results.get('evaluated_turns', 0)}")
    print(f"            Missing turn plans = {eval_results.get('missing_turn_plans', 0)}")
    for name, label in (
        ("response_mode_accuracy", "Response mode accuracy"),
        ("fulfillment_score", "Fulfillment score"),
        ("preserve_score", "Preserve score"),
    ):
        if metrics.get(name) is not None:
            print(f"            {label} = {metrics.get(name):.2f}")
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
            print(f"            {label} = {metrics.get(name):.2f}")
