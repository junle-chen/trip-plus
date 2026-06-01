"""Runner for multi-turn evaluation.

Each turn is evaluated along three axes: response mode, fulfillment of the
current turn request, and preservation of active prior constraints.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..single_turn import (
    _build_evaluation_result,
)
from ..feasibility.runner import eval_itinerary_feasibility
from ..hard import calculate_hard_score, eval_hard
from ..response_cases import build_unsat_result, evaluate_unsat_case
from ..runtime import attach_evaluation_metadata, load_runtime_stats_for_plan_file
from .files import find_turn_plan, resolve_multiturn_database_path
from .fulfillment import evaluate_turn_alignment
from .ground_truth import derive_turn_ground_truth, load_query_records
from .response_mode import (
    build_response_mode_result,
    _response_expectation_result,
)
from .summary import summarize_multiturn_results


def evaluate_multiturn_record(
    record: Dict[str, Any],
    plans_dir: Path,
    database_root: Path,
    language: str,
    query_file: Path,
    target_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    sample_id = str(record.get("id"))
    target_set = {str(target_id) for target_id in target_ids or []}
    evaluate_all_turns = not target_set or sample_id in target_set
    database_path = resolve_multiturn_database_path(
        record, database_root, language, query_file
    )
    turn_results: List[Dict[str, Any]] = []
    for turn in record.get("turns", []) or []:
        if not isinstance(turn, dict):
            continue
        turn_id = turn.get("turn_id")
        turn_key = f"{sample_id}_turn_{turn_id}"
        if not evaluate_all_turns and turn_key not in target_set:
            continue
        turn_gt = derive_turn_ground_truth(record, turn)
        meta = turn_gt["meta_info"]
        plan_path = find_turn_plan(
            plans_dir, sample_id, turn_id, record.get("base_query_id")
        )
        if plan_path is None:
            turn_results.append(
                {
                    "turn_id": turn_id,
                    "success": False,
                    "error": "missing_turn_plan",
                    "query": str(turn.get("utterance") or ""),
                    "ground_truth": turn_gt["ground_truth"],
                }
            )
            continue

        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        runtime_stats = load_runtime_stats_for_plan_file(plan_path)
        query_text = str(turn.get("utterance") or "")
        solution_status = str(meta.get("solution_status", "sat")).lower()
        response_expectation = str(meta.get("response_expectation") or "plan").lower()
        expects_non_itinerary = response_expectation in {
            "clarification",
            "conflict_resolution",
            "infeasible",
            "no_solution",
        }
        response_check = _response_expectation_result(plan, response_expectation)
        response_matches = bool(response_check.get("passed"))
        if not response_matches:
            evaluation_result = build_response_mode_result(
                sample_id=f"{sample_id}_turn_{turn_id}",
                plan=plan,
                response_expectation=response_expectation,
                evaluation_mode="response_expectation_mismatch",
                feasibility_score=0.0,
                strict_feasibility=0.0,
                requirement_score=0.0,
            )
            hard_result = {
                "score": 0.0,
                "constraints": {
                    "response_expectation": response_check,
                },
            }
            feasibility_results = {}
        elif solution_status == "unsat":
            unsat_eval = evaluate_unsat_case(plan, meta)
            evaluation_result = build_unsat_result(
                f"{sample_id}_turn_{turn_id}", unsat_eval
            )
            hard_result = unsat_eval.get("hard_constraint_dimension_score", {})
            feasibility_results: Dict[str, Tuple[bool, Optional[str]]] = {}
        elif expects_non_itinerary:
            evaluation_result = build_response_mode_result(
                sample_id=f"{sample_id}_turn_{turn_id}",
                plan=plan,
                response_expectation=response_expectation,
                evaluation_mode=response_expectation,
                feasibility_score=None,
                strict_feasibility=None,
                requirement_score=1.0,
            )
            hard_result = {
                "score": evaluation_result["scores"]["requirement_score"],
                "constraints": {
                    "response_expectation": evaluation_result[
                        "response_expectation_details"
                    ],
                },
            }
            feasibility_results = {}
        else:
            feasibility_results = eval_itinerary_feasibility(
                plan, meta, database_dir=database_path
            )
            hard = eval_hard(plan, meta)
            hard_result = calculate_hard_score(hard)
            evaluation_result = _build_evaluation_result(
                sample_id=f"{sample_id}_turn_{turn_id}",
                plan=plan,
                meta=meta,
                feasibility_results=feasibility_results,
                hard_result=hard_result,
                database_dir=database_path,
            )

        turn_alignment = evaluate_turn_alignment(
            plan, meta, hard_result, feasibility_results, database_dir=database_path
        )
        evaluation_result["turn_alignment"] = turn_alignment
        evaluation_result["ground_truth"] = turn_gt["ground_truth"]
        evaluation_result["plan_file"] = str(plan_path)
        attach_evaluation_metadata(
            evaluation_result,
            sample_record=record,
            query=query_text,
            plan_file=plan_path,
            runtime_stats=runtime_stats,
        )

        turn_results.append(
            {
                "turn_id": turn_id,
                "success": True,
                "query": query_text,
                "evaluation_result": evaluation_result,
            }
        )

    return {
        "sample_id": sample_id,
        "base_query_id": record.get("base_query_id"),
        "interaction_type": record.get("interaction_type"),
        "database_path": str(database_path) if database_path else None,
        "turn_results": turn_results,
    }


def evaluate_multiturn_plans(
    query_file: Path,
    plans_dir: Path,
    database_root: Path,
    language: str,
    workers: int = 10,
    target_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    records = load_query_records(query_file)
    if target_ids is not None:
        target_set = {str(sample_id) for sample_id in target_ids}
        filtered_records = []
        for record in records:
            sample_id = str(record.get("id"))
            if sample_id in target_set:
                filtered_records.append(record)
                continue
            turn_ids = {
                f"{sample_id}_turn_{turn.get('turn_id')}"
                for turn in record.get("turns", []) or []
                if isinstance(turn, dict)
            }
            if turn_ids & target_set:
                filtered_records.append(record)
        records = filtered_records
    workers = max(1, int(workers or 1))
    print(f"🚀 Using {workers} threads for multi-turn evaluation")

    def evaluate_record(record: Dict[str, Any]) -> Dict[str, Any]:
        return evaluate_multiturn_record(
            record=record,
            plans_dir=plans_dir,
            database_root=database_root,
            language=language,
            query_file=query_file,
            target_ids=target_ids,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        evaluated = list(executor.map(evaluate_record, records))
    return summarize_multiturn_results(evaluated)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate multi-turn travel plans with query-derived ground truth."
    )
    parser.add_argument("--query-file", required=True, type=Path)
    parser.add_argument("--plans-dir", required=True, type=Path)
    parser.add_argument("--database-dir", required=True, type=Path)
    parser.add_argument("--language", default="en", choices=["en"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    summary = evaluate_multiturn_plans(
        query_file=args.query_file,
        plans_dir=args.plans_dir,
        database_root=args.database_dir,
        language=args.language,
        workers=args.workers,
        target_ids=None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
