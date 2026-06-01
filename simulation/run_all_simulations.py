#!/usr/bin/env python3
"""Run all default simulator judges and aggregate their median score."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from evaluation.multiturn.ground_truth import load_query_records
from simulation.aggregate_median import DEFAULT_JUDGES, aggregate_median
from simulation.batch_runner import (
    REPO_ROOT,
    REUSE_SIMULATIONS_ENV,
    check_simulator_config,
    custom_runs,
    load_dotenv,
    relative_path,
    resolve_repo_path,
    run_user_simulations_only,
    target_id_filter,
    validate_result_dir,
    write_json,
)
from simulation.models import default_output_root, normalize_simulator_model, slug


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all simulator judges for completed Trip-Plus results."
    )
    parser.add_argument(
        "--result-dir",
        action="append",
        required=True,
        type=Path,
        help="Model result dir, evaluation dir, or converted_plans dir. Can repeat.",
    )
    parser.add_argument(
        "--query-file", type=Path, default=Path("query/query_en/multiturn/query.json")
    )
    parser.add_argument("--database-dir", type=Path, default=Path("database/sample/en"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--judge",
        action="append",
        default=None,
        help="Simulator judge to run. Can repeat. Defaults to the four median judges.",
    )
    parser.add_argument(
        "--target-id",
        action="append",
        default=None,
        help="Optional query id to evaluate. Can repeat.",
    )
    parser.add_argument("--min-judges", type=int, default=3)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--require-user-simulator", action="store_true")
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Run judge artifacts only; do not write the median summary.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    load_dotenv(REPO_ROOT / ".env")

    query_file = resolve_repo_path(args.query_file)
    database_dir = resolve_repo_path(args.database_dir)
    if not query_file.exists():
        raise FileNotFoundError(f"Missing query file: {query_file}")
    if not database_dir.exists():
        raise FileNotFoundError(f"Missing database dir: {database_dir}")

    runs = custom_runs(args.result_dir)
    for name, result_dir in runs:
        validate_result_dir(name, result_dir)

    judges = [normalize_simulator_model(judge) for judge in (args.judge or DEFAULT_JUDGES)]
    judge_configs = {judge: check_simulator_config(judge) for judge in judges}
    records = target_id_filter(load_query_records(query_file), args.target_id)

    previous_reuse = os.environ.get(REUSE_SIMULATIONS_ENV)
    os.environ[REUSE_SIMULATIONS_ENV] = "1" if args.reuse_existing else "0"

    try:
        for display_name, result_dir in runs:
            run_slug = slug(display_name)
            median_output = (
                Path("simulation")
                / "median"
                / run_slug
                / "user_simulation_median_summary.json"
            )
            manifest: dict[str, Any] = {
                "mode": "all_user_simulations",
                "source_result_dir": relative_path(result_dir),
                "query_file": relative_path(query_file),
                "database_dir": relative_path(database_dir),
                "target_ids": args.target_id or [],
                "judges": judges,
                "runs": [],
                "median_output": relative_path(median_output),
            }

            print("All-judge user simulation")
            print(f"Source: {result_dir}")
            print(f"Plans: {len(records)}")
            print(f"Judges: {', '.join(judges)}")

            for index, judge in enumerate(judges, start=1):
                output_dir = resolve_repo_path(default_output_root(judge) / run_slug)
                config = judge_configs[judge]
                print("")
                print(f"[{index}/{len(judges)}] {judge} via {config.get('base_url')}")
                print(f"  output: {output_dir}")

                started = time.time()
                summary = run_user_simulations_only(
                    display_name=display_name,
                    result_dir=result_dir,
                    records=records,
                    simulator_model=judge,
                    output_dir=output_dir,
                    workers=args.workers,
                    require_user_simulator=args.require_user_simulator,
                    reuse_existing=args.reuse_existing,
                )
                metrics = {
                    "mean_score": (summary.get("scores") or {}).get("mean_score"),
                    "mean_score_1_5": (summary.get("scores") or {}).get(
                        "mean_score_1_5"
                    ),
                    "counts": summary.get("counts"),
                    "elapsed_seconds": round(time.time() - started, 2),
                    "output_dir": relative_path(output_dir),
                }
                manifest["runs"].append({"judge": judge, **metrics})
                write_json(
                    median_output.parent / "run_all_simulations_manifest.json",
                    manifest,
                )

            if not args.skip_aggregate:
                median_summary = aggregate_median(
                    result_dir=result_dir,
                    run_slug=run_slug,
                    judges=judges,
                    output=median_output,
                    min_judges=args.min_judges,
                )
                manifest["median"] = {
                    "output": relative_path(median_output),
                    "mean_median_score": median_summary.get("mean_median_score"),
                    "mean_median_score_1_5": median_summary.get(
                        "mean_median_score_1_5"
                    ),
                    "plan_count_included_in_mean": median_summary.get(
                        "plan_count_included_in_mean"
                    ),
                }
                write_json(
                    median_output.parent / "run_all_simulations_manifest.json",
                    manifest,
                )
                print("")
                print(f"Median summary: {median_output}")
                print(f"Mean median score: {median_summary.get('mean_median_score')}")
    finally:
        if previous_reuse is None:
            os.environ.pop(REUSE_SIMULATIONS_ENV, None)
        else:
            os.environ[REUSE_SIMULATIONS_ENV] = previous_reuse


if __name__ == "__main__":
    main()
