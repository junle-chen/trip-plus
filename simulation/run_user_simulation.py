#!/usr/bin/env python3
"""Run traveler experience simulation on completed model runs."""

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
    discover_runs,
    display_metrics,
    load_dotenv,
    load_json,
    parse_model_filter,
    relative_path,
    resolve_repo_path,
    run_user_simulations_only,
    summary_path,
    target_id_filter,
    validate_result_dir,
    write_json,
)
from simulation.models import (
    DEFAULT_SIMULATOR_MODEL,
    default_output_root,
    normalize_simulator_model,
    slug,
)


LANGUAGE = "en"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OpenAI-compatible LLM user simulation for completed Trip-Plus runs."
    )
    parser.add_argument(
        "--all-judges",
        action="store_true",
        help="Run all median judges, then aggregate the median score.",
    )
    parser.add_argument(
        "--simulator-model",
        default=DEFAULT_SIMULATOR_MODEL,
        help=(
            "Model config name or alias. Supported aliases include "
            "gpt/gpt-nano, claude/haiku, gemini/gemini-flash, and qwen. "
            "The released config includes qwen; add local OpenAI-compatible configs for the other judges."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to result/user_simulation/<simulator-model>.",
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
        help="Judge model to run with --all-judges. Can repeat. Defaults to the four median judges.",
    )
    parser.add_argument("--min-judges", type=int, default=3)
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="With --all-judges, run judge artifacts only; do not write the median summary.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional source model display names, comma-separated or spaced.",
    )
    parser.add_argument(
        "--result-dir",
        action="append",
        type=Path,
        default=[],
        help="Model-language result dir, evaluation dir, or converted_plans dir. Can repeat.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("result"),
        help="Search root used when --result-dir is omitted.",
    )
    parser.add_argument(
        "--run-pattern",
        default="query_en_multiturn",
        help="Optional substring for auto-discovered result paths. Use empty string to disable.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="When auto-discovering, run every matching result dir instead of the latest one per source model.",
    )
    parser.add_argument(
        "--target-id",
        action="append",
        default=None,
        help="Optional query id to evaluate. Can repeat.",
    )
    parser.add_argument("--require-user-simulator", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse matching artifacts already present under output-root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the run plan only.",
    )
    return parser


def _run_single_judge(
    *,
    args: argparse.Namespace,
    runs: list[tuple[str, Path]],
    records: list[dict[str, Any]],
    query_file: Path,
    database_dir: Path,
) -> None:
    args.simulator_model = normalize_simulator_model(args.simulator_model)
    simulator_config = check_simulator_config(args.simulator_model)
    output_root = resolve_repo_path(
        args.output_root or default_output_root(args.simulator_model)
    )

    manifest: dict[str, Any] = {
        "mode": "user_simulation_only",
        "simulator_model": args.simulator_model,
        "simulator_model_name": simulator_config.get("model_name"),
        "base_url": simulator_config.get("base_url"),
        "api_key_env": simulator_config.get("api_key_env"),
        "query_file": relative_path(query_file),
        "database_dir": relative_path(database_dir),
        "output_root": relative_path(output_root),
        "reuse_existing": args.reuse_existing,
        "runs": [],
    }

    print("OpenAI-compatible user simulation")
    print(f"Simulator: {args.simulator_model} via {simulator_config.get('base_url')}")
    print(f"Output root: {output_root}")
    print("Mode: user_simulation_only")
    print(f"Query records: {len(records)}")
    print(f"Runs: {len(runs)}")

    for index, (display_name, result_dir) in enumerate(runs, start=1):
        run_slug = slug(display_name)
        output_dir = output_root / run_slug
        simulation_dir = output_dir / "user_simulations"
        summary_file = output_dir / "user_simulation_summary.json"
        source_summary_file = summary_path(result_dir)
        source_summary = (
            load_json(source_summary_file) if source_summary_file else None
        )

        run_record: dict[str, Any] = {
            "display_name": display_name,
            "source_result_dir": relative_path(result_dir),
            "source_summary": relative_path(source_summary_file)
            if source_summary_file
            else None,
            "output_summary": relative_path(summary_file),
            "user_simulation_dir": relative_path(simulation_dir),
            "source_metrics": display_metrics(source_summary)
            if isinstance(source_summary, dict)
            else None,
        }

        print("")
        print(f"[{index}/{len(runs)}] {display_name}")
        print(f"  source: {result_dir}")
        print(f"  output: {output_dir}")

        if args.dry_run:
            run_record["status"] = "dry_run"
            manifest["runs"].append(run_record)
            continue

        started = time.time()
        summary = run_user_simulations_only(
            display_name=display_name,
            result_dir=result_dir,
            records=records,
            simulator_model=args.simulator_model,
            output_dir=output_dir,
            workers=args.workers,
            require_user_simulator=args.require_user_simulator,
            reuse_existing=args.reuse_existing,
        )
        elapsed_seconds = round(time.time() - started, 2)

        run_record["status"] = "ok"
        run_record["elapsed_seconds"] = elapsed_seconds
        run_record["new_metrics"] = {
            "mean_score": (summary.get("scores") or {}).get("mean_score"),
            "mean_score_1_5": (summary.get("scores") or {}).get("mean_score_1_5"),
            "ok": (summary.get("counts") or {}).get("ok"),
            "failed": (summary.get("counts") or {}).get("failed"),
            "skipped": (summary.get("counts") or {}).get("skipped"),
            "runtime_totals": summary.get("runtime_totals"),
        }
        run_record["simulation_artifact_count"] = len(
            list(simulation_dir.glob("*_user_simulation.json"))
        )
        manifest["runs"].append(run_record)

        metrics = run_record["new_metrics"]
        print(
            "  done: "
            f"ok={metrics.get('ok')} "
            f"failed={metrics.get('failed')} "
            f"skipped={metrics.get('skipped')} "
            f"mean_score={metrics.get('mean_score')} "
            f"elapsed={elapsed_seconds}s"
        )
        write_json(output_root / "manifest.json", manifest)

    print("")
    if args.dry_run:
        print("Dry run finished; manifest was not written.")
    else:
        write_json(output_root / "manifest.json", manifest)
        print(f"Manifest written to {output_root / 'manifest.json'}")


def _run_all_judges(
    *,
    args: argparse.Namespace,
    runs: list[tuple[str, Path]],
    records: list[dict[str, Any]],
    query_file: Path,
    database_dir: Path,
) -> None:
    if args.output_root:
        raise SystemExit("--output-root is only supported for single-judge runs.")

    judges = [
        normalize_simulator_model(judge)
        for judge in (args.judge or DEFAULT_JUDGES)
    ]
    judge_configs = {judge: check_simulator_config(judge) for judge in judges}

    for display_name, result_dir in runs:
        run_slug = slug(display_name)
        median_output = (
            Path("result")
            / "user_simulation"
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

            if args.dry_run:
                manifest["runs"].append(
                    {
                        "judge": judge,
                        "status": "dry_run",
                        "output_dir": relative_path(output_dir),
                    }
                )
                continue

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
                "mean_score_1_5": (summary.get("scores") or {}).get("mean_score_1_5"),
                "counts": summary.get("counts"),
                "elapsed_seconds": round(time.time() - started, 2),
                "output_dir": relative_path(output_dir),
            }
            manifest["runs"].append({"judge": judge, **metrics})
            write_json(median_output.parent / "all_judges_manifest.json", manifest)

        if args.dry_run:
            print("")
            print("Dry run finished; manifest was not written.")
            continue

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
                "mean_median_score_1_5": median_summary.get("mean_median_score_1_5"),
                "plan_count_included_in_mean": median_summary.get(
                    "plan_count_included_in_mean"
                ),
            }
            write_json(median_output.parent / "all_judges_manifest.json", manifest)
            print("")
            print(f"Median summary: {median_output}")
            print(f"Mean median score: {median_summary.get('mean_median_score')}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    load_dotenv(REPO_ROOT / ".env")
    query_file = resolve_repo_path(args.query_file)
    database_dir = resolve_repo_path(args.database_dir)
    if not query_file.exists():
        raise FileNotFoundError(f"Missing query file: {query_file}")
    if not database_dir.exists():
        raise FileNotFoundError(f"Missing database dir: {database_dir}")

    model_filter = parse_model_filter(args.models)
    if args.result_dir:
        runs = custom_runs(args.result_dir)
    else:
        runs = discover_runs(
            result_root=resolve_repo_path(args.result_root),
            language=LANGUAGE,
            run_pattern=(args.run_pattern or None),
            model_filter=model_filter,
            latest_per_model=not args.all_runs,
        )
    if not runs:
        raise SystemExit(
            "No runs selected. Pass --result-dir, or adjust --result-root/--run-pattern/--models."
        )
    for name, result_dir in runs:
        validate_result_dir(name, result_dir)

    previous_reuse = os.environ.get(REUSE_SIMULATIONS_ENV)
    os.environ[REUSE_SIMULATIONS_ENV] = "1" if args.reuse_existing else "0"
    records = target_id_filter(load_query_records(query_file), args.target_id)

    try:
        if args.all_judges:
            _run_all_judges(
                args=args,
                runs=runs,
                records=records,
                query_file=query_file,
                database_dir=database_dir,
            )
        else:
            if args.judge:
                raise SystemExit("--judge is only supported with --all-judges.")
            if args.skip_aggregate:
                raise SystemExit("--skip-aggregate is only supported with --all-judges.")
            _run_single_judge(
                args=args,
                runs=runs,
                records=records,
                query_file=query_file,
                database_dir=database_dir,
            )
    finally:
        if previous_reuse is None:
            os.environ.pop(REUSE_SIMULATIONS_ENV, None)
        else:
            os.environ[REUSE_SIMULATIONS_ENV] = previous_reuse


if __name__ == "__main__":
    main()
