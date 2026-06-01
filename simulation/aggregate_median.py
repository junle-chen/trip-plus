#!/usr/bin/env python3
"""Aggregate per-plan median user-simulation scores across simulator judges."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.models import normalize_simulator_model, slug

DEFAULT_JUDGES = (
    "qwen3.6-27b-vllm",
    "gpt-5.4-nano",
    "claude-haiku-4-5-20251001",
    "gemini-3.1-flash-lite",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _score_from_artifact(path: Path) -> tuple[float | None, float | None]:
    payload = _load_json(path)
    recalculation = payload.get("score_recalculation") if isinstance(payload, dict) else {}
    if not isinstance(recalculation, dict):
        return None, None
    score = recalculation.get("recomputed_score")
    score_1_5 = recalculation.get("recomputed_score_1_5")
    return (
        float(score) if isinstance(score, (int, float)) else None,
        float(score_1_5) if isinstance(score_1_5, (int, float)) else None,
    )


def _sample_key(path: Path) -> str:
    stem = path.name.removesuffix("_user_simulation.json").removeprefix("id_")
    match = re.match(r"(.+)_turn_(\d+)$", stem)
    if not match:
        return stem
    return f"{match.group(1)}_turn_{match.group(2)}"


def _artifact_dirs_for_judge(judge: str, result_dir: Path, run_slug: str) -> list[Path]:
    normalized_judge = normalize_simulator_model(judge)
    if normalized_judge == "qwen3.6-27b-vllm":
        return [
            result_dir / "evaluation" / "user_simulations",
            Path("simulation") / normalized_judge / run_slug / "user_simulations",
        ]
    return [Path("simulation") / normalized_judge / run_slug / "user_simulations"]


def _collect_scores(judge: str, artifact_dir: Path) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    if not artifact_dir.exists():
        return scores
    for artifact in sorted(artifact_dir.glob("*_user_simulation.json")):
        score, score_1_5 = _score_from_artifact(artifact)
        if score is None:
            continue
        scores[_sample_key(artifact)] = {
            "judge": judge,
            "score": score,
            "score_1_5": score_1_5,
            "artifact_path": str(artifact),
        }
    return scores


def aggregate_median(
    *,
    result_dir: Path,
    run_slug: str,
    judges: list[str],
    output: Path,
    min_judges: int,
) -> dict[str, Any]:
    by_plan: dict[str, dict[str, Any]] = {}
    judge_counts: dict[str, int] = {}

    for judge in judges:
        judge_scores: dict[str, dict[str, Any]] = {}
        for artifact_dir in _artifact_dirs_for_judge(judge, result_dir, run_slug):
            judge_scores.update(_collect_scores(judge, artifact_dir))
        judge_counts[judge] = len(judge_scores)
        for sample_key, item in judge_scores.items():
            plan_entry = by_plan.setdefault(sample_key, {"sample_key": sample_key, "judge_scores": {}})
            plan_entry["judge_scores"][judge] = {
                "score": item["score"],
                "score_1_5": item["score_1_5"],
                "artifact_path": item["artifact_path"],
            }

    rows = []
    median_scores = []
    median_scores_1_5 = []
    for sample_key in sorted(by_plan):
        entry = by_plan[sample_key]
        scores = [
            value["score"]
            for value in entry["judge_scores"].values()
            if isinstance(value.get("score"), (int, float))
        ]
        scores_1_5 = [
            value["score_1_5"]
            for value in entry["judge_scores"].values()
            if isinstance(value.get("score_1_5"), (int, float))
        ]
        median_score = statistics.median(scores) if scores else None
        median_score_1_5 = statistics.median(scores_1_5) if scores_1_5 else None
        included_in_mean = len(scores) >= min_judges
        if included_in_mean and median_score is not None:
            median_scores.append(float(median_score))
        if included_in_mean and median_score_1_5 is not None:
            median_scores_1_5.append(float(median_score_1_5))
        rows.append({
            **entry,
            "judge_count": len(scores),
            "included_in_mean": included_in_mean,
            "median_score": median_score,
            "median_score_1_5": median_score_1_5,
        })

    summary = {
        "mode": "median_user_simulation",
        "source_result_dir": str(result_dir),
        "run_slug": run_slug,
        "judges": judges,
        "min_judges": min_judges,
        "judge_artifact_counts": judge_counts,
        "plan_count_with_any_score": len(rows),
        "plan_count_included_in_mean": sum(1 for row in rows if row["included_in_mean"]),
        "plan_count_with_all_judges": sum(1 for row in rows if row["judge_count"] == len(judges)),
        "mean_median_score": round(sum(median_scores) / len(median_scores), 6) if median_scores else None,
        "mean_median_score_1_5": (
            round(sum(median_scores_1_5) / len(median_scores_1_5), 6)
            if median_scores_1_5
            else None
        ),
        "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate median user-simulation scores across judges.")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--judge", action="append", default=None, help="Judge model name. Can repeat.")
    parser.add_argument(
        "--min-judges",
        type=int,
        default=3,
        help="Minimum successful judge scores required for a plan to contribute to the mean.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to simulation/median/<source-model>/user_simulation_median_summary.json.",
    )
    args = parser.parse_args()
    run_slug = slug(args.result_dir.name.removesuffix("_en"))
    output = args.output or Path("simulation") / "median" / run_slug / "user_simulation_median_summary.json"
    judges = [normalize_simulator_model(judge) for judge in (args.judge or list(DEFAULT_JUDGES))]

    summary = aggregate_median(
        result_dir=args.result_dir,
        run_slug=run_slug,
        judges=judges,
        output=output,
        min_judges=args.min_judges,
    )
    print(f"wrote {output}")
    print(f"judge counts: {summary['judge_artifact_counts']}")
    print(f"plans with any score: {summary['plan_count_with_any_score']}")
    print(f"plans included in mean: {summary['plan_count_included_in_mean']}")
    print(f"plans with all judges: {summary['plan_count_with_all_judges']}")
    print(f"mean median score: {summary['mean_median_score']}")


if __name__ == "__main__":
    main()
