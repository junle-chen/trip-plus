"""Command-line configuration and path setup for the runner."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


def _model_result_slug(model_name: str) -> str:
    first_model = str(model_name or "model").split()[0]
    first_model = re.sub(r"-?vllm$", "", first_model)
    slug = re.sub(r"[^A-Za-z0-9._]+", "_", first_model).strip("_") or "model"
    return re.sub(r"([0-9]+)b($|_)", r"\1B\2", slug)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Trip-Plus agent inference, conversion, and evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model configuration name from models_config.json",
    )
    parser.add_argument(
        "--inference-model",
        type=str,
        default=None,
        help="Model for inference stage (default: --model)",
    )
    parser.add_argument(
        "--conversion-model",
        type=str,
        default=None,
        help="Model for conversion stage (default: inference model)",
    )
    parser.add_argument(
        "--evaluation-model",
        type=str,
        default=None,
        help="Model label for evaluation stage (default: inference model)",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test data JSON file. Defaults to query/query_en/multiturn/query.json",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of concurrent workers (default: 10)",
    )
    parser.add_argument(
        "--local-vllm-worker-cap",
        type=int,
        default=4,
        help="Max inference/conversion workers for local vLLM models (default: 4; 0 disables cap)",
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=150,
        help="Maximum LLM calls per sample (default: 150)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root. Defaults to result/{model_slug}/{model}_en_{timestamp}",
    )
    parser.add_argument(
        "--database-dir",
        type=str,
        default=None,
        help="Database root. Supports database, database/en, or database/sample/en",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default="inference",
        choices=["inference", "conversion", "evaluation"],
        help="Which step to start from (default: inference = run all steps)",
    )
    parser.add_argument(
        "--rerun-ids",
        type=str,
        default=None,
        help='Comma-separated list of IDs to rerun (e.g., "0,5,10" or "0-10,15,20-25")',
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    args.language = "en"
    return args


def resolve_stage_models(args):
    """Resolve stage-specific model names from the CLI defaults."""
    args.inference_model = args.inference_model or args.model
    args.conversion_model = args.conversion_model or args.inference_model
    args.evaluation_model = args.evaluation_model or args.inference_model
    args.model = args.inference_model
    return args


def effective_workers_for_model(
    requested_workers: int,
    model_name: str,
    stage_name: str = "",
    local_vllm_worker_cap: int = 4,
) -> int:
    """Cap local vLLM concurrency to avoid overloading one endpoint."""
    workers = max(1, int(requested_workers or 1))
    if "vllm" not in str(model_name or "").lower():
        return workers

    cap = int(local_vllm_worker_cap or 0)
    if cap <= 0:
        return workers

    capped_workers = min(workers, cap)
    if capped_workers < workers:
        stage_prefix = f"{stage_name}: " if stage_name else ""
        print(
            f"  ⚠️  {stage_prefix}capping workers from {workers} to {capped_workers} "
            f"for local vLLM model '{model_name}'"
        )
    return capped_workers


def setup_paths(args):
    """Resolve input, output, database, and tool-schema paths."""
    base_dir = Path(__file__).resolve().parents[1]

    if args.test_data is None:
        args.test_data = (
            base_dir
            / "query"
            / f"query_{args.language}"
            / "multiturn"
            / "query.json"
        )
    else:
        candidate = Path(args.test_data)
        args.test_data = candidate if candidate.is_absolute() else base_dir / candidate

    if not args.test_data.exists():
        raise FileNotFoundError(f"Test data file not found: {args.test_data}")

    with open(args.test_data, "r", encoding="utf-8") as f:
        args.test_samples = json.load(f)
    args.total_ids = len(args.test_samples)

    user_output_dir = getattr(args, "_user_output_dir", None)
    dir_name = f"{args.model}_{args.language}"
    if user_output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = (
            base_dir
            / "result"
            / _model_result_slug(args.model)
            / f"{dir_name}_{timestamp}"
        )
    else:
        args.output_dir = Path(user_output_dir) / dir_name

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("trajectories", "reports", "converted_plans", "evaluation"):
        (args.output_dir / subdir).mkdir(exist_ok=True)

    args.database_dir = _resolve_database_dir(
        base_dir, getattr(args, "_user_database_dir", None), args.language
    )
    args.tool_schema_path = base_dir / "tools" / f"tool_schema_{args.language}.json"
    return args


def _resolve_database_dir(base_dir: Path, user_database_dir, language: str) -> Path:
    if user_database_dir is None:
        release_db = base_dir / "database"
        release_city_db = release_db / language
        default_sample_db = base_dir / "database" / f"database_{language}"
        default_city_db = base_dir / "database" / "database_by_city" / language
        if (release_city_db / "city_index.json").exists():
            return release_db
        return default_sample_db if default_sample_db.exists() else default_city_db

    candidate = Path(user_database_dir)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    if (candidate / "city_index.json").exists():
        return candidate
    if (candidate / language / "city_index.json").exists():
        return candidate / language
    return candidate


def print_config(args):
    """Print the resolved pipeline configuration."""
    print("=" * 80)
    print("Trip-Plus Integrated Runner")
    print("=" * 80)
    print(f"Default model:      {args.model}")
    print(f"Inference model:    {args.inference_model}")
    print(f"Conversion model:   {args.conversion_model}")
    print(f"Evaluation model:   {args.evaluation_model}")
    print(
        "LLM user simulator: Standalone only (python -m simulation.run_user_simulation)"
    )
    print(f"Language:           {args.language}")
    print(f"Workers:            {args.workers}")
    print(f"Local vLLM cap:     {getattr(args, 'local_vllm_worker_cap', 4)}")
    print(f"Max LLM calls:      {args.max_llm_calls}")
    print(f"Test data:          {args.test_data}")
    print(f"Output directory:   {args.output_dir}")
    print(f"Database directory: {args.database_dir}")
    print(f"Tool schema:        {args.tool_schema_path}")

    steps = {
        "inference": ["1. Inference", "2. Conversion", "3. Evaluation"],
        "conversion": ["2. Conversion", "3. Evaluation"],
        "evaluation": ["3. Evaluation"],
    }[args.start_from]
    print(f"Pipeline steps:     {' → '.join(steps)}")
    print(f"Start from:         {args.start_from.capitalize()}")
    print("=" * 80)
    print()
