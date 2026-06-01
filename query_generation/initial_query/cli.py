"""Command-line entry point for English single-turn query generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from query_generation.city_database import DEFAULT_CITY_DB_ROOT
from query_generation.common import BASE_DIR, load_env_file, write_json
from query_generation.initial_query.config import DEFAULT_DB_ROOT, DEFAULT_OUTPUT, DEFAULT_QUERY_ROOT
from query_generation.initial_query.output import write_grouped_queries
from query_generation.initial_query.pipeline import generate_initial_queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate English single-turn travel queries.")
    parser.add_argument("--language", choices=["en"], default="en")
    parser.add_argument("--count", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--model", default="qwen3.6-27b-vllm")
    parser.add_argument("--city-db-root", default=str(DEFAULT_CITY_DB_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-db-root", default=str(DEFAULT_DB_ROOT))
    parser.add_argument("--query-root", default=str(DEFAULT_QUERY_ROOT))
    parser.add_argument("--min-days", type=int, default=2)
    parser.add_argument("--max-days", type=int, default=7)
    parser.add_argument(
        "--date-policy",
        choices=["curated", "all", "range"],
        default="curated",
        help="curated uses 2026-04-30..2026-05-05 routes plus bundled seasonal manifests when present.",
    )
    parser.add_argument("--min-depart-date", default="", help="Optional YYYY-MM-DD lower bound for sampled departure dates.")
    parser.add_argument("--max-depart-date", default="", help="Optional YYYY-MM-DD upper bound for sampled departure dates.")
    parser.add_argument(
        "--curated-date-window",
        action="append",
        default=None,
        help="Curated date window in START:END format. May be repeated.",
    )
    parser.add_argument(
        "--route-manifest",
        action="append",
        default=None,
        help="Optional seasonal route manifest file. May be repeated.",
    )
    parser.add_argument("--no-route-manifest", action="store_true", help="Ignore bundled seasonal route manifests.")
    parser.add_argument(
        "--destination-coverage",
        choices=["off", "reachable", "strict"],
        default="reachable",
        help="Destination coverage policy over cities with feasible round-trip route data.",
    )
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--render-workers", type=int, default=1, help="Concurrent LLM render requests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(BASE_DIR)
    records = generate_initial_queries(args)
    write_json(Path(args.output), records)
    write_grouped_queries(records, Path(args.query_root))
    print(f"Wrote {len(records)} single-turn queries to {args.output}")


if __name__ == "__main__":
    main()
