"""Command-line entry point for English multi-turn query generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from query_generation.multiturn_query.config import (
    DEFAULT_DB_ROOT,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    DEFAULT_QUERY_ROOT,
    DEFAULT_TURN_RENDER_MODEL,
)
from query_generation.multiturn_query.output import write_outputs
from query_generation.multiturn_query.records import (
    generate_multiturn_records,
    refresh_environment_records_from_template,
)
from query_generation.multiturn_query.rendering import llm_render_multiturn_surfaces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate English multi-turn benchmark queries from English single-turn records.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--query-root", default=str(DEFAULT_QUERY_ROOT))
    parser.add_argument("--database-root", default=str(DEFAULT_DB_ROOT))
    parser.add_argument("--generation-language", choices=["en"], default="en")
    parser.add_argument("--count", type=int, default=0, help="Number of single-turn records to convert. 0 means all.")
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--refresh-environment-from", default="")
    parser.add_argument("--environment-output", default="")
    parser.add_argument("--llm-render-turns", action="store_true")
    parser.add_argument("--turn-render-model", default=DEFAULT_TURN_RENDER_MODEL)
    parser.add_argument("--turn-render-language", choices=["en"], default="en")
    parser.add_argument("--turn-render-temperature", type=float, default=1.0)
    parser.add_argument("--turn-render-max-tokens", type=int, default=260)
    parser.add_argument("--turn-render-workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = generate_multiturn_records(args)
    records = refresh_environment_records_from_template(records, args)
    records = llm_render_multiturn_surfaces(records, args)
    output_path = Path(args.output)
    counts = write_outputs(records, output_path, Path(args.query_root), args.environment_output)
    print(f"Wrote {len(records)} multi-turn queries to {output_path}")
    print(f"Wrote grouped multi-turn queries to {args.query_root}")
    print(json.dumps(dict(sorted(counts.items())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
