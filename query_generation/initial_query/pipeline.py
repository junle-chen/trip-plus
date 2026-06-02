"""Sampling loop for English single-turn initial-query generation."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from query_generation.city_database import (
    build_city_sample_database,
)
from query_generation.initial_query.config import (
    DEFAULT_INITIAL_RENDER_MAX_TOKENS,
    DEFAULT_INITIAL_RENDER_TEMPERATURE,
    interaction_targets,
)
from query_generation.initial_query.records import build_initial_query_record, query_signature
from query_generation.initial_query.rendering import render_initial_query_candidates
from query_generation.sample_database import materialize_generated_sample_database
from query_generation.initial_query.constraints.transport_routes import (
    build_initial_route_choices,
    route_choice_key,
    route_sampling_bucket,
    route_sampling_targets,
    select_route_choice,
)
from query_generation.initial_query.visibility import (
    ensure_visible_initial_hard_constraints,
    prune_initial_record_for_output,
)
from query_generation.user_profile import ObservableProfileSampler

RoutePair = tuple[str, str]
RouteChoiceKey = tuple[str, str, str, str, str, str]


@dataclass
class SamplingUsage:
    routes: Counter[RoutePair]
    cities: Counter[str]
    destinations: Counter[str]
    buckets: Counter[str]
    choices: Counter[RouteChoiceKey]
    archetypes: Counter[str]

    @classmethod
    def empty(cls) -> "SamplingUsage":
        return cls(Counter(), Counter(), Counter(), Counter(), Counter(), Counter())

    def fork(self) -> "SamplingUsage":
        return SamplingUsage(
            routes=self.routes.copy(),
            cities=self.cities.copy(),
            destinations=self.destinations.copy(),
            buckets=self.buckets.copy(),
            choices=self.choices.copy(),
            archetypes=self.archetypes.copy(),
        )

    def record_choice(self, option: Any, choice: Any, archetype: str) -> None:
        self.routes[(option.origin_city, option.dest_city)] += 1
        self.cities[option.origin_city] += 1
        self.cities[option.dest_city] += 1
        self.destinations[option.dest_city] += 1
        self.buckets[route_sampling_bucket(choice)] += 1
        self.choices[route_choice_key(choice)] += 1
        self.archetypes[archetype] += 1


def generate_initial_queries(args: Any) -> list[dict[str, Any]]:
    """Generate pruned single-turn records and their per-query sample databases."""
    rng = random.Random(args.seed)
    city_db_root = Path(args.city_db_root)
    output_db_root = Path(args.output_db_root)
    output_db_root.mkdir(parents=True, exist_ok=True)

    route_choices, route_choice_summary = build_initial_route_choices(args, city_db_root)
    if not route_choices:
        raise RuntimeError("No route options found for initial query generation.")

    sampler = ObservableProfileSampler()
    records: list[dict[str, Any]] = []
    seen_signatures: set[tuple[Any, ...]] = set()
    seen_queries: set[str] = set()
    usage = SamplingUsage.empty()

    archetype_targets = interaction_targets(args.count)
    route_bucket_targets = route_sampling_targets(route_choices, args.count)
    required_destinations = set(
        route_choice_summary.get("destination_coverage", {}).get("required_destinations") or []
    )

    attempts = 0
    render_workers = max(1, int(getattr(args, "render_workers", 1) or 1))
    candidate_batch_size = 1 if args.skip_llm else render_workers
    while len(records) < args.count and attempts < args.count * 40:
        candidates: list[dict[str, Any]] = []
        local_seen_signatures: set[tuple[Any, ...]] = set()
        batch_usage = usage.fork()

        while (
            len(candidates) < candidate_batch_size
            and len(records) + len(candidates) < args.count
            and attempts < args.count * 40
        ):
            attempts += 1
            remaining_archetypes = [
                label
                for label, target in archetype_targets.items()
                if batch_usage.archetypes[label] < target
            ]
            if not remaining_archetypes:
                break

            interaction_archetype = rng.choice(remaining_archetypes)
            choice = select_route_choice(
                route_choices,
                route_bucket_targets,
                batch_usage.buckets,
                batch_usage.choices,
                batch_usage.routes,
                batch_usage.cities,
                batch_usage.destinations,
                required_destinations,
                rng,
            )
            option = choice.option
            db, route_headers = build_city_sample_database(city_db_root, option)
            record = build_initial_query_record(
                sample_id=f"single_candidate_{attempts:04d}",
                option=option,
                db=db,
                city_db_root=city_db_root,
                interaction_archetype=interaction_archetype,
                sampler=sampler,
                rng=rng,
            )

            signature = query_signature(record)
            if signature in seen_signatures or signature in local_seen_signatures:
                continue

            local_seen_signatures.add(signature)
            batch_usage.record_choice(option, choice, interaction_archetype)
            candidates.append(
                {
                    "record": record,
                    "db": db,
                    "route_headers": route_headers,
                    "option": option,
                    "choice": choice,
                    "interaction_archetype": interaction_archetype,
                    "signature": signature,
                }
            )

        if not candidates:
            break

        render_initial_query_candidates(
            candidates,
            model=args.model,
            skip_llm=args.skip_llm,
            workers=render_workers,
            temperature=getattr(args, "render_temperature", DEFAULT_INITIAL_RENDER_TEMPERATURE),
            max_tokens=getattr(args, "render_max_tokens", DEFAULT_INITIAL_RENDER_MAX_TOKENS),
        )

        for candidate in candidates:
            if len(records) >= args.count:
                break
            record = candidate["record"]
            ensure_visible_initial_hard_constraints(record)
            record["query"] = str(record.get("query", "")).strip()
            if not record["query"] or record["query"] in seen_queries:
                continue

            final_sample_id = f"single_{len(records):04d}"
            record["id"] = final_sample_id
            materialize_generated_sample_database(
                city_db_root,
                candidate["option"],
                final_sample_id,
                candidate["db"],
                candidate["route_headers"],
                output_db_root,
            )

            records.append(record)
            seen_signatures.add(candidate["signature"])
            seen_queries.add(record["query"])

            option = candidate["option"]
            choice = candidate["choice"]
            usage.record_choice(option, choice, candidate["interaction_archetype"])

    if len(records) < args.count:
        raise RuntimeError(f"Only generated {len(records)} single-turn queries after {attempts} attempts.")

    missing_required_destinations = sorted(required_destinations - set(usage.destinations))
    if missing_required_destinations:
        raise RuntimeError(
            "Destination coverage target was not satisfied after generation. "
            f"Missing destinations: {', '.join(missing_required_destinations)}"
        )

    return [prune_initial_record_for_output(record) for record in records]
