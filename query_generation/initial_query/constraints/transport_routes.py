"""Route sampling and balancing for transport-backed initial queries.

This module chooses feasible round-trip train/flight frames before the query
record is built. It lives next to transport constraints because both use the
same route evidence, but it does not build visible hard constraints.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from query_generation.city_database import RouteOption, load_city_index, build_route_options
from query_generation.common import load_csv, safe_int
from query_generation.initial_query.constraints.seasonal_routes import (
    SeasonalRouteChoice,
    build_route_choices_from_manifests,
)
from query_generation.initial_query.config import (
    CURATED_MAIN_REMAINDER_SHARE,
    DEFAULT_CURATED_DATE_WINDOWS,
    DEFAULT_FALLBACK_CURATED_WINDOW,
    DEFAULT_MAIN_CURATED_WINDOW,
    DEFAULT_SEASONAL_ROUTE_MANIFESTS,
    DESTINATION_COVERAGE_REFERENCE,
    ROUTE_CATEGORY_BY_MODE,
    ROUTE_FILENAME_BY_MODE,
    SEASONAL_COVERAGE_MANIFESTS,
)


def _route_csv_path_for_option(city_db_root: Path, option: RouteOption, *, outbound: bool) -> Path:
    category = ROUTE_CATEGORY_BY_MODE[option.mode]
    filename = ROUTE_FILENAME_BY_MODE[option.mode]
    origin = option.origin_folder if outbound else option.dest_folder
    dest = option.dest_folder if outbound else option.origin_folder
    return city_db_root / origin / category / dest / filename


def _min_route_duration_for_date(city_db_root: Path, option: RouteOption, *, outbound: bool) -> int | None:
    path = _route_csv_path_for_option(city_db_root, option, outbound=outbound)
    if not path.exists():
        return None
    target_date = option.depart_date if outbound else option.return_date
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in load_csv(path):
        if str(row.get("dep_date", "")).strip() != target_date:
            continue
        route_index = str(row.get("route_index", "")).strip()
        if not route_index:
            continue
        grouped.setdefault(route_index, []).append(row)
    durations = []
    for rows in grouped.values():
        total = sum(safe_int(row.get("duration"), 0) for row in rows)
        if total > 0:
            durations.append(total)
    return min(durations) if durations else None


def _realistic_min_days_for_route(city_db_root: Path, option: RouteOption) -> int:
    durations = [
        value
        for value in (
            _min_route_duration_for_date(city_db_root, option, outbound=True),
            _min_route_duration_for_date(city_db_root, option, outbound=False),
        )
        if value is not None
    ]
    if not durations:
        return 2
    longest_leg = max(durations)
    if option.mode == "flight":
        return 3 if longest_leg >= 120 else 2
    if option.mode == "train":
        if longest_leg >= 480:
            return 4
        return 3 if longest_leg >= 240 else 2
    return 2


def _parse_date_window(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise ValueError(f"Invalid date window {value!r}; expected START:END.")
    start, end = [part.strip() for part in value.split(":", 1)]
    if not start or not end:
        raise ValueError(f"Invalid date window {value!r}; expected START:END.")
    datetime.strptime(start, "%Y-%m-%d")
    datetime.strptime(end, "%Y-%m-%d")
    if end < start:
        raise ValueError(f"Invalid date window {value!r}; END is earlier than START.")
    return start, end


def _curated_date_windows(args: argparse.Namespace) -> list[tuple[str, str]]:
    if not args.curated_date_window:
        return list(DEFAULT_CURATED_DATE_WINDOWS)
    return [_parse_date_window(item) for item in args.curated_date_window]


def _route_manifest_paths(args: argparse.Namespace) -> list[Path]:
    if args.no_route_manifest:
        return []
    if args.route_manifest:
        paths = [Path(item) for item in args.route_manifest]
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Route manifest not found: {missing[0]}")
        return paths
    if args.date_policy == "curated":
        return [path for path in DEFAULT_SEASONAL_ROUTE_MANIFESTS if path.exists()]
    return []


def _option_in_windows(option: RouteOption, windows: list[tuple[str, str]]) -> bool:
    return any(start <= option.depart_date <= end for start, end in windows)


def _route_choices_from_options(options: list[RouteOption], reference: dict[str, Any]) -> list[SeasonalRouteChoice]:
    return [SeasonalRouteChoice(option=option, reference=dict(reference)) for option in options]


def route_choice_key(choice: SeasonalRouteChoice) -> tuple[str, str, str, str, str, str]:
    """Return the stable key used to deduplicate and balance route choices."""
    option = choice.option
    return (
        option.origin_city,
        option.dest_city,
        option.mode,
        option.depart_date,
        option.return_date,
        str(choice.reference.get("manifest_id") or "database_route_options"),
    )


def _local_data_destination_names(city_db_root: Path) -> list[str]:
    city_index = load_city_index(city_db_root)
    names = [
        str(info.get("city_name", "")).strip()
        for info in city_index.values()
        if info.get("has_local_data") and str(info.get("city_name", "")).strip()
    ]
    return sorted(set(names))


def _apply_destination_coverage_policy(
    args: argparse.Namespace,
    city_db_root: Path,
    choices: list[SeasonalRouteChoice],
    all_options: list[RouteOption],
    summary: dict[str, Any],
) -> list[SeasonalRouteChoice]:
    mode = str(getattr(args, "destination_coverage", "reachable") or "reachable")
    summary["destination_coverage"] = {
        "mode": mode,
        "database_destination_count": 0,
        "reachable_destination_count": 0,
        "required_destination_count": 0,
        "supplemental_route_options": 0,
        "unreachable_destinations": [],
        "inactive_reason": None,
    }
    if mode == "off":
        summary["destination_coverage"]["inactive_reason"] = "disabled"
        return choices

    database_destinations = set(_local_data_destination_names(city_db_root))
    all_coverage_choices = _route_choices_from_options(
        all_options,
        {"manifest_id": DESTINATION_COVERAGE_REFERENCE, "season": "mixed", "route_kind": "destination_coverage"},
    )
    reachable_destinations = {choice.option.dest_city for choice in all_coverage_choices}
    unreachable_destinations = sorted(database_destinations - reachable_destinations)
    required_destinations = sorted(database_destinations & reachable_destinations)

    if mode == "strict" and unreachable_destinations:
        raise RuntimeError(
            "Strict destination coverage is impossible with the current round-trip route database. "
            f"Cities without feasible destination routes: {', '.join(unreachable_destinations)}"
        )
    requested_count = int(getattr(args, "count", len(choices)) or len(choices))
    if requested_count < len(required_destinations):
        summary["destination_coverage"]["inactive_reason"] = (
            f"count {requested_count} is smaller than required destination count {len(required_destinations)}"
        )
        required_destinations = []

    present_destinations = {choice.option.dest_city for choice in choices}
    missing_present = set(required_destinations) - present_destinations
    existing_keys = {route_choice_key(choice) for choice in choices}
    supplemental = [
        choice
        for choice in all_coverage_choices
        if choice.option.dest_city in missing_present and route_choice_key(choice) not in existing_keys
    ]

    summary["destination_coverage"].update(
        {
            "database_destination_count": len(database_destinations),
            "reachable_destination_count": len(reachable_destinations),
            "required_destination_count": len(required_destinations),
            "required_destinations": required_destinations,
            "unreachable_destinations": unreachable_destinations,
            "missing_from_base_sampling": sorted(missing_present),
            "supplemental_route_options": len(supplemental),
        }
    )
    return choices + supplemental


def route_sampling_bucket(choice: SeasonalRouteChoice) -> str:
    """Map a route choice to the sampling bucket used by the generator."""
    manifest_id = str(choice.reference.get("manifest_id") or "database_route_options")
    if manifest_id == DESTINATION_COVERAGE_REFERENCE:
        return DESTINATION_COVERAGE_REFERENCE
    if manifest_id in SEASONAL_COVERAGE_MANIFESTS:
        return manifest_id
    depart_date = choice.option.depart_date
    if DEFAULT_MAIN_CURATED_WINDOW[0] <= depart_date <= DEFAULT_MAIN_CURATED_WINDOW[1]:
        return "curated_apr_may"
    if DEFAULT_FALLBACK_CURATED_WINDOW[0] <= depart_date <= DEFAULT_FALLBACK_CURATED_WINDOW[1]:
        return "curated_fallback"
    return manifest_id


def _bucket_availability(route_choices: list[SeasonalRouteChoice]) -> Counter[str]:
    availability: Counter[str] = Counter()
    for choice in route_choices:
        availability[route_sampling_bucket(choice)] += 1
    return availability


def route_sampling_targets(route_choices: list[SeasonalRouteChoice], count: int) -> Counter[str]:
    """Compute per-bucket route targets for the requested generation count."""
    availability = _bucket_availability(route_choices)
    targets: Counter[str] = Counter()
    remaining = count

    seasonal_available = sum(availability[bucket] for bucket in SEASONAL_COVERAGE_MANIFESTS)
    if seasonal_available and remaining >= seasonal_available:
        for bucket in SEASONAL_COVERAGE_MANIFESTS:
            targets[bucket] = availability[bucket]
        remaining -= seasonal_available
    elif seasonal_available:
        seasonal_buckets = [bucket for bucket in SEASONAL_COVERAGE_MANIFESTS if availability[bucket]]
        index = 0
        while remaining > 0 and seasonal_buckets:
            bucket = seasonal_buckets[index % len(seasonal_buckets)]
            if targets[bucket] < availability[bucket]:
                targets[bucket] += 1
                remaining -= 1
            if all(targets[item] >= availability[item] for item in seasonal_buckets):
                break
            index += 1
        return targets

    main_bucket = "curated_apr_may"
    fallback_bucket = "curated_fallback"
    if remaining <= 0:
        return targets
    if availability[main_bucket]:
        main_target = max(1, round(remaining * CURATED_MAIN_REMAINDER_SHARE))
        targets[main_bucket] += min(remaining, main_target)
        remaining -= targets[main_bucket]
    if remaining > 0 and availability[fallback_bucket]:
        targets[fallback_bucket] += remaining
        remaining = 0
    if remaining > 0:
        for bucket in sorted(availability):
            if bucket in targets:
                continue
            targets[bucket] += remaining
            remaining = 0
            break
    if remaining > 0 and availability[main_bucket]:
        targets[main_bucket] += remaining
        remaining = 0
    if remaining > 0:
        for bucket in sorted(availability):
            if availability[bucket]:
                targets[bucket] += remaining
                break
    return targets


def select_route_choice(
    route_choices: list[SeasonalRouteChoice],
    bucket_targets: Counter[str],
    bucket_usage: Counter[str],
    choice_usage: Counter[tuple[str, str, str, str, str, str]],
    route_usage: Counter[tuple[str, str]],
    city_usage: Counter[str],
    destination_usage: Counter[str],
    required_destinations: set[str],
    rng: random.Random,
) -> SeasonalRouteChoice:
    """Select the next route while balancing destinations, cities, and date buckets."""
    priority = {
        "routes_january": 0,
        "routes_july": 1,
        "curated_apr_may": 2,
        "curated_fallback": 3,
    }
    missing_destinations = [
        destination
        for destination in sorted(required_destinations)
        if destination_usage[destination] <= 0
    ]
    if missing_destinations:
        candidates = [
            choice
            for choice in route_choices
            if choice.option.dest_city in missing_destinations
        ]
    else:
        under_target = [
            bucket
            for bucket, target in bucket_targets.items()
            if bucket_usage[bucket] < target
        ]
        seasonal_under = [bucket for bucket in under_target if bucket in SEASONAL_COVERAGE_MANIFESTS]
        candidate_buckets = seasonal_under or under_target
        if candidate_buckets:
            selected_bucket = min(
                candidate_buckets,
                key=lambda bucket: (
                    bucket_usage[bucket] / max(1, bucket_targets[bucket]),
                    priority.get(bucket, 9),
                    bucket,
                ),
            )
            candidates = [choice for choice in route_choices if route_sampling_bucket(choice) == selected_bucket]
        else:
            candidates = list(route_choices)

    if not candidates:
        candidates = list(route_choices)

    if missing_destinations:
        candidates = sorted(
            candidates,
            key=lambda item: (
                destination_usage[item.option.dest_city],
                choice_usage[route_choice_key(item)],
                route_usage[(item.option.origin_city, item.option.dest_city)],
                item.option.days,
                item.option.depart_date,
                item.option.return_date,
                item.option.origin_city,
            ),
        )
        return rng.choice(candidates[: min(24, len(candidates))])

    unused = [choice for choice in candidates if choice_usage[route_choice_key(choice)] == 0]
    if unused:
        ranked_unused = sorted(
            unused,
            key=lambda item: (
                city_usage[item.option.dest_city],
                city_usage[item.option.origin_city],
                route_usage[(item.option.origin_city, item.option.dest_city)],
                item.option.days,
                item.option.depart_date,
                item.option.return_date,
                item.reference.get("season", ""),
            ),
        )
        return rng.choice(ranked_unused[: min(24, len(ranked_unused))])

    ranked = sorted(
        candidates,
        key=lambda item: (
            choice_usage[route_choice_key(item)],
            city_usage[item.option.dest_city],
            city_usage[item.option.origin_city],
            route_usage[(item.option.origin_city, item.option.dest_city)],
            route_usage[(item.option.dest_city, item.option.origin_city)],
            item.option.days,
            item.option.depart_date,
            item.option.return_date,
            item.reference.get("season", ""),
        ),
    )
    return rng.choice(ranked[: min(24, len(ranked))])


def build_initial_route_choices(args: argparse.Namespace, city_db_root: Path) -> tuple[list[SeasonalRouteChoice], dict[str, Any]]:
    """Build route choices for initial-query sampling from DB dates and manifests."""
    all_options = build_route_options(
        city_db_root,
        min_days=args.min_days,
        max_days=args.max_days,
    )
    summary: dict[str, Any] = {
        "date_policy": args.date_policy,
        "base_route_options": len(all_options),
        "date_windows": [],
        "window_route_options": 0,
        "seasonal_route_options": 0,
        "manifest_summary": {},
    }

    choices: list[SeasonalRouteChoice] = []
    if args.date_policy == "all":
        choices.extend(
            _route_choices_from_options(
                all_options,
                {"manifest_id": "database_all_dates", "season": "mixed", "route_kind": "single_city_round_trip"},
            )
        )
    else:
        windows: list[tuple[str, str]] = []
        if args.date_policy == "curated":
            windows = _curated_date_windows(args)
        elif args.min_depart_date or args.max_depart_date:
            windows = [(args.min_depart_date or "0001-01-01", args.max_depart_date or "9999-12-31")]

        if windows:
            summary["date_windows"] = [f"{start}:{end}" for start, end in windows]
            window_options = [option for option in all_options if _option_in_windows(option, windows)]
            summary["window_route_options"] = len(window_options)
            choices.extend(
                _route_choices_from_options(
                    window_options,
                    {"manifest_id": "database_curated_windows", "season": "mixed", "route_kind": "single_city_round_trip"},
                )
            )

    manifest_paths = _route_manifest_paths(args)
    if manifest_paths:
        seasonal_choices, manifest_summary = build_route_choices_from_manifests(
            city_db_root,
            manifest_paths,
            min_days=args.min_days,
            max_days=args.max_days,
        )
        choices.extend(seasonal_choices)
        summary["seasonal_route_options"] = len(seasonal_choices)
        summary["manifest_summary"] = manifest_summary

    if args.min_depart_date or args.max_depart_date:
        choices = [
            choice
            for choice in choices
            if (not args.min_depart_date or choice.option.depart_date >= args.min_depart_date)
            and (not args.max_depart_date or choice.option.depart_date <= args.max_depart_date)
        ]

    deduped: list[SeasonalRouteChoice] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for choice in choices:
        option = choice.option
        key = (option.origin_city, option.dest_city, option.mode, option.depart_date, option.return_date)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(choice)

    realistic = [
        choice
        for choice in deduped
        if choice.option.days >= _realistic_min_days_for_route(city_db_root, choice.option)
    ]
    summary["deduped_route_options"] = len(deduped)
    summary["realistic_route_options"] = len(realistic)
    realistic = _apply_destination_coverage_policy(args, city_db_root, realistic, all_options, summary)
    summary["route_options_after_destination_coverage"] = len(realistic)
    return realistic, summary
