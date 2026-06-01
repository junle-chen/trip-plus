"""Seasonal route-manifest loading for curated transport coverage."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from query_generation.city_database import RouteOption, build_route_options


ROUTE_LINE_RE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})\s+(.+?)\s*->\s*(.+?)\s*$")


@dataclass(frozen=True)
class RouteSegment:
    date: str
    origin_city: str
    dest_city: str


@dataclass(frozen=True)
class SeasonalRouteChoice:
    option: RouteOption
    reference: dict[str, Any]


def _season_from_manifest_id(manifest_id: str) -> str:
    lowered = manifest_id.lower()
    if "january" in lowered or "winter" in lowered:
        return "winter"
    if "july" in lowered or "summer" in lowered:
        return "summer"
    return "seasonal"


def _parse_route_groups(path: Path) -> list[list[RouteSegment]]:
    groups: list[list[RouteSegment]] = []
    current: list[RouteSegment] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                groups.append(current)
                current = []
            continue
        if line.startswith("#"):
            continue
        match = ROUTE_LINE_RE.match(line)
        if not match:
            continue
        current.append(
            RouteSegment(
                date=match.group(1),
                origin_city=match.group(2).strip(),
                dest_city=match.group(3).strip(),
            )
        )
    if current:
        groups.append(current)
    return groups


def _days_inclusive(depart_date: str, return_date: str) -> int:
    depart = datetime.strptime(depart_date, "%Y-%m-%d").date()
    ret = datetime.strptime(return_date, "%Y-%m-%d").date()
    return (ret - depart).days + 1


def _is_reverse_pair(outbound: RouteSegment, inbound: RouteSegment) -> bool:
    return outbound.origin_city == inbound.dest_city and outbound.dest_city == inbound.origin_city


def _is_closed_loop(segments: list[RouteSegment]) -> bool:
    if len(segments) < 3:
        return False
    for current, nxt in zip(segments, segments[1:]):
        if current.dest_city != nxt.origin_city:
            return False
    return segments[-1].dest_city == segments[0].origin_city


def load_seasonal_route_requests(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    summary = {
        "manifest_ids": [],
        "round_trip_requests": 0,
        "multi_city_loops": 0,
        "unsupported_segments": 0,
    }
    seen_requests: set[tuple[str, str, str, str, str]] = set()

    for path in paths:
        manifest_id = path.stem
        season = _season_from_manifest_id(manifest_id)
        summary["manifest_ids"].append(manifest_id)
        for group_index, group in enumerate(_parse_route_groups(path)):
            index = 0
            while index < len(group):
                if index + 1 < len(group) and _is_reverse_pair(group[index], group[index + 1]):
                    outbound = group[index]
                    inbound = group[index + 1]
                    key = (
                        manifest_id,
                        outbound.origin_city,
                        outbound.dest_city,
                        outbound.date,
                        inbound.date,
                    )
                    if key not in seen_requests:
                        seen_requests.add(key)
                        requests.append(
                            {
                                "origin_city": outbound.origin_city,
                                "dest_city": outbound.dest_city,
                                "depart_date": outbound.date,
                                "return_date": inbound.date,
                                "days": _days_inclusive(outbound.date, inbound.date),
                                "season": season,
                                "manifest_id": manifest_id,
                                "route_kind": "single_city_round_trip",
                                "group_index": group_index,
                            }
                        )
                        summary["round_trip_requests"] += 1
                    index += 2
                    continue
                if _is_closed_loop(group[index:]):
                    summary["multi_city_loops"] += 1
                    break
                summary["unsupported_segments"] += 1
                index += 1
    summary["manifest_ids"] = sorted(set(summary["manifest_ids"]))
    return requests, summary


def build_route_choices_from_manifests(
    city_db_root: Path,
    manifest_paths: list[Path],
    *,
    min_days: int = 2,
    max_days: int = 7,
) -> tuple[list[SeasonalRouteChoice], dict[str, Any]]:
    requests, summary = load_seasonal_route_requests(manifest_paths)
    all_options = build_route_options(city_db_root, min_days=min_days, max_days=max_days)
    options_by_key: dict[tuple[str, str, str, str], list[RouteOption]] = {}
    for option in all_options:
        key = (option.origin_city, option.dest_city, option.depart_date, option.return_date)
        options_by_key.setdefault(key, []).append(option)

    choices: list[SeasonalRouteChoice] = []
    seen_choices: set[tuple[str, str, str, str, str, str]] = set()
    missing = 0
    for request in requests:
        key = (
            request["origin_city"],
            request["dest_city"],
            request["depart_date"],
            request["return_date"],
        )
        matched_options = options_by_key.get(key, [])
        if not matched_options:
            missing += 1
            continue
        for option in matched_options:
            choice_key = (
                request["manifest_id"],
                option.origin_city,
                option.dest_city,
                option.depart_date,
                option.return_date,
                option.mode,
            )
            if choice_key in seen_choices:
                continue
            seen_choices.add(choice_key)
            choices.append(
                SeasonalRouteChoice(
                    option=option,
                    reference={
                        "manifest_id": request["manifest_id"],
                        "season": request["season"],
                        "route_kind": request["route_kind"],
                    },
                )
            )

    summary = dict(summary)
    summary["matched_round_trip_options"] = len(choices)
    summary["missing_round_trip_requests"] = missing
    return choices, summary
