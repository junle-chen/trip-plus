"""City database loading and route enumeration."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import BASE_DIR, load_csv

LOCAL_DATA_FILES = {
    "attractions": ("attractions", "attractions.csv"),
    "hotels": ("hotels", "hotels.csv"),
    "locations": ("locations", "locations_coords.csv"),
    "location_aliases": ("locations", "location_aliases.csv"),
    "location_entities": ("locations", "location_entities.csv"),
    "restaurants": ("restaurants", "restaurants.csv"),
    "transportation": ("transportation", "distance_matrix.csv"),
}

ROUTE_DATA_FILES = {
    "flights": ("flights", "flights.csv"),
    "trains": ("trains", "trains.csv"),
}

DEFAULT_CITY_DB_ROOT = BASE_DIR / "database" / "en"


@dataclass(frozen=True)
class RouteOption:
    origin_city: str
    origin_folder: str
    dest_city: str
    dest_folder: str
    mode: str
    depart_date: str
    return_date: str
    days: int


def _read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {csv_path}")
        return list(reader.fieldnames), list(reader)

def load_city_index(city_db_root: Path) -> dict[str, dict[str, Any]]:
    city_index_path = city_db_root / "city_index.json"
    data = json.loads(city_index_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid city index format: {city_index_path}")
    return data


def _city_name_to_folder(city_index: dict[str, dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for key, info in city_index.items():
        city_name = str(info.get("city_name", "")).strip()
        folder_name = str(info.get("folder_name", key)).strip()
        if city_name:
            mapping[city_name] = folder_name
        if folder_name:
            mapping.setdefault(folder_name, folder_name)
    return mapping


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _route_csv_path(city_db_root: Path, origin_folder: str, category: str, dest_folder: str) -> Path:
    _, filename = ROUTE_DATA_FILES[category]
    return city_db_root / origin_folder / category / dest_folder / filename


def _route_dates(route_csv_path: Path) -> list[str]:
    if not route_csv_path.exists():
        return []
    _, rows = _read_csv_rows(route_csv_path)
    return sorted({str(row.get("dep_date", "")).strip() for row in rows if str(row.get("dep_date", "")).strip()})


def build_route_options(
    city_db_root: Path,
    *,
    min_days: int = 2,
    max_days: int = 7,
) -> list[RouteOption]:
    city_index = load_city_index(city_db_root)
    name_to_folder = _city_name_to_folder(city_index)
    options: list[RouteOption] = []

    for origin_folder, origin_info in city_index.items():
        origin_city = str(origin_info.get("city_name", "")).strip()
        if not origin_city:
            continue

        for category, mode in (("trains", "train"), ("flights", "flight")):
            for dest_city in origin_info.get("connected_cities", {}).get(category, []):
                dest_city = str(dest_city).strip()
                dest_folder = name_to_folder.get(dest_city)
                if not dest_city or not dest_folder:
                    continue

                dest_info = city_index.get(dest_folder)
                if not dest_info or not dest_info.get("has_local_data"):
                    continue

                reverse_connected = {str(city).strip() for city in dest_info.get("connected_cities", {}).get(category, [])}
                if origin_city not in reverse_connected:
                    continue

                outbound_csv = _route_csv_path(city_db_root, origin_folder, category, dest_folder)
                inbound_csv = _route_csv_path(city_db_root, dest_folder, category, origin_folder)
                outbound_dates = _route_dates(outbound_csv)
                inbound_dates = _route_dates(inbound_csv)
                if not outbound_dates or not inbound_dates:
                    continue

                for depart_date in outbound_dates:
                    depart_dt = _parse_date(depart_date)
                    for return_date in inbound_dates:
                        return_dt = _parse_date(return_date)
                        if return_dt < depart_dt:
                            continue
                        days = (return_dt - depart_dt).days + 1
                        if min_days <= days <= max_days:
                            options.append(
                                RouteOption(
                                    origin_city=origin_city,
                                    origin_folder=origin_folder,
                                    dest_city=dest_city,
                                    dest_folder=dest_folder,
                                    mode=mode,
                                    depart_date=depart_date,
                                    return_date=return_date,
                                    days=days,
                                )
                            )
    return options


def _load_city_local_database(city_db_root: Path, dest_folder: str) -> dict[str, list[dict[str, str]]]:
    city_dir = city_db_root / dest_folder
    return {
        "trains": [],
        "flights": [],
        "hotels": load_csv(city_dir / "hotels" / "hotels.csv"),
        "restaurants": load_csv(city_dir / "restaurants" / "restaurants.csv"),
        "attractions": load_csv(city_dir / "attractions" / "attractions.csv"),
        "locations": load_csv(city_dir / "locations" / "locations_coords.csv"),
    }


def _load_route_rows_for_dates(
    city_db_root: Path,
    origin_folder: str,
    dest_folder: str,
    category: str,
    depart_date: str,
    return_date: str,
) -> tuple[list[str], list[dict[str, str]]]:
    headers: list[str] = []
    merged_rows: list[dict[str, str]] = []

    for src_folder, dst_folder, target_date in (
        (origin_folder, dest_folder, depart_date),
        (dest_folder, origin_folder, return_date),
    ):
        route_csv = _route_csv_path(city_db_root, src_folder, category, dst_folder)
        if not route_csv.exists():
            continue
        route_headers, route_rows = _read_csv_rows(route_csv)
        if not headers:
            headers = route_headers
        merged_rows.extend(row for row in route_rows if str(row.get("dep_date", "")).strip() == target_date)

    return headers, merged_rows


def build_city_sample_database(
    city_db_root: Path,
    option: RouteOption,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[str]]]:
    db = _load_city_local_database(city_db_root, option.dest_folder)
    route_headers: dict[str, list[str]] = {}

    for category in ROUTE_DATA_FILES:
        headers, rows = _load_route_rows_for_dates(
            city_db_root,
            option.origin_folder,
            option.dest_folder,
            category,
            option.depart_date,
            option.return_date,
        )
        route_headers[category] = headers
        db[category] = rows

    return db, route_headers
