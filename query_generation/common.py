"""Shared dataclasses and parsing helpers for query generation."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from util import env as _env
from util import io as _io
from util.numeric import safe_float, safe_int

BASE_DIR = Path(__file__).resolve().parents[1]

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

load_env_file = _env.load_env_file
load_json = _io.read_json
write_json = _io.write_json


SERVICE_KEY_MAP = {
    "Swimming pool": "Swimming_Pool",
    "Swimming Pool": "Swimming_Pool",
    "Pool": "Swimming_Pool",
    "Washer and dryer": "Washer_Dryer",
    "Washer and Dryer": "Washer_Dryer",
    "Gym": "Gym",
    "SPA service": "Spa",
    "SPA services": "Spa",
    "SPA Service": "Spa",
    "SPA Services": "Spa",
    "Spa services": "Spa",
    "Robot room service": "Robot_Room_Service",
    "Robot Room Service": "Robot_Room_Service",
    "Robot Service": "Robot_Room_Service",
    "TV screen casting": "Screen_Casting_TV",
    "TV Screen Casting": "Screen_Casting_TV",
    "TV screen mirroring": "Screen_Casting_TV",
    "TV Screen Mirroring": "Screen_Casting_TV",
}

TAG_KEY_MAP = {
    "Birthday set meal service": "Birthday_Package",
    "Birthday Set Meal Service": "Birthday_Package",
    "Birthday set meals": "Birthday_Package",
    "Private rooms available": "Private_Room",
    "Private Rooms Available": "Private_Room",
    "Private rooms": "Private_Room",
    "Private Rooms": "Private_Room",
    "Outdoor seating": "Outdoor_Seating",
    "Outdoor Seating": "Outdoor_Seating",
    "Waiting area available": "Waiting_Area",
    "Waiting Area Available": "Waiting_Area",
    "Waiting Area": "Waiting_Area",
    "Online queueing": "Queue_Remote",
    "Online Queueing": "Queue_Remote",
    "Online Queue Number": "Queue_Remote",
    "Online queue number": "Queue_Remote",
    "Must-Eat List Top 10": "Must_Eat_List_Top10",
}

GENERIC_CUISINES = {
    "Catering Services",
    "Catering services",
    "Catering service",
    "Catering Service",
    "Dining Related",
    "Dining Related Venues",
    "Dining Services",
    "Dining services",
    "Dining Venues",
}


@dataclass
class ConstraintSpec:
    key: str
    data: Dict[str, Any]
    visible_hint: str
    query_bullet: str


@dataclass
class SampleContext:
    sample_id: str
    org: str
    dest: str
    days: int
    depart_date: str
    return_date: str
    people_number: int
    room_number: int
    depart_weekday: int


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_dt(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def split_semicolon_field(value: str) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def extract_cuisine_label(value: str) -> str:
    for token in reversed(split_semicolon_field(value)):
        if token not in GENERIC_CUISINES:
            return token
    return ""


def visible_weekday(depart_weekday: int) -> str:
    index = depart_weekday - 1
    return WEEKDAY_NAMES[index] if 0 <= index < 7 else ""


def date_text(date_str: str) -> str:
    year, month, day = date_str.split("-")
    return f"{year}-{int(month):02d}-{int(day):02d}"


def common_room_number(people_number: int) -> int:
    return max(1, math.ceil(people_number / 2))


def build_direct_routes(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """Return rows that match the tool-side definition of direct transport.

    Route indices are not globally unique in generated sample databases: the
    same route_index can appear in both directions, and incomplete transfer
    fragments can share the requested origin/destination labels.  A direct
    candidate must therefore be scoped by direction and must be a first segment
    that reaches a terminal station observed for the OD pair when transfer
    routes exist.
    """
    route_groups: Dict[tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        route_index = str(row.get("route_index") or "").strip()
        if not route_index:
            continue
        key = (str(row.get("origin_city") or ""), str(row.get("destination_city") or ""), route_index)
        route_groups[key].append(row)

    terminal_codes: Dict[tuple[str, str], set[str]] = defaultdict(set)
    terminal_names: Dict[tuple[str, str], set[str]] = defaultdict(set)
    for (origin, destination, _route_index), group in route_groups.items():
        max_segment_index = max(safe_int(row.get("segment_index")) for row in group)
        if max_segment_index <= 1:
            continue
        pair = (origin, destination)
        for row in group:
            if safe_int(row.get("segment_index")) != max_segment_index:
                continue
            code = str(row.get("arr_station_code") or "").strip()
            name = str(row.get("arr_station_name") or "").strip()
            if code:
                terminal_codes[pair].add(code)
            if name:
                terminal_names[pair].add(name)

    direct_routes: Dict[str, List[Dict[str, str]]] = {}
    for (origin, destination, route_index), group in route_groups.items():
        pair = (origin, destination)
        final_codes = terminal_codes.get(pair) or set()
        final_names = terminal_names.get(pair) or set()
        first_rows = [row for row in group if safe_int(row.get("segment_index")) == 1]
        if final_codes or final_names:
            first_rows = [
                row
                for row in first_rows
                if (
                    str(row.get("arr_station_code") or "").strip() in final_codes
                    or str(row.get("arr_station_name") or "").strip() in final_names
                )
            ]
        for index, row in enumerate(first_rows):
            direct_routes[f"{origin}\t{destination}\t{route_index}\t{index}"] = [row]
    return direct_routes


def route_direction(row: Dict[str, str], ctx: SampleContext) -> Optional[str]:
    if row.get("origin_city") == ctx.org and row.get("destination_city") == ctx.dest:
        return "outbound"
    if row.get("origin_city") == ctx.dest and row.get("destination_city") == ctx.org:
        return "inbound"
    return None


def filtered_rows_by_direction(rows: Iterable[Dict[str, str]], ctx: SampleContext, direction: str) -> List[Dict[str, str]]:
    return [row for row in rows if route_direction(row, ctx) == direction]


def euclidean_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return math.sqrt(((lat1 - lat2) * 111_000) ** 2 + ((lon1 - lon2) * 85_000) ** 2)


def distance_to_nearby_attraction(row: Dict[str, str]) -> float:
    coords = row.get("nearby_attraction_coords", "")
    if "," not in coords:
        return 999999.0
    lng_str, lat_str = coords.split(",", 1)
    lat1 = safe_float(row.get("latitude"))
    lon1 = safe_float(row.get("longitude"))
    lat2 = safe_float(lat_str)
    lon2 = safe_float(lng_str)
    if not all([lat1, lon1, lat2, lon2]):
        return 999999.0
    return euclidean_distance_m(lat1, lon1, lat2, lon2)
