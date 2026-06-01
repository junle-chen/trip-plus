from __future__ import annotations

import csv
import difflib
import heapq
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base_travel_tool import BaseTravelTool, register_tool
from .city_db_access import (
    load_city_location_aliases,
    load_city_location_entities,
    load_city_locations,
    load_city_subway,
    load_city_subway_stations,
    load_city_transportation,
    load_city_weather,
    resolve_city_db_root,
)
from .weather_query_tool import build_weather_advisory, classify_weather_condition


WALKING_SPEED_KMH = 5.0
METRO_SPEED_KMH = 30.0

TAXI_BASE_FARE = {
    "Beijing": (13.0, 3.0, 2.3),
    "Shanghai": (14.0, 3.0, 2.7),
    "Nanjing": (11.0, 3.0, 2.4),
    "Guangzhou": (12.0, 3.0, 2.6),
    "Chengdu": (10.0, 2.0, 1.9),
    "Hangzhou": (13.0, 3.0, 2.5),
    "Wuhan": (10.0, 3.0, 2.0),
    "Shenzhen": (10.0, 2.0, 2.6),
    "Suzhou": (10.0, 3.0, 2.0),
    "Chongqing": (10.0, 3.0, 2.0),
    "Sanya": (10.0, 2.0, 2.0),
    "Urumqi": (10.0, 3.0, 1.3),
    "Nanning": (9.0, 2.0, 1.6),
    "Hefei": (10.0, 2.5, 1.8),
    "Hohhot": (8.0, 3.0, 1.6),
    "Harbin": (8.0, 3.0, 1.9),
    "Tianjin": (11.0, 3.0, 1.7),
    "Ningbo": (11.0, 3.0, 2.4),
    "Kunming": (8.0, 3.0, 1.8),
    "Shenyang": (9.0, 3.0, 1.8),
    "Jinan": (12.0, 3.0, 1.8),
    "Zhuhai": (10.0, 3.0, 2.6),
    "Shijiazhuang": (8.0, 3.0, 1.6),
    "Fuzhou": (10.0, 3.0, 2.0),
    "Xi'an": (9.0, 3.0, 2.0),
    "Guiyang": (10.0, 3.0, 1.8),
    "Changchun": (8.0, 2.5, 1.9),
}
DEFAULT_TAXI_FARE = (10.0, 3.0, 2.2)
DEFAULT_CURRENCY = "CNY"
HONG_KONG_LOCAL_TO_CNY_RATE = 0.9
HK_CITY_ALIASES = {"Hong Kong", "hong kong", "hong_kong", "hk", "hksar"}
DEPARTURE_TIME_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%d")

METRO_RULES = {
    "Beijing": "Beijing metro fare estimate: CNY 3 within 6 km, CNY 4 for 6-12 km, CNY 5 for 12-22 km, CNY 6 for 22-32 km, then +CNY 1 per additional 20 km; airport express lines excluded.",
    "Shanghai": "Shanghai metro fare estimate: CNY 3 within 6 km, CNY 4 for 6-16 km, CNY 5 for 16-26 km, CNY 6 for 26-36 km, CNY 7 for 36-46 km, then +CNY 1 per additional 20 km.",
    "Guangzhou": "Guangzhou metro fare estimate: CNY 2 within 4 km, CNY 3 for 4-12 km, CNY 4 for 12-24 km, then +CNY 1 per additional 8 km.",
    "Shenzhen": "Shenzhen metro fare estimate: CNY 2 within 4 km, CNY 3 for 4-12 km, CNY 4 for 12-24 km, then +CNY 1 per additional 8 km.",
    "Hong Kong": "Hong Kong urban MTR fare estimate converted to CNY from local distance bands; airport express, light rail, payment-medium differences, and special discounts are excluded.",
    "Sanya": "Sanya rail/tram estimate: CNY 2 base fare.",
    "default": "Default metro fare estimate: CNY 2 for 0-4 km, CNY 3 for 4-9 km, CNY 4 for 9-14 km, CNY 5 for 14-21 km, CNY 6 for 21-28 km, CNY 7 for 28-35 km, CNY 8 for 35-43 km, CNY 9 for 43-51 km, CNY 10 for 51-61 km, then +CNY 1 per additional 15 km.",
}

PREFERENCE_ALIASES = {
    "balanced": "balanced",
    "budget_first": "budget_first",
    "budget": "budget_first",
    "min_walking": "min_walking",
    "comfort": "min_walking",
    "time_first": "time_first",
    "fastest": "time_first",
}


def compact_place_name(value: object) -> str:
    return (
        str(value or "")
        .strip()
        .replace(" ", "")
        .replace("　", "")
        .replace("（", "(")
        .replace("）", ")")
    )


def alias_lookup_keys(value: object) -> list[str]:
    raw = str(value or "").strip()
    compact = compact_place_name(raw)
    keys = []
    for key in (raw, compact):
        if key and key not in keys:
            keys.append(key)
    return keys


def ordered_unique_texts(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = compact_place_name(text).lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _normalize_transport_hub_name(value: object) -> str:
    text = compact_place_name(value)
    text = re.sub(r"\bterminal\s*\d+\b|\bt\d+\b", "", text, flags=re.I)
    return text


def _english_hub_core(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\bterminal\s*\d+\b|\bt\d+\b", " ", text)
    text = re.sub(r"\binternational\b|\brailway\b|\brailroad\b|\btrain\b|\bstation\b", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _english_core_suffix_match(query_core: str, row_core: str) -> bool:
    if not query_core or not row_core:
        return False
    generic_direction_cores = {"north", "south", "east", "west"}
    if query_core in generic_direction_cores or row_core in generic_direction_cores:
        return query_core == row_core
    if query_core == row_core or query_core in row_core or row_core in query_core:
        return True
    min_len = max(4, len(query_core) - 2)
    max_len = min(len(row_core), len(query_core) + 4)
    for size in range(min_len, max_len + 1):
        suffix = row_core[-size:]
        if difflib.SequenceMatcher(None, query_core, suffix).ratio() >= 0.86:
            return True
    return False


def _english_transport_hub_name_matches(query_name: object, row_name: object) -> bool:
    query_raw = str(query_name or "").lower()
    row_raw = str(row_name or "").lower()

    if "airport" in query_raw and "airport" in row_raw:
        query_core = _english_hub_core(re.sub(r"\bairport\b", " ", str(query_name), flags=re.I))
        row_core = _english_hub_core(re.sub(r"\bairport\b", " ", str(row_name), flags=re.I))
        if not query_core or not row_core:
            return False
        return (
            _english_core_suffix_match(query_core, row_core)
            or query_core == row_core
            or query_core.endswith(row_core)
            or row_core.endswith(query_core)
            or query_core in row_core
            or row_core in query_core
            or difflib.SequenceMatcher(None, query_core, row_core).ratio() >= 0.86
        )

    station_tokens = ("station", "railway")
    direction_tokens = ("north", "south", "east", "west")
    query_is_station_like = any(token in query_raw for token in station_tokens) or any(
        re.search(rf"\b{token}\b", query_raw) for token in direction_tokens
    )
    if query_is_station_like:
        query_core = _english_hub_core(query_name)
        row_core = _english_hub_core(row_name)
        if not query_core or not row_core:
            return False
        generic_direction_cores = {"north", "south", "east", "west"}
        if query_core in generic_direction_cores or row_core in generic_direction_cores:
            return query_core == row_core
        if not any(token in row_raw for token in station_tokens) and query_core != row_core:
            return False
        return (
            _english_core_suffix_match(query_core, row_core)
            or query_core == row_core
            or query_core.endswith(row_core)
            or row_core.endswith(query_core)
            or query_core in row_core
            or row_core in query_core
        )
    return False


def _transport_hub_name_matches(query_name: object, row_name: object) -> bool:
    query = _normalize_transport_hub_name(query_name)
    row = _normalize_transport_hub_name(row_name)
    if not query or not row:
        return False
    return _english_transport_hub_name_matches(query_name, row_name)


def _can_fuzzy_match_transport_hub(row: dict[str, str]) -> bool:
    poi_type = str(row.get("poi_type", "") or row.get("type", "")).lower()
    source = str(row.get("source", "")).lower()
    row_name = str(row.get("poi_name", "")).strip()
    if any(token in poi_type for token in ("hotel", "restaurant", "attraction")):
        return False
    if re.search(r"\b(hotel|restaurant|cafe|coffee|buffet|cuisine|dining|lounge|bistro|bar)\b", row_name, re.I):
        return False
    return (
        any(token in poi_type for token in ("station", "airport", "subway", "railway", "transport"))
        or source in {"city_subway", "transport_hub", "route_station_name"}
        or bool(re.search(r"\b(airport|railway station|train station|station)\b", row_name, re.I))
    )


def read_location_aliases(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists():
        return {}
    aliases: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            alias = str(row.get("alias", "")).strip()
            canonical_name = str(row.get("canonical_name", "")).strip()
            if not alias or not canonical_name or alias == canonical_name:
                continue
            for key in alias_lookup_keys(alias):
                aliases.setdefault(key, canonical_name)
    return aliases


def find_place_by_alias(
    rows: list[dict[str, str]],
    place_name: str,
    city_name: str = "",
    aliases: dict[str, str] | None = None,
) -> dict[str, str] | None:
    aliases = aliases or {}
    place_text = str(place_name or "").strip()
    place_compact = compact_place_name(place_text)

    for row in rows:
        row_name = str(row.get("poi_name", "")).strip()
        if row_name == place_text or compact_place_name(row_name) == place_compact:
            return row
        if _can_fuzzy_match_transport_hub(row) and _transport_hub_name_matches(place_text, row_name):
            return row

    canonical_name = None
    for key in alias_lookup_keys(place_name):
        canonical_name = aliases.get(key)
        if canonical_name:
            break
    if canonical_name:
        canonical_compact = compact_place_name(canonical_name)
        for row in rows:
            row_name = str(row.get("poi_name", "")).strip()
            if row_name == canonical_name or compact_place_name(row_name) == canonical_compact:
                return row
    return None


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def meters_to_km_text(distance_meters: float) -> str:
    return f"{distance_meters / 1000.0:.1f}km"


def minutes_from_distance(distance_meters: float, speed_kmh: float) -> int:
    if distance_meters <= 0:
        return 0
    return max(1, int(round((distance_meters / 1000.0) / speed_kmh * 60)))


def estimate_taxi_cost(city_name: str, distance_meters: float) -> int:
    cost, _, _ = estimate_taxi_price(city_name, distance_meters)
    return int(round(cost))


def calculate_metro_cost(distance_meters: float) -> int:
    cost, _, _ = calculate_metro_price("", distance_meters)
    return int(round(cost))


def normalize_pricing_city(city_name: str) -> str:
    raw = str(city_name or "").strip()
    normalized = raw.lower().replace("-", "_").replace(" ", "_")
    return "Hong Kong" if raw in HK_CITY_ALIASES or normalized in HK_CITY_ALIASES else raw


def local_currency(city_name: str) -> str:
    return DEFAULT_CURRENCY


def round_price(value: float) -> float | int:
    return int(value) if float(value).is_integer() else round(value, 1)


def convert_hong_kong_local_price_to_cny(cost: float | int) -> float | int:
    return round_price(float(cost) * HONG_KONG_LOCAL_TO_CNY_RATE)


def coordinate_text(row: dict[str, str]) -> str:
    return f"{str(row.get('latitude', '')).strip()},{str(row.get('longitude', '')).strip()}"


def estimate_taxi_price(city_name: str, distance_meters: float) -> tuple[float, str, str]:
    pricing_city = normalize_pricing_city(city_name)
    currency = DEFAULT_CURRENCY
    if distance_meters <= 1200:
        return 0.0, currency, "Taxi is not recommended within the walking threshold; cost is recorded as 0."

    distance_km = distance_meters / 1000.0
    if pricing_city == "Hong Kong":
        if distance_km <= 2.0:
            return convert_hong_kong_local_price_to_cny(29.0), DEFAULT_CURRENCY, (
                "Hong Kong urban red taxi fare estimate converted to CNY: about CNY 26.1 for the first 2 km."
            )
        remaining_steps = math.ceil((distance_km - 2.0) / 0.2)
        first_tier_steps = min(remaining_steps, 35)
        second_tier_steps = max(0, remaining_steps - 35)
        cost = 29.0 + first_tier_steps * 2.1 + second_tier_steps * 1.4
        rule = (
            "Hong Kong urban red taxi fare estimate converted to CNY: about CNY 26.1 for the first 2 km, "
            "then about CNY 1.9 per 200 m until about 9 km, then about CNY 1.3 per 200 m; "
            "waiting time, tunnel tolls, and booking surcharges are excluded."
        )
        return convert_hong_kong_local_price_to_cny(cost), DEFAULT_CURRENCY, rule

    start_fare, included_km, per_km = TAXI_BASE_FARE.get(pricing_city, DEFAULT_TAXI_FARE)
    extra_km = max(0.0, distance_km - included_km)
    cost = start_fare + extra_km * per_km
    rule = (
        f"{pricing_city or 'default city'} taxi fare estimate: base fare {start_fare:.0f} CNY "
        f"covers the first {included_km:g} km, then {per_km:g} CNY/km."
    )
    return round_price(cost), DEFAULT_CURRENCY, rule


def calculate_metro_price(
    city_name: str,
    distance_meters: float,
    line_path: Optional[list[str]] = None,
) -> tuple[float, str, str]:
    pricing_city = normalize_pricing_city(city_name)
    distance_km = distance_meters / 1000.0
    line_path = line_path or []

    if pricing_city == "Beijing":
        thresholds = [(6.0, 3), (12.0, 4), (22.0, 5), (32.0, 6)]
        for upper, price in thresholds:
            if distance_km <= upper:
                return price, DEFAULT_CURRENCY, METRO_RULES["Beijing"]
        return 6 + math.ceil((distance_km - 32.0) / 20.0), DEFAULT_CURRENCY, METRO_RULES["Beijing"]

    if pricing_city == "Shanghai":
        thresholds = [(6.0, 3), (16.0, 4), (26.0, 5), (36.0, 6), (46.0, 7)]
        for upper, price in thresholds:
            if distance_km <= upper:
                return price, DEFAULT_CURRENCY, METRO_RULES["Shanghai"]
        return 7 + math.ceil((distance_km - 46.0) / 20.0), DEFAULT_CURRENCY, METRO_RULES["Shanghai"]

    if pricing_city in {"Guangzhou", "Shenzhen"}:
        thresholds = [(4.0, 2), (12.0, 3), (24.0, 4)]
        for upper, price in thresholds:
            if distance_km <= upper:
                return price, DEFAULT_CURRENCY, METRO_RULES[pricing_city]
        return 4 + math.ceil((distance_km - 24.0) / 8.0), DEFAULT_CURRENCY, METRO_RULES[pricing_city]

    if pricing_city == "Hong Kong":
        if any("Airport Express" in line for line in line_path):
            return convert_hong_kong_local_price_to_cny(35.0), DEFAULT_CURRENCY, (
                "When a Hong Kong route includes Airport Express, fares vary strongly by station pair; "
                "this estimate uses a conservative low common adult fare tier converted to CNY."
            )
        thresholds = [(5.0, 5.0), (10.0, 7.0), (15.0, 9.0), (20.0, 11.0), (25.0, 13.0), (30.0, 15.0)]
        for upper, price in thresholds:
            if distance_km <= upper:
                return convert_hong_kong_local_price_to_cny(price), DEFAULT_CURRENCY, METRO_RULES["Hong Kong"]
        local_price = 15.0 + math.ceil((distance_km - 30.0) / 5.0) * 2.0
        return convert_hong_kong_local_price_to_cny(local_price), DEFAULT_CURRENCY, METRO_RULES["Hong Kong"]

    if pricing_city == "Sanya":
        return 2, DEFAULT_CURRENCY, METRO_RULES["Sanya"]

    thresholds = [
        (4.0, 2),
        (9.0, 3),
        (14.0, 4),
        (21.0, 5),
        (28.0, 6),
        (35.0, 7),
        (43.0, 8),
        (51.0, 9),
        (61.0, 10),
    ]
    for upper, price in thresholds:
        if distance_km <= upper:
            return price, DEFAULT_CURRENCY, METRO_RULES["default"]
    return 10 + math.ceil((distance_km - 61.0) / 15.0), DEFAULT_CURRENCY, METRO_RULES["default"]


def normalize_subway_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not payload:
        return []

    if isinstance(payload.get("lines"), list):
        lines = []
        for line in payload["lines"]:
            stations = []
            for station in line.get("stations", []):
                name = str(station.get("name", "")).strip()
                position = str(station.get("position", "")).strip()
                lon = station.get("longitude")
                lat = station.get("latitude")
                if position and "," in position:
                    lon, lat = [part.strip() for part in position.split(",", 1)]
                if not name or lon in (None, "") or lat in (None, ""):
                    continue
                stations.append(
                    {
                        "name": name,
                        "lat": to_float(lat),
                        "lon": to_float(lon),
                    }
                )
            if stations:
                lines.append({"name": str(line.get("name", "")).strip(), "stations": stations})
        return lines

    amap_lines = payload.get("l") or []
    lines = []
    for line in amap_lines:
        stations = []
        for station in line.get("st", []):
            name = str(station.get("n", "")).strip()
            position = str(station.get("sl", "")).strip()
            if not name or "," not in position:
                continue
            lon, lat = [part.strip() for part in position.split(",", 1)]
            stations.append(
                {
                    "name": name,
                    "lat": to_float(lat),
                    "lon": to_float(lon),
                }
            )
        if stations:
            lines.append({"name": str(line.get("ln", "")).strip(), "stations": stations})
    return lines


@register_tool("query_city_transport_plan")
class CityTransportQueryTool(BaseTravelTool):
    """Query intra-city transport recommendations with walking / taxi / metro options."""

    LANG_FIELDS = {
        "en": {
            "missing_city_db": "City-level database not found. Build a sample cache from database/{lang} before using this tool",
            "city_not_found": lambda city: f"City-level data not found for {city}",
            "place_not_found": lambda name: f"Location {name} was not found. Use the exact place name returned by tool results",
            "subway_not_found": lambda city: f"Metro data not found for city {city}; falling back to walking / taxi suggestions",
        },
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.cfg = cfg or {}
        self.fields = self.LANG_FIELDS.get(self.language, self.LANG_FIELDS["en"])
        self.city_db_root = resolve_city_db_root(self.cfg)
        sample_db_path = str(self.cfg.get("sample_db_path", "")).strip()
        self.sample_db_path = Path(sample_db_path) if sample_db_path else None
        self.sample_local_city = ""
        if self.sample_db_path:
            meta_path = self.sample_db_path / ".build_meta.json"
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
                dest = meta.get("sample_signature", {}).get("dest", [])
                if isinstance(dest, list) and dest:
                    self.sample_local_city = str(dest[0]).strip()
            except Exception:
                self.sample_local_city = ""

    def _normalize_preference(self, value: object) -> str:
        raw = str(value or "").strip()
        if not raw:
            return "balanced"
        return PREFERENCE_ALIASES.get(raw, PREFERENCE_ALIASES.get(raw.lower(), "balanced"))

    def _parse_departure_time(self, value: object) -> tuple[datetime | None, str | None, bool]:
        raw = str(value or "").strip()
        if not raw:
            return None, None, False
        for fmt in DEPARTURE_TIME_FORMATS:
            try:
                parsed = datetime.strptime(raw, fmt)
            except ValueError:
                continue
            return parsed, parsed.strftime("%Y-%m-%d"), "%H:%M" in fmt
        raise ValueError("Invalid departure_time. Supported formats: YYYY-MM-DD or YYYY-MM-DD HH:MM")

    def _weather_context(self, city_name: str, date_text: str | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if self.city_db_root is None or not date_text:
            return None, None
        for row in load_city_weather(self.city_db_root, city_name):
            if str(row.get("date", "")).strip() != date_text:
                continue
            return classify_weather_condition(row), build_weather_advisory(row)
        return None, None

    def _localize_pricing_rule(self, rule: str, city_name: str) -> str:
        if str(rule) == "Walking has no transport fare.":
            return rule
        pricing_city = normalize_pricing_city(city_name)
        if "taxi fare estimate" in str(rule):
            start_fare, included_km, per_km = TAXI_BASE_FARE.get(pricing_city, DEFAULT_TAXI_FARE)
            city_label = "Hong Kong" if pricing_city == "Hong Kong" else (city_name or "the city")
            return (
                f"{city_label} taxi fare estimate: base fare {start_fare:.0f} CNY "
                f"covers the first {included_km:g} km, then {per_km:g} CNY/km."
            )
        return rule

    def _find_place(self, city_name: str, place_name: str) -> dict[str, str] | None:
        sample_rows = self._load_sample_locations(city_name)
        sample_match = find_place_by_alias(
            sample_rows,
            place_name,
            city_name,
            self._load_sample_location_aliases(city_name),
        )
        if sample_match is not None:
            return sample_match
        city_rows = load_city_locations(self.city_db_root, city_name) if self.city_db_root is not None else []
        if self.city_db_root is not None:
            city_rows.extend(load_city_location_entities(self.city_db_root, city_name))
            city_rows.extend(load_city_subway_stations(self.city_db_root, city_name))
        city_aliases = load_city_location_aliases(self.city_db_root, city_name) if self.city_db_root is not None else {}
        city_match = find_place_by_alias(city_rows, place_name, city_name, city_aliases)
        if city_match is not None:
            return city_match
        subway_rows = load_city_subway_stations(self.city_db_root, city_name) if self.city_db_root is not None else []
        return find_place_by_alias(subway_rows, place_name, city_name, city_aliases)

    def _load_sample_location_aliases(self, city_name: str) -> dict[str, str]:
        if self.sample_db_path is None:
            return {}
        if self.sample_local_city and city_name != self.sample_local_city:
            return {}
        return read_location_aliases(self.sample_db_path / "locations" / "location_aliases.csv")

    def _load_sample_locations(self, city_name: str) -> list[dict[str, str]]:
        if self.sample_db_path is None:
            return []
        if self.sample_local_city and city_name != self.sample_local_city:
            return []
        csv_path = self.sample_db_path / "locations" / "locations_coords.csv"
        if not csv_path.exists():
            return []
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle))
        except Exception:
            return []

    def _load_sample_transportation(self, city_name: str) -> list[dict[str, str]]:
        if self.sample_db_path is None:
            return []
        if self.sample_local_city and city_name != self.sample_local_city:
            return []
        csv_path = self.sample_db_path / "transportation" / "distance_matrix.csv"
        if not csv_path.exists():
            return []
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle))
        except Exception:
            return []

    def _road_candidate(self, city_name: str, origin_row: dict[str, str], destination_row: dict[str, str], direct_distance: float) -> dict[str, Any]:
        origin = f"{origin_row['latitude']},{origin_row['longitude']}"
        destination = f"{destination_row['latitude']},{destination_row['longitude']}"
        rows = self._load_sample_transportation(city_name)
        if self.city_db_root is not None:
            rows.extend(load_city_transportation(self.city_db_root, city_name))
        for row in rows:
            if row.get("origin") == origin and row.get("destination") == destination:
                is_walking = to_float(row.get("cost")) <= 0
                distance_meters = int(round(to_float(row.get("distance_meters"), direct_distance)))
                cost, currency, pricing_rule = (
                    (0.0, local_currency(city_name), "Walking has no transport fare.")
                    if is_walking
                    else estimate_taxi_price(city_name, distance_meters)
                )
                return {
                    "mode": "walking" if is_walking else "taxi",
                    "distance_meters": distance_meters,
                    "duration_minutes": int(round(to_float(row.get("duration_minutes"), minutes_from_distance(direct_distance, 30.0)))),
                    "cost": cost,
                    "currency": currency,
                    "pricing_rule": self._localize_pricing_rule(pricing_rule, city_name),
                }
        fallback_mode = "walking" if direct_distance <= 1500 else "taxi"
        cost, currency, pricing_rule = (
            (0.0, local_currency(city_name), "Walking has no transport fare.")
            if fallback_mode == "walking"
            else estimate_taxi_price(city_name, direct_distance)
        )
        return {
            "mode": fallback_mode,
            "distance_meters": int(round(direct_distance)),
            "duration_minutes": minutes_from_distance(direct_distance, WALKING_SPEED_KMH if fallback_mode == "walking" else 30.0),
            "cost": cost,
            "currency": currency,
            "pricing_rule": self._localize_pricing_rule(pricing_rule, city_name),
        }

    def _nearest_station(self, lines: list[dict[str, Any]], lat: float, lon: float) -> dict[str, Any] | None:
        nearest = None
        best_distance = None
        for line in lines:
            for station in line["stations"]:
                distance = haversine_meters(lat, lon, station["lat"], station["lon"])
                if best_distance is None or distance < best_distance:
                    nearest = {
                        "name": station["name"],
                        "lat": station["lat"],
                        "lon": station["lon"],
                        "distance_meters": distance,
                    }
                    best_distance = distance
        return nearest

    def _build_graph(self, lines: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        stations: dict[str, dict[str, Any]] = {}
        graph: dict[str, list[dict[str, Any]]] = {}
        for line in lines:
            line_name = str(line.get("name", "")).strip()
            station_list = line.get("stations", [])
            for station in station_list:
                stations.setdefault(
                    station["name"],
                    {"name": station["name"], "lat": station["lat"], "lon": station["lon"], "lines": set()},
                )
                stations[station["name"]]["lines"].add(line_name)
            for current, nxt in zip(station_list, station_list[1:]):
                edge_distance = haversine_meters(current["lat"], current["lon"], nxt["lat"], nxt["lon"])
                graph.setdefault(current["name"], []).append({"to": nxt["name"], "line": line_name, "distance_meters": edge_distance})
                graph.setdefault(nxt["name"], []).append({"to": current["name"], "line": line_name, "distance_meters": edge_distance})
        return stations, graph

    def _shortest_station_path(self, graph: dict[str, list[dict[str, Any]]], start: str, end: str) -> list[dict[str, Any]]:
        if start == end:
            return []

        start_state = (start, "")
        heap: list[tuple[tuple[int, int, float], str, str]] = [((0, 0, 0.0), start, "")]
        best_cost = {start_state: (0, 0, 0.0)}
        prev: dict[tuple[str, str], tuple[tuple[str, str], str, float]] = {}
        end_state: tuple[str, str] | None = None

        while heap:
            cost, node, current_line = heapq.heappop(heap)
            state = (node, current_line)
            if cost != best_cost.get(state):
                continue
            if node == end:
                end_state = state
                break
            for edge in graph.get(node, []):
                edge_line = str(edge.get("line", "")).strip()
                edge_distance = edge["distance_meters"]
                transfer_count = 1 if current_line and edge_line != current_line else 0
                next_cost = (
                    cost[0] + transfer_count,
                    cost[1] + 1,
                    cost[2] + edge_distance,
                )
                target = edge["to"]
                next_state = (target, edge_line)
                if next_state not in best_cost or next_cost < best_cost[next_state]:
                    best_cost[next_state] = next_cost
                    prev[next_state] = (state, edge_line, edge_distance)
                    heapq.heappush(heap, (next_cost, target, edge_line))

        if end_state is None:
            return []

        path: list[dict[str, Any]] = []
        current_state = end_state
        while current_state != start_state:
            prev_state, line_name, edge_distance = prev[current_state]
            path.append(
                {
                    "from": prev_state[0],
                    "to": current_state[0],
                    "line": line_name,
                    "distance_meters": edge_distance,
                }
            )
            current_state = prev_state
        path.reverse()
        return path

    def _choose_mode(
        self,
        *,
        preference: str,
        direct_distance: float,
        road: dict[str, Any],
        metro: dict[str, Any] | None,
    ) -> str:
        walk_threshold = 2200.0

        if direct_distance <= walk_threshold and preference != "min_walking":
            return "walking"

        if metro is None:
            return "taxi" if road["mode"] == "taxi" else "walking"

        metro_walk = metro["walk_start_meters"] + metro["walk_end_meters"]
        if metro["same_station"]:
            return "walking"

        if preference == "budget_first":
            if direct_distance <= 3000:
                return "walking"
            return "subway" if metro["cost"] <= road["cost"] and metro_walk <= 3000 else "taxi"

        if preference == "min_walking":
            return "subway" if metro_walk <= 1200 and metro["duration_minutes"] <= road["duration_minutes"] + 20 else "taxi"

        if preference == "time_first":
            if road["mode"] == "taxi" and road["duration_minutes"] <= metro["duration_minutes"] + 8:
                return "taxi"
            return "subway"

        if metro["duration_minutes"] <= road["duration_minutes"] + 12 and metro_walk <= 2600:
            return "subway"

        return "walking" if direct_distance <= walk_threshold else "taxi"

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)

        city_name = str(params.get("city", "")).strip()
        origin_place = str(params.get("origin_place", "")).strip()
        destination_place = str(params.get("destination_place", "")).strip()
        preference = self._normalize_preference(params.get("traveler_preference"))
        try:
            departure_dt, weather_date, has_departure_time = self._parse_departure_time(params.get("departure_time"))
        except ValueError as exc:
            return str(exc)

        has_sample_locations = bool(self._load_sample_locations(city_name))
        has_city_locations = (
            bool(load_city_locations(self.city_db_root, city_name))
            or bool(load_city_location_entities(self.city_db_root, city_name))
        ) if self.city_db_root is not None else False
        if not has_sample_locations and not has_city_locations:
            if self.city_db_root is None:
                return self.fields["missing_city_db"]
            return self.fields["city_not_found"](city_name)

        origin_row = self._find_place(city_name, origin_place)
        if origin_row is None:
            return self.fields["place_not_found"](origin_place)

        destination_row = self._find_place(city_name, destination_place)
        if destination_row is None:
            return self.fields["place_not_found"](destination_place)

        origin_lat = to_float(origin_row.get("latitude"))
        origin_lon = to_float(origin_row.get("longitude"))
        destination_lat = to_float(destination_row.get("latitude"))
        destination_lon = to_float(destination_row.get("longitude"))
        direct_distance = haversine_meters(origin_lat, origin_lon, destination_lat, destination_lon)
        origin_coordinates = coordinate_text(origin_row)
        destination_coordinates = coordinate_text(destination_row)
        origin_name = str(origin_row.get("poi_name") or origin_place).strip() or origin_place
        destination_name = str(destination_row.get("poi_name") or destination_place).strip() or destination_place

        road = self._road_candidate(city_name, origin_row, destination_row, direct_distance)

        subway_payload = load_city_subway(self.city_db_root, city_name) if self.city_db_root is not None else None
        lines = normalize_subway_payload(subway_payload or {})
        metro = None
        fallback_reason = None

        if lines:
            stations, graph = self._build_graph(lines)
            nearest_start = self._nearest_station(lines, origin_lat, origin_lon)
            nearest_end = self._nearest_station(lines, destination_lat, destination_lon)
            if nearest_start and nearest_end:
                same_station = nearest_start["name"] == nearest_end["name"]
                station_path = self._shortest_station_path(graph, nearest_start["name"], nearest_end["name"])
                if same_station:
                    fallback_reason = "same_station"
                elif station_path:
                    metro_distance = haversine_meters(
                        nearest_start["lat"],
                        nearest_start["lon"],
                        nearest_end["lat"],
                        nearest_end["lon"],
                    )
                    edge_distance = sum(edge["distance_meters"] for edge in station_path)
                    metro_edge_line_path = [edge["line"] for edge in station_path]
                    metro_line_path = ordered_unique_texts(metro_edge_line_path)
                    metro_cost, metro_currency, metro_pricing_rule = calculate_metro_price(
                        city_name,
                        metro_distance,
                        metro_edge_line_path,
                    )
                    metro_duration = (
                        minutes_from_distance(nearest_start["distance_meters"], WALKING_SPEED_KMH)
                        + minutes_from_distance(metro_distance, METRO_SPEED_KMH)
                        + minutes_from_distance(nearest_end["distance_meters"], WALKING_SPEED_KMH)
                    )
                    metro = {
                        "mode": "subway",
                        "start_station": nearest_start,
                        "end_station": nearest_end,
                        "station_path": [nearest_start["name"]] + [edge["to"] for edge in station_path],
                        "line_path": metro_line_path,
                        "edge_line_path": metro_edge_line_path,
                        "station_hops": len(station_path),
                        "walk_start_meters": int(round(nearest_start["distance_meters"])),
                        "walk_end_meters": int(round(nearest_end["distance_meters"])),
                        "metro_distance_meters": int(round(metro_distance)),
                        "path_distance_meters": int(round(edge_distance)),
                        "duration_minutes": metro_duration,
                        "cost": metro_cost,
                        "currency": metro_currency,
                        "pricing_rule": self._localize_pricing_rule(metro_pricing_rule, city_name),
                        "same_station": False,
                    }
                else:
                    fallback_reason = "no_path"
        else:
            fallback_reason = "no_subway_data"

        if metro is None and fallback_reason == "same_station":
            metro = {
                "mode": "walking",
                "start_station": None,
                "end_station": None,
                "station_path": [],
                "line_path": [],
                "edge_line_path": [],
                "station_hops": 0,
                "walk_start_meters": int(round(direct_distance)),
                "walk_end_meters": 0,
                "metro_distance_meters": 0,
                "path_distance_meters": 0,
                "duration_minutes": minutes_from_distance(direct_distance, WALKING_SPEED_KMH),
                "cost": 0,
                "same_station": True,
            }

        chosen_mode = self._choose_mode(
            preference=preference,
            direct_distance=direct_distance,
            road=road,
            metro=metro,
        )

        if chosen_mode == "subway" and metro is not None:
            final_duration = metro["duration_minutes"]
            final_cost = metro["cost"]
            final_currency = metro["currency"]
            details = {
                "start_station": metro["start_station"]["name"],
                "end_station": metro["end_station"]["name"],
                "station_path": metro["station_path"],
                "line_path": metro["line_path"],
                "edge_line_path": metro["edge_line_path"],
                "station_hops": metro["station_hops"],
                "walk_start_meters": metro["walk_start_meters"],
                "walk_end_meters": metro["walk_end_meters"],
                "metro_distance_meters": metro["metro_distance_meters"],
                "path_distance_meters": metro["path_distance_meters"],
                "pricing_rule": metro["pricing_rule"],
            }
        elif chosen_mode == "walking":
            final_duration = minutes_from_distance(direct_distance, WALKING_SPEED_KMH)
            final_cost = 0
            final_currency = local_currency(city_name)
            details = {
                "distance_meters": int(round(direct_distance)),
                "reason": "distance_short" if direct_distance <= 2200 else fallback_reason or "walking_preferred",
            }
        else:
            final_duration = road["duration_minutes"]
            final_cost = road["cost"]
            final_currency = road["currency"]
            details = {
                "distance_meters": road["distance_meters"],
                "estimated_mode": "taxi",
                "fallback_reason": fallback_reason,
                "pricing_rule": road["pricing_rule"],
            }

        estimated_arrival_time = None
        if departure_dt is not None and has_departure_time:
            estimated_arrival_time = (departure_dt + timedelta(minutes=final_duration)).strftime("%Y-%m-%d %H:%M")
        weather, weather_advisory = self._weather_context(city_name, weather_date)

        result = {
            "city": city_name,
            "origin_place": origin_name,
            "destination_place": destination_name,
            "origin_coordinates": origin_coordinates,
            "destination_coordinates": destination_coordinates,
            "direct_distance_meters": int(round(direct_distance)),
            "recommended_mode": chosen_mode,
            "estimated_duration_minutes": final_duration,
            "estimated_cost": final_cost,
            "currency": final_currency,
            "estimated_arrival_time": estimated_arrival_time,
            "traveler_preference": preference,
            "weather": weather,
            "weather_advisory": weather_advisory,
            "details": details,
        }

        if fallback_reason == "no_subway_data":
            result["note"] = self.fields["subway_not_found"](city_name)

        return self.format_result_as_json(result)
