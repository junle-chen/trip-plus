"""Materialize per-query sample databases for generated queries."""

from __future__ import annotations

import csv
import difflib
import json
import math
import re
import shutil
from pathlib import Path
from threading import Lock
from typing import Iterable, Optional

from query_generation.city_database import LOCAL_DATA_FILES, ROUTE_DATA_FILES, RouteOption


ROUTE_STATION_CANONICAL_OVERRIDES = {
    ("Nanchang", "code:HOG"): "Nanchang South Station",
}

LOCATION_HEADERS = ["poi_name", "latitude", "longitude", "address", "poi_type"]
LOCATION_ALIAS_HEADERS = ["alias", "canonical_name", "entity_type", "source"]
TRANSPORT_HEADERS = ["origin", "destination", "distance_meters", "duration_minutes", "cost"]
WALKING_DISTANCE_METERS = 1200
DEFAULT_TAXI_PRICING = (10.0, 3.0, 2.2)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CITY_LOCATION_ROWS_CACHE: dict[tuple[object, ...], list[dict[str, str]]] = {}
_CITY_SUBWAY_STATION_ROWS_CACHE: dict[tuple[object, ...], list[dict[str, str]]] = {}
_JSON_CACHE_LOCK = Lock()
_JSON_CACHE: dict[tuple[str, int, int], object] = {}


def _read_json(path: Path) -> object:
    try:
        stat = path.stat()
        key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    except OSError:
        return json.loads(path.read_text(encoding="utf-8"))

    with _JSON_CACHE_LOCK:
        cached = _JSON_CACHE.get(key)
    if cached is not None:
        return cached

    loaded = json.loads(path.read_text(encoding="utf-8"))
    with _JSON_CACHE_LOCK:
        stale_keys = [old_key for old_key in _JSON_CACHE if old_key[0] == key[0] and old_key != key]
        for old_key in stale_keys:
            _JSON_CACHE.pop(old_key, None)
        _JSON_CACHE[key] = loaded
    return loaded


def _read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {csv_path}")
        return list(reader.fieldnames), list(reader)


def _file_signature(path: Path) -> dict[str, object] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _file_signature_key(path: Path) -> tuple[str, int, int] | None:
    signature = _file_signature(path)
    if signature is None:
        return None
    return (str(path), int(signature["size"]), int(signature["mtime_ns"]))


def _merge_headers(existing_headers: list[str] | None, incoming_headers: list[str]) -> list[str]:
    if existing_headers is None:
        return list(incoming_headers)

    merged_headers = list(existing_headers)
    for header in incoming_headers:
        if header not in merged_headers:
            merged_headers.append(header)
    return merged_headers


def _row_key(headers: list[str], row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(header, "") for header in headers)


def _write_csv(csv_path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: _row_key(headers, item)):
            writer.writerow({header: row.get(header, "") for header in headers})

def _slugify_city_name(city_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", city_name.lower()).strip("_")
    return slug or "unknown_city"


def _folder_candidates(city_name: str, mapped_folder: Optional[str]) -> list[str]:
    candidates: list[str] = []
    for candidate in (mapped_folder, str(city_name).strip(), _slugify_city_name(str(city_name).strip())):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _to_float(value: object) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _format_coord(lat: float, lon: float) -> str:
    return f"{lat:.6f},{lon:.6f}"


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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


def _estimate_duration_minutes(distance_meters: float) -> int:
    if distance_meters <= 0:
        return 0
    speed_kmh = 5.0 if distance_meters <= WALKING_DISTANCE_METERS else 30.0
    return max(5, int(round((distance_meters / 1000.0) / speed_kmh * 60)))


def _estimate_cost(distance_meters: float) -> int:
    if distance_meters <= WALKING_DISTANCE_METERS:
        return 0
    start_fare, included_km, per_km = DEFAULT_TAXI_PRICING
    distance_km = distance_meters / 1000.0
    return int(round(start_fare + max(0.0, distance_km - included_km) * per_km))


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        value = str(value or "").strip()
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def _has_latin_letter(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", str(text or "")))


def _city_prefixed_name(city_name: str, place_name: str) -> str:
    city = str(city_name or "").strip()
    place = str(place_name or "").strip()
    if not city:
        return place
    if not place:
        return city
    if place.startswith(city) or place.replace(" ", "").startswith(city.replace(" ", "")):
        return place
    separator = " " if _has_latin_letter(city) or _has_latin_letter(place) else ""
    return f"{city}{separator}{place}"


def _station_address(city_name: str, address: object, poi_type: object = "", poi_name: object = "") -> str:
    city = str(city_name or "").strip()
    text = str(address or "").strip()
    if not _has_latin_letter(city):
        return text or f"{city}交通枢纽"

    replacements = {
        "交通枢纽": " transport hub",
        "地铁站": " subway station",
        "火车站": " railway station",
        "机场": " airport",
    }
    cleaned = text
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and not CJK_RE.search(cleaned):
        return cleaned

    poi_type_lower = str(poi_type or "").lower()
    poi_name_lower = str(poi_name or "").lower()
    if "subway" in poi_type_lower:
        return f"{city} subway station"
    if "airport" in poi_type_lower or "airport" in poi_name_lower:
        return f"{city} airport"
    return f"{city} transport hub"


def _station_name_variants(station_name: str, city_name: str = "", station_code: str = "") -> list[str]:
    name = str(station_name or "").strip()
    city = str(city_name or "").strip()
    code = str(station_code or "").strip()
    candidates: list[str] = []

    def bare_direction_hub_alias(value: str) -> bool:
        if not _has_latin_letter(value):
            return False
        words = set(re.findall(r"[a-z]+", value.lower()))
        return bool(words) and words.issubset({"north", "south", "east", "west", "station", "railway", "train"})

    if name:
        candidates.append(name)
        if city and not name.startswith(city):
            candidates.append(_city_prefixed_name(city, name))
        if city and name.startswith(city) and len(name) > len(city):
            remainder = name[len(city):].strip()
            if len(remainder) >= 4 and not bare_direction_hub_alias(remainder):
                candidates.append(remainder)
        if name.endswith("国际机场"):
            short_airport = name[:-4] + "机场"
            candidates.append(short_airport)
            if city and short_airport.startswith(city) and len(short_airport) > len(city):
                short_remainder = short_airport[len(city):].strip()
                if len(short_remainder) >= 4 and not bare_direction_hub_alias(short_remainder):
                    candidates.append(short_remainder)
            if city and not short_airport.startswith(city):
                candidates.append(_city_prefixed_name(city, short_airport))
        elif name.endswith("机场") and city and name.startswith(city) and len(name) > len(city):
            remainder = name[len(city):].strip()
            if len(remainder) >= 4 and not bare_direction_hub_alias(remainder):
                candidates.append(remainder)
        if name.endswith("站") and city and name.startswith(city) and len(name) > len(city):
            remainder = name[len(city):].strip()
            if len(remainder) >= 4 and not bare_direction_hub_alias(remainder):
                candidates.append(remainder)
        if not name.endswith(("站", "机场")):
            if _has_latin_letter(name):
                if not re.search(r"\b(station|airport)\b", name, re.I):
                    candidates.append(f"{name} Station")
                    candidates.append(f"{name} Railway Station")
            else:
                candidates.append(f"{name}站")

    if code and len(code) >= 3:
        candidates.append(code)
    return _dedupe_preserve_order(candidates)


def _subway_station_variants(station_name: str, city_name: str = "") -> list[str]:
    candidates = _station_name_variants(station_name, city_name)
    for name in list(candidates):
        if name.endswith("国际机场"):
            candidates.append(name[:-4] + "机场")
            candidates.append(name[:-4])
        elif name.endswith("机场"):
            candidates.append(name[:-2])
        if name.endswith("站"):
            candidates.append(name[:-1])
    return _dedupe_preserve_order(candidates)


def _load_city_location_rows(
    city_db_root: Path,
    city_to_folder: dict[str, str],
    city_name: str,
) -> list[dict[str, str]]:
    for folder in _folder_candidates(city_name, city_to_folder.get(city_name)):
        csv_path = city_db_root / folder / "locations" / "locations_coords.csv"
        if csv_path.exists():
            signature_key = _file_signature_key(csv_path)
            cache_key = ("locations", city_name, signature_key)
            cached = _CITY_LOCATION_ROWS_CACHE.get(cache_key)
            if cached is not None:
                return cached
            _, rows = _read_csv_rows(csv_path)
            _CITY_LOCATION_ROWS_CACHE[cache_key] = rows
            return rows
    return []


def _load_city_subway_station_rows(
    city_db_root: Path,
    city_to_folder: dict[str, str],
    city_name: str,
) -> list[dict[str, str]]:
    subway_paths = [
        city_db_root / folder / "subway" / "subway.json"
        for folder in _folder_candidates(city_name, city_to_folder.get(city_name))
    ]
    signature_key = tuple(
        key for key in (_file_signature_key(path) for path in subway_paths if path.exists()) if key is not None
    )
    if not signature_key:
        return []
    cache_key = ("subway", city_name, signature_key)
    cached = _CITY_SUBWAY_STATION_ROWS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for subway_path in subway_paths:
        if not subway_path.exists():
            continue
        try:
            payload = _read_json(subway_path)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue
        line_payloads = payload.get("lines") if isinstance(payload.get("lines"), list) else payload.get("l", [])
        if not isinstance(line_payloads, list):
            continue
        for line in line_payloads:
            if not isinstance(line, dict):
                continue
            station_payloads = line.get("stations") if isinstance(line.get("stations"), list) else line.get("st", [])
            if not isinstance(station_payloads, list):
                continue
            for station in station_payloads:
                if not isinstance(station, dict):
                    continue
                name = str(station.get("name") or station.get("n") or "").strip()
                lon = station.get("longitude")
                lat = station.get("latitude")
                position = str(station.get("position") or station.get("sl") or "").strip()
                if position and "," in position:
                    lon, lat = [part.strip() for part in position.split(",", 1)]
                lat_value = _to_float(lat)
                lon_value = _to_float(lon)
                if not name or lat_value is None or lon_value is None:
                    continue
                key = (name, f"{lat_value:.6f}", f"{lon_value:.6f}")
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "poi_name": name,
                        "latitude": f"{lat_value:.6f}",
                        "longitude": f"{lon_value:.6f}",
                        "address": _station_address(city_name, "", "subway_station", name),
                        "poi_type": "subway_station",
                        "source": "city_subway",
                    }
                )
    _CITY_SUBWAY_STATION_ROWS_CACHE[cache_key] = rows
    return rows


def _resolve_station_location(
    city_db_root: Path,
    city_to_folder: dict[str, str],
    city_name: str,
    station_name: str,
    station_code: str = "",
) -> dict[str, str] | None:
    def plausible_transport_hub_row(row: dict[str, str]) -> bool:
        poi_type = str(row.get("poi_type", "") or row.get("type", "")).lower()
        poi_name = str(row.get("poi_name", "")).strip()
        if any(token in poi_type for token in ("hotel", "restaurant", "attraction")):
            return False
        if re.search(r"酒店|宾馆|民宿|公寓|餐厅|饭店|食堂|咖啡|茶餐厅", poi_name):
            return False
        if re.search(r"\b(hotel|restaurant|cafe|coffee|buffet|cuisine|dining|lounge|bistro|bar)\b", poi_name, re.I):
            return False
        return True

    def airport_match(candidate_name: str, row_name: str) -> bool:
        if "机场" not in candidate_name or "机场" not in row_name:
            return False
        candidate_norm = re.sub(r"(T\d+|[一二三四五六七八九]+号)?航站楼|机场站", "", candidate_name.replace("国际", ""))
        row_norm = re.sub(r"(T\d+|[一二三四五六七八九]+号)?航站楼|机场站", "", row_name.replace("国际", ""))
        if candidate_norm == "机场" or row_norm == "机场":
            return False
        return (
            candidate_norm == row_norm
            or candidate_norm.endswith(row_norm)
            or row_norm.endswith(candidate_norm)
            or candidate_norm in row_norm
            or row_norm in candidate_norm
        )

    def english_hub_core(text: str) -> str:
        value = str(text or "").lower()
        city_text = str(city_name or "").lower()
        value = re.sub(r"\bcoastal\b", " binhai ", value)
        value = re.sub(r"\bbinbei\b", " binhai north ", value)
        value = re.sub(r"\bterminal\s*\d+\b|\bt\d+\b", " ", value)
        value = re.sub(r"\binternational\b|\brailway\b|\brailroad\b|\btrain\b|\bstation\b", " ", value)
        value = re.sub(r"[^a-z0-9]+", "", value)
        city_key = re.sub(r"[^a-z0-9]+", "", city_text)
        if city_key and value.startswith(city_key) and len(value) > len(city_key) + 2:
            value = value[len(city_key):]
        return value

    def near_english_hub_core(candidate_core: str, row_core: str) -> bool:
        if not candidate_core or not row_core:
            return False
        shorter = min(candidate_core, row_core, key=len)
        longer = max(candidate_core, row_core, key=len)
        if len(shorter) < 6:
            return False
        common_prefix = 0
        for left, right in zip(shorter, longer):
            if left != right:
                break
            common_prefix += 1
        return common_prefix >= 6 and difflib.SequenceMatcher(None, shorter, longer).ratio() >= 0.80

    def mentions_other_latin_city(text: str) -> bool:
        value = str(text or "").lower()
        current_city = str(city_name or "").lower()
        if not _has_latin_letter(value):
            return False
        for known_city in city_to_folder:
            known = str(known_city or "").strip().lower()
            if not known or known == current_city or not _has_latin_letter(known):
                continue
            if re.search(rf"\b{re.escape(known)}\b", value):
                return True
        return False

    def english_airport_match(candidate_name: str, row_name: str) -> bool:
        candidate_lower = str(candidate_name or "").lower()
        row_lower = str(row_name or "").lower()
        if "airport" not in candidate_lower or "airport" not in row_lower:
            return False
        if mentions_other_latin_city(candidate_name):
            return False
        candidate_core = english_hub_core(re.sub(r"\bairport\b", " ", str(candidate_name), flags=re.I))
        row_core = english_hub_core(re.sub(r"\bairport\b", " ", str(row_name), flags=re.I))
        if not candidate_core or not row_core:
            return False
        return (
            candidate_core == row_core
            or candidate_core.endswith(row_core)
            or row_core.endswith(candidate_core)
            or candidate_core in row_core
            or row_core in candidate_core
            or near_english_hub_core(candidate_core, row_core)
            or difflib.SequenceMatcher(None, candidate_core, row_core).ratio() >= 0.86
        )

    def english_station_match(candidate_name: str, row_name: str) -> bool:
        candidate_lower = str(candidate_name or "").lower()
        row_lower = str(row_name or "").lower()
        if not any(token in candidate_lower for token in ("station", "railway", "north", "south", "east", "west")):
            return False
        if mentions_other_latin_city(candidate_name):
            return False
        candidate_core = english_hub_core(candidate_name)
        row_core = english_hub_core(row_name)
        if not candidate_core or not row_core:
            return False
        generic_direction_cores = {"north", "south", "east", "west"}
        if candidate_core in generic_direction_cores or row_core in generic_direction_cores:
            return candidate_core == row_core
        direction_suffixes = ("north", "south", "east", "west")
        candidate_direction = next((direction for direction in direction_suffixes if candidate_core.endswith(direction)), "")
        row_direction = next((direction for direction in direction_suffixes if row_core.endswith(direction)), "")
        if candidate_direction and row_direction != candidate_direction and candidate_core != row_core:
            return False
        if not any(token in row_lower for token in ("station", "railway")) and candidate_core != row_core:
            return False
        return (
            candidate_core == row_core
            or candidate_core.endswith(row_core)
            or row_core.endswith(candidate_core)
            or candidate_core in row_core
            or row_core in candidate_core
        )

    exact_variants = set(_station_name_variants(station_name, city_name, station_code))
    location_rows = _load_city_location_rows(city_db_root, city_to_folder, city_name)

    # Prefer exact names before fuzzy hub matching; otherwise a nearby station
    # such as "Shenzhen Airport Railway Station" can steal an airport match
    # before the exact airport row later in the file is seen.
    for row in location_rows:
        poi_name = str(row.get("poi_name", "")).strip()
        lat = _to_float(row.get("latitude"))
        lon = _to_float(row.get("longitude"))
        if lat is None or lon is None:
            continue
        if poi_name in exact_variants:
            return {
                "poi_name": poi_name,
                "latitude": f"{lat:.6f}",
                "longitude": f"{lon:.6f}",
                "address": row.get("address", ""),
                "poi_type": row.get("poi_type", "station") or "station",
            }

    for row in location_rows:
        poi_name = str(row.get("poi_name", "")).strip()
        lat = _to_float(row.get("latitude"))
        lon = _to_float(row.get("longitude"))
        if lat is None or lon is None:
            continue
        is_hub_match = plausible_transport_hub_row(row) and any(
            airport_match(variant, poi_name)
            or english_airport_match(variant, poi_name)
            or english_station_match(variant, poi_name)
            for variant in exact_variants
        )
        if is_hub_match:
            return {
                "poi_name": poi_name,
                "latitude": f"{lat:.6f}",
                "longitude": f"{lon:.6f}",
                "address": row.get("address", ""),
                "poi_type": row.get("poi_type", "station") or "station",
            }

    subway_variants = set(_subway_station_variants(station_name, city_name))
    for row in _load_city_subway_station_rows(city_db_root, city_to_folder, city_name):
        poi_name = str(row.get("poi_name", "")).strip()
        if (
            poi_name in subway_variants
            or any(airport_match(variant, poi_name) for variant in subway_variants)
            or any(english_airport_match(variant, poi_name) for variant in subway_variants)
            or any(english_station_match(variant, poi_name) for variant in subway_variants)
            or any(
                variant and "站" in poi_name and poi_name.startswith(f"{variant}站")
                for variant in subway_variants
                if "站" not in variant
            )
        ):
            return row

    def fallback_current_city_airport() -> dict[str, str] | None:
        if "airport" not in str(station_name or "").lower():
            return None
        airport_rows: list[dict[str, str]] = []
        city_key = re.sub(r"[^a-z0-9]+", "", str(city_name or "").lower())
        for row in location_rows:
            poi_name = str(row.get("poi_name", "")).strip()
            poi_lower = poi_name.lower()
            if "airport" not in poi_lower or not plausible_transport_hub_row(row):
                continue
            if any(token in poi_lower for token in ("terminal", "under construction")):
                continue
            if city_key and city_key not in re.sub(r"[^a-z0-9]+", "", poi_lower):
                continue
            lat = _to_float(row.get("latitude"))
            lon = _to_float(row.get("longitude"))
            if lat is None or lon is None:
                continue
            airport_rows.append(row)
        if len(airport_rows) != 1:
            return None
        row = airport_rows[0]
        return {
            "poi_name": str(row.get("poi_name", "")).strip(),
            "latitude": f"{_to_float(row.get('latitude')):.6f}",
            "longitude": f"{_to_float(row.get('longitude')):.6f}",
            "address": row.get("address", ""),
            "poi_type": row.get("poi_type", "station") or "station",
        }

    fallback_airport = fallback_current_city_airport()
    if fallback_airport:
        return fallback_airport

    return None


def _is_subway_station_location(row: dict[str, str]) -> bool:
    poi_type = str(row.get("poi_type", "") or row.get("type", "")).lower()
    source = str(row.get("source", "")).lower()
    return "subway" in poi_type or source == "city_subway"


def _latin_route_name_is_good_canonical(city_name: str, station_name: str) -> bool:
    raw_name = str(station_name or "").strip()
    if not raw_name or not _has_latin_letter(raw_name):
        return False
    raw_lower = raw_name.lower()
    if any(token in raw_lower for token in ("airport", "railway station", "train station", " station")):
        return True
    city_key = re.sub(r"[^a-z0-9]+", "", str(city_name or "").lower())
    raw_key = re.sub(r"[^a-z0-9]+", "", raw_lower)
    return bool(
        city_key
        and raw_key.startswith(city_key)
        and any(direction in raw_lower.split() for direction in ("north", "south", "east", "west"))
    )


def _canonical_station_name(city_name: str, station_name: str, resolved: dict[str, str]) -> str:
    raw_name = str(station_name or "").strip()
    city = str(city_name or "").strip()
    resolved_name = str(resolved.get("poi_name", "")).strip()
    if not raw_name:
        return resolved_name

    if _latin_route_name_is_good_canonical(city, raw_name):
        return raw_name

    if _is_subway_station_location(resolved):
        return _city_prefixed_name(city, raw_name) if city and not raw_name.startswith(city) else raw_name

    if "机场" in raw_name and resolved_name and (
        "航站楼" in resolved_name or "机场站" in resolved_name
    ):
        return _city_prefixed_name(city, raw_name) if city and not raw_name.startswith(city) else raw_name

    return resolved_name or raw_name


def _append_location_alias(
    alias_rows: list[dict[str, str]],
    seen_aliases: set[tuple[str, str]],
    alias: str,
    canonical_name: str,
    entity_type: str,
    source: str,
) -> None:
    alias = str(alias or "").strip()
    canonical_name = str(canonical_name or "").strip()
    if not alias or not canonical_name or alias == canonical_name:
        return
    key = (alias, canonical_name)
    if key in seen_aliases:
        return
    seen_aliases.add(key)
    alias_rows.append(
        {
            "alias": alias,
            "canonical_name": canonical_name,
            "entity_type": str(entity_type or "station").strip() or "station",
            "source": str(source or "transport_hub").strip() or "transport_hub",
        }
    )


def _station_location_aliases(
    *,
    city_name: str,
    station_name: str,
    station_code: str,
    canonical_name: str,
    entity_type: str,
    source: str,
    alias_rows: list[dict[str, str]],
    seen_aliases: set[tuple[str, str]],
) -> None:
    variants = _station_name_variants(station_name, city_name, station_code)
    if canonical_name and canonical_name not in variants:
        variants.append(canonical_name)
    for alias in variants:
        _append_location_alias(
            alias_rows,
            seen_aliases,
            alias,
            canonical_name,
            entity_type,
            source,
        )


def _parse_segment_index(row: dict[str, str]) -> int:
    try:
        return int(float(str(row.get("segment_index", "")).strip() or "1"))
    except ValueError:
        return 1


def _route_max_segment_by_index(route_rows: Iterable[dict[str, str]]) -> dict[str, int]:
    max_segment_by_route: dict[str, int] = {}
    for row in route_rows:
        route_index = str(row.get("route_index", "")).strip()
        segment_index = _parse_segment_index(row)
        max_segment_by_route[route_index] = max(max_segment_by_route.get(route_index, segment_index), segment_index)
    return max_segment_by_route


def _endpoint_station_refs_for_row(
    row: dict[str, str],
    max_segment_by_route: dict[str, int],
) -> list[dict[str, str]]:
    """Return only route endpoint stations, not intermediate transfer hubs.

    In the intercity CSVs, ``origin_city`` and ``destination_city`` describe
    the whole route, while each row may be only one segment of a connecting
    itinerary. Treating every segment departure/arrival as belonging to those
    endpoint cities creates false aliases such as a Shanghai transfer airport
    being resolved inside Shenzhen. Only the first segment departure and the
    last segment arrival are local endpoint hubs.
    """
    route_index = str(row.get("route_index", "")).strip()
    segment_index = _parse_segment_index(row)
    max_segment_index = max_segment_by_route.get(route_index, segment_index)
    refs: list[dict[str, str]] = []
    if segment_index == 1:
        city_name = str(row.get("origin_city", "")).strip()
        station_name = str(row.get("dep_station_name", "")).strip()
        station_code = str(row.get("dep_station_code", "")).strip()
        if city_name and station_name:
            refs.append({"city_name": city_name, "station_name": station_name, "station_code": station_code, "prefix": "dep"})
    if segment_index == max_segment_index:
        city_name = str(row.get("destination_city", "")).strip()
        station_name = str(row.get("arr_station_name", "")).strip()
        station_code = str(row.get("arr_station_code", "")).strip()
        if city_name and station_name:
            refs.append({"city_name": city_name, "station_name": station_name, "station_code": station_code, "prefix": "arr"})
    return refs


def _extract_station_refs(route_rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    route_rows = list(route_rows)
    max_segment_by_route = _route_max_segment_by_index(route_rows)
    refs: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in route_rows:
        for ref in _endpoint_station_refs_for_row(row, max_segment_by_route):
            city_name = ref["city_name"]
            station_name = ref["station_name"]
            station_code = ref["station_code"]
            key = (city_name, station_name, station_code)
            if key in seen:
                continue
            seen.add(key)
            refs.append({"city_name": city_name, "station_name": station_name, "station_code": station_code})
    return refs


def _route_station_ref_key(ref: dict[str, str]) -> tuple[str, str]:
    city_name = str(ref.get("city_name", "")).strip()
    station_code = str(ref.get("station_code", "")).strip()
    station_name = str(ref.get("station_name", "")).strip()
    if station_code:
        return city_name, f"code:{station_code}"
    return city_name, f"name:{station_name}"


def _route_station_name_score(city_name: str, station_name: str) -> tuple[int, int, int, str]:
    name = str(station_name or "").strip()
    lower = name.lower()
    compact_name = re.sub(r"[^a-z0-9]+", "", lower)
    city_key = re.sub(r"[^a-z0-9]+", "", str(city_name or "").lower())
    direction_words = {"north", "south", "east", "west"}
    words = set(re.findall(r"[a-z0-9]+", lower))
    has_hub_token = any(token in lower for token in ("airport", "railway station", "train station", " station"))
    has_city = bool(city_key and compact_name.startswith(city_key))
    is_generic = lower in {"airport", "station", "railway station", "train station"} or words.issubset(direction_words | {"station"})
    if "airport" in lower:
        hub_rank = 0
    elif has_hub_token:
        hub_rank = 1
    elif has_city and words.intersection(direction_words):
        hub_rank = 2
    else:
        hub_rank = 3
    city_prefix_penalty = 1 if has_city and "airport" in lower and len(compact_name) > len(city_key) + 3 else 0
    return (1 if is_generic else 0, hub_rank, city_prefix_penalty + len(compact_name), compact_name)


def _preferred_route_station_names(refs: Iterable[dict[str, str]]) -> dict[tuple[str, str], str]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for ref in refs:
        station_name = str(ref.get("station_name", "")).strip()
        if not station_name:
            continue
        grouped.setdefault(_route_station_ref_key(ref), []).append(station_name)

    preferred: dict[tuple[str, str], str] = {}
    for key, names in grouped.items():
        override_name = ROUTE_STATION_CANONICAL_OVERRIDES.get(key)
        if override_name:
            preferred[key] = override_name
            continue
        if not any(_has_latin_letter(name) for name in names):
            continue
        city_name = key[0]
        unique_names = _dedupe_preserve_order(names)
        unique_names = [
            name
            for name in unique_names
            if not _has_latin_letter(name) or _latin_route_name_is_good_canonical(city_name, name)
        ]
        if not unique_names:
            continue
        preferred[key] = min(unique_names, key=lambda name: _route_station_name_score(city_name, name))
    return preferred


def _station_location_rows_for_routes(
    city_db_root: Path,
    city_to_folder: dict[str, str],
    city_name: str,
    route_rows: Iterable[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    alias_rows: list[dict[str, str]] = []
    seen_names: set[str] = set()
    seen_aliases: set[tuple[str, str]] = set()

    def append_station_row(name: str, resolved: dict[str, str]) -> None:
        name = str(name or "").strip()
        if not name or name in seen_names:
            return
        seen_names.add(name)
        rows.append(
            {
                "poi_name": name,
                "latitude": resolved["latitude"],
                "longitude": resolved["longitude"],
                "address": _station_address(
                    city_name,
                    resolved.get("address", ""),
                    resolved.get("poi_type", "station") or "station",
                    name,
                ),
                "poi_type": resolved.get("poi_type", "station") or "station",
            }
        )

    refs = _extract_station_refs(route_rows)
    preferred_names = _preferred_route_station_names(refs)
    for ref in refs:
        if ref["city_name"] != city_name:
            continue
        resolved = _resolve_station_location(
            city_db_root,
            city_to_folder,
            city_name,
            ref["station_name"],
            ref["station_code"],
        )
        if resolved is None:
            continue
        canonical_name = preferred_names.get(_route_station_ref_key(ref)) or _canonical_station_name(
            city_name, ref["station_name"], resolved
        )
        append_station_row(canonical_name, resolved)
        _station_location_aliases(
            city_name=city_name,
            station_name=ref["station_name"],
            station_code=ref["station_code"],
            canonical_name=canonical_name,
            entity_type=resolved.get("poi_type", "station") or "station",
            source="transport_hub",
            alias_rows=alias_rows,
            seen_aliases=seen_aliases,
        )
    return rows, alias_rows

def _ensure_regular_file(path: Path) -> None:
    if not path.exists() or not path.is_symlink():
        return
    payload = path.read_bytes()
    path.unlink()
    path.write_bytes(payload)


def _write_location_aliases(temp_dir: Path, alias_rows: list[dict[str, str]]) -> None:
    if not alias_rows:
        return
    alias_csv = temp_dir / "locations" / "location_aliases.csv"
    _ensure_regular_file(alias_csv)
    if alias_csv.exists():
        alias_headers, existing_alias_rows = _read_csv_rows(alias_csv)
    else:
        alias_headers, existing_alias_rows = LOCATION_ALIAS_HEADERS, []
    alias_headers = _merge_headers(alias_headers, LOCATION_ALIAS_HEADERS)

    merged_rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    seen_alias_targets: dict[str, str] = {}
    for row in [*existing_alias_rows, *alias_rows]:
        alias = str(row.get("alias", "")).strip()
        canonical_name = str(row.get("canonical_name", "")).strip()
        if not alias or not canonical_name or alias == canonical_name:
            continue
        existing_target = seen_alias_targets.get(alias)
        if existing_target and existing_target != canonical_name:
            continue
        key = (alias, canonical_name)
        if key in seen:
            continue
        seen.add(key)
        seen_alias_targets[alias] = canonical_name
        merged_rows.append({header: row.get(header, "") for header in alias_headers})

    if merged_rows:
        _write_csv(alias_csv, alias_headers, merged_rows)


def _augment_locations_and_transport(
    temp_dir: Path,
    station_rows: list[dict[str, str]],
    alias_rows: list[dict[str, str]] | None = None,
) -> None:
    alias_rows = alias_rows or []

    locations_csv = temp_dir / "locations" / "locations_coords.csv"
    transportation_csv = temp_dir / "transportation" / "distance_matrix.csv"
    _ensure_regular_file(locations_csv)

    if locations_csv.exists():
        location_headers, location_rows = _read_csv_rows(locations_csv)
    else:
        location_headers, location_rows = LOCATION_HEADERS, []
    location_headers = _merge_headers(location_headers, LOCATION_HEADERS)

    def transport_location_like(row: dict[str, str]) -> bool:
        poi_type = str(row.get("poi_type", "") or row.get("type", "")).lower()
        poi_name = str(row.get("poi_name", "")).strip()
        if any(token in poi_type for token in ("hotel", "restaurant", "attraction")):
            return False
        if re.search(r"酒店|宾馆|民宿|公寓|餐厅|饭店|食堂|咖啡|茶餐厅", poi_name):
            return False
        if re.search(r"\b(hotel|restaurant|cafe|coffee|buffet|cuisine|dining|lounge|bistro|bar)\b", poi_name, re.I):
            return False
        return (
            any(token in poi_type for token in ("station", "airport", "subway", "railway", "transport"))
            or bool(re.search(r"机场|火车站|高铁站|地铁站|客运站", poi_name))
            or bool(re.search(r"\b(airport|railway station|train station|station)\b", poi_name, re.I))
        )

    alias_to_canonical = {
        str(row.get("alias", "")).strip(): str(row.get("canonical_name", "")).strip()
        for row in alias_rows
        if str(row.get("alias", "")).strip()
        and str(row.get("canonical_name", "")).strip()
        and str(row.get("alias", "")).strip() != str(row.get("canonical_name", "")).strip()
    }
    canonicalized_station_rows: list[dict[str, str]] = []
    seen_station_rows: set[tuple[str, str, str]] = set()
    for row in station_rows:
        station_row = dict(row)
        poi_name = str(station_row.get("poi_name", "")).strip()
        canonical = alias_to_canonical.get(poi_name)
        if canonical:
            station_row["poi_name"] = canonical
        key = (
            str(station_row.get("poi_name", "")).strip(),
            str(station_row.get("latitude", "")).strip(),
            str(station_row.get("longitude", "")).strip(),
        )
        if key in seen_station_rows:
            continue
        seen_station_rows.add(key)
        canonicalized_station_rows.append(station_row)
    station_rows = canonicalized_station_rows

    station_names = {str(row.get("poi_name", "")).strip() for row in station_rows if str(row.get("poi_name", "")).strip()}
    existing_names_before_prune = {str(row.get("poi_name", "")).strip() for row in location_rows}
    pruned_location_rows: list[dict[str, str]] = []
    original_location_row_count = len(location_rows)
    for row in location_rows:
        poi_name = str(row.get("poi_name", "")).strip()
        canonical = alias_to_canonical.get(poi_name)
        if (
            canonical
            and canonical != poi_name
            and transport_location_like(row)
            and (canonical in station_names or canonical in existing_names_before_prune)
        ):
            continue
        pruned_location_rows.append(row)
    location_rows = pruned_location_rows
    location_rows_changed = len(location_rows) != original_location_row_count

    existing_names = {str(row.get("poi_name", "")).strip() for row in location_rows}
    added_rows: list[dict[str, str]] = []
    for row in station_rows:
        poi_name = str(row.get("poi_name", "")).strip()
        lat = _to_float(row.get("latitude"))
        lon = _to_float(row.get("longitude"))
        if not poi_name or lat is None or lon is None or poi_name in existing_names:
            continue
        normalized_row = {header: row.get(header, "") for header in location_headers}
        normalized_row["poi_name"] = poi_name
        normalized_row["latitude"] = f"{lat:.6f}"
        normalized_row["longitude"] = f"{lon:.6f}"
        normalized_row["poi_type"] = normalized_row.get("poi_type", "") or "station"
        location_rows.append(normalized_row)
        added_rows.append(normalized_row)
        existing_names.add(poi_name)

    if location_rows_changed or added_rows:
        _write_csv(locations_csv, location_headers, location_rows)

    local_alias_rows = [
        row
        for row in alias_rows
        if str(row.get("canonical_name", "")).strip() in existing_names
    ]
    _write_location_aliases(temp_dir, local_alias_rows)

    if not added_rows:
        return

    _ensure_regular_file(transportation_csv)
    if transportation_csv.exists():
        transport_headers, transport_rows = _read_csv_rows(transportation_csv)
    else:
        transport_headers, transport_rows = TRANSPORT_HEADERS, []
    transport_headers = _merge_headers(transport_headers, TRANSPORT_HEADERS)
    existing_edges = {
        (str(row.get("origin", "")).strip(), str(row.get("destination", "")).strip())
        for row in transport_rows
    }

    all_coords: dict[str, tuple[float, float]] = {}
    for row in location_rows:
        lat = _to_float(row.get("latitude"))
        lon = _to_float(row.get("longitude"))
        if lat is None or lon is None:
            continue
        all_coords.setdefault(_format_coord(lat, lon), (lat, lon))
    new_coords = {
        _format_coord(_to_float(row.get("latitude")) or 0.0, _to_float(row.get("longitude")) or 0.0)
        for row in added_rows
        if _to_float(row.get("latitude")) is not None and _to_float(row.get("longitude")) is not None
    }

    for origin, (origin_lat, origin_lon) in all_coords.items():
        for destination, (destination_lat, destination_lon) in all_coords.items():
            if origin == destination:
                continue
            if origin not in new_coords and destination not in new_coords:
                continue
            if (origin, destination) in existing_edges:
                continue
            distance = _haversine_meters(origin_lat, origin_lon, destination_lat, destination_lon)
            transport_rows.append(
                {
                    "origin": origin,
                    "destination": destination,
                    "distance_meters": str(int(round(distance))),
                    "duration_minutes": str(_estimate_duration_minutes(distance)),
                    "cost": str(_estimate_cost(distance)),
                }
            )
            existing_edges.add((origin, destination))

    _write_csv(transportation_csv, transport_headers, transport_rows)


def _load_city_name_to_folder(city_db_root: Path) -> dict[str, str]:
    city_index = _read_json(city_db_root / "city_index.json")
    if not isinstance(city_index, dict):
        raise ValueError(f"Invalid city index format: {city_db_root / 'city_index.json'}")

    name_to_folder: dict[str, str] = {}
    for key, info in city_index.items():
        if not isinstance(info, dict):
            continue
        city_name = str(info.get("city_name", "")).strip()
        folder_name = str(info.get("folder_name", key)).strip()
        if city_name:
            name_to_folder[city_name] = folder_name
        if folder_name:
            name_to_folder.setdefault(folder_name, folder_name)
        if key:
            name_to_folder.setdefault(str(key).strip(), folder_name or str(key).strip())

    # Some checked-in databases may have stale folder_name metadata. Only fill
    # missing mappings from explicit city columns in the on-disk CSV files;
    # never overwrite a mapping that already resolves to an existing folder.
    actual_dirs = [path for path in city_db_root.iterdir() if path.is_dir() and not path.name.startswith(".")]
    for folder in actual_dirs:
        inferred_names: set[str] = set()
        for rel_path in (
            ("hotels", "hotels.csv"),
            ("attractions", "attractions.csv"),
            ("restaurants", "restaurants.csv"),
            ("locations", "locations_coords.csv"),
        ):
            csv_path = folder / rel_path[0] / rel_path[1]
            if not csv_path.exists():
                continue
            try:
                headers, rows = _read_csv_rows(csv_path)
            except Exception:
                continue

            for row in rows[:50]:
                for field in ("city", "destination_city", "origin_city"):
                    value = str(row.get(field, "")).strip()
                    if value:
                        inferred_names.add(value)

            if inferred_names:
                break

        for city_name in inferred_names:
            existing_folder = name_to_folder.get(city_name)
            if existing_folder and (city_db_root / existing_folder).exists():
                continue
            name_to_folder[city_name] = folder.name
        name_to_folder.setdefault(folder.name, folder.name)
    return name_to_folder
def _copy_local_city_files(city_db_root: Path, dest_folder: str, target_dir: Path) -> None:
    city_dir = city_db_root / dest_folder
    for _, (subdir, filename) in LOCAL_DATA_FILES.items():
        src = city_dir / subdir / filename
        if src.exists():
            dst = target_dir / subdir / filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def materialize_generated_sample_database(
    city_db_root: Path,
    option: RouteOption,
    sample_id: str,
    db: dict[str, list[dict[str, str]]],
    route_headers: dict[str, list[str]],
    output_root: Path,
) -> Path:
    target_dir = output_root / f"id_{sample_id}"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    _copy_local_city_files(city_db_root, option.dest_folder, target_dir)

    default_route_headers = {
        "trains": [
            "origin_city", "destination_city", "dep_date", "dep_station_code", "dep_station_name",
            "arr_station_code", "arr_station_name", "dep_datetime", "arr_datetime", "duration",
            "train_no", "train_type", "seat_class", "price", "segment_index",
            "route_index", "sample_id",
        ],
        "flights": [
            "origin_city", "destination_city", "dep_date", "dep_airport_code", "dep_airport_name",
            "arr_airport_code", "arr_airport_name", "dep_datetime", "arr_datetime", "duration",
            "flight_no", "airline", "aircraft_type", "seat_class", "price",
            "segment_index", "route_index", "sample_id",
        ],
    }

    for category, (subdir, filename) in ROUTE_DATA_FILES.items():
        rows = [dict(row, sample_id=str(sample_id)) for row in db[category]]
        headers = list(route_headers.get(category) or default_route_headers[category])
        if "sample_id" not in headers:
            headers.append("sample_id")
        _write_csv(target_dir / subdir / filename, headers, rows)

    city_to_folder = _load_city_name_to_folder(city_db_root)
    route_rows_for_station_lookup = [
        dict(row)
        for category in ROUTE_DATA_FILES
        for row in db.get(category, [])
    ]
    station_rows, station_alias_rows = _station_location_rows_for_routes(
        city_db_root,
        city_to_folder,
        option.dest_city,
        route_rows_for_station_lookup,
    )
    _augment_locations_and_transport(target_dir, station_rows, station_alias_rows)

    return target_dir
