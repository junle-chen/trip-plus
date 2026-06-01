"""
Utility functions for travel planning evaluation.
Contains common helper functions for parsing, validation, and data loading.
"""

import re
import csv
import json
import math
import os
import unicodedata
from functools import lru_cache
from itertools import product
from datetime import datetime, time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


# ----------------------
# String Parsing Utilities
# ----------------------

def extract_from_to(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract 'from' and 'to' cities from text like 'from CityA to CityB'."""
    if not isinstance(text, str):
        return None, None
    m = re.search(r"from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # English plans often write the day header as "CityA to CityB" without the
    # leading "from". Treat this as the same intercity state instead of making
    # every cross-city day fail continuity checks.
    m = re.search(r"^\s*(.+?)\s+to\s+([^,]+?)\s*$", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    return None, None


def normalize_city(text: Optional[str]) -> Optional[str]:
    """Normalize city name by removing parentheses and content inside."""
    if text is None:
        return None
    return re.sub(r"[（(].*?[)）]", "", text).strip()


def normalize_entity_name(text: Optional[str]) -> str:
    """
    Normalize entity names for robust matching across conversion runs.

    This tolerates harmless formatting drift introduced by the conversion LLM,
    such as full-width punctuation and accidental whitespace insertion.
    """
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text)).strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1].strip()
    else:
        normalized = normalized.lstrip("[").rstrip("]").strip()
    normalized = re.sub(
        r"\s*[,，]\s*(?:free\s+entry|free\s+admission|no\s+ticket|required\s+ticket|ticket\s*[:：]?\s*[\d.]+|[¥￥]\s*[\d.]+(?:\s*/\s*person)?)\s*$",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip()
    normalized = re.sub(
        r"\s*\(\s*\d+\s*(?:rooms?|room)\s*\)\s*$",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip()
    return re.sub(r"\s+", "", normalized)


def normalize_entity_key(text: Optional[str]) -> str:
    """Normalize an entity name for stable internal map keys."""
    return normalize_entity_name(text).casefold()


def looks_like_price_text(text: Optional[str]) -> bool:
    """Return True when text is clearly a price/rate rather than an entity name."""
    value = str(text or "").strip()
    if not value:
        return False
    return bool(
        re.search(
            r"(?:\brmb\b|\bcny\b|/room|/night|\d+\s*yuan(?:\b|/))",
            value,
            flags=re.IGNORECASE,
        )
    )


def looks_like_generic_hotel_activity_text(text: Optional[str]) -> bool:
    """Return True for hotel activity labels that are not hotel entity names."""
    value = str(text or "").strip()
    if not value:
        return False
    key = re.sub(r"[^0-9a-z]+", "", unicodedata.normalize("NFKC", value).casefold())
    return key in {
        "rest",
        "hotel",
        "restathotel",
        "restinhotel",
        "hotelrest",
        "checkin",
        "checkout",
        "checkinrest",
        "checkoutrest",
        "returntohotel",
        "backtohotel",
        "stayathotel",
        "overnight",
        "overnightstay",
    }


def extract_hotel_name_from_activity(details: Dict[str, Any]) -> str:
    """Extract hotel name from a hotel activity, tolerating price-only names.

    Some conversion outputs put the room rate in ``details.name`` and the real
    hotel name in ``details.activity`` (for example, "Check-in, Xining Hotel").
    This helper uses that fallback only for price-like names so normal entity
    matching stays exact.
    """
    name = str((details or {}).get("name") or "").strip()
    if name and not looks_like_price_text(name) and not looks_like_generic_hotel_activity_text(name):
        return name

    activity = str((details or {}).get("activity") or (details or {}).get("description") or "").strip()
    if not activity:
        return "" if looks_like_generic_hotel_activity_text(name) else name
    candidates: List[str] = []
    if "," in activity:
        candidates.append(activity.rsplit(",", 1)[-1].strip())
    if "，" in activity:
        candidates.append(activity.rsplit("，", 1)[-1].strip())
    match = re.search(
        r"(?:check-?in(?: at)?|stay at|return to|arrive at)\s*[,，:]?\s*(.+)$",
        activity,
        flags=re.IGNORECASE,
    )
    if match:
        candidates.append(match.group(1).strip())
    for candidate in candidates:
        cleaned = re.sub(r"\s*\(.*?\)\s*$", "", candidate).strip()
        if (
            cleaned
            and not looks_like_price_text(cleaned)
            and not looks_like_generic_hotel_activity_text(cleaned)
        ):
            return cleaned
    return "" if looks_like_generic_hotel_activity_text(name) else name


def extract_entity_name(activity: Dict[str, Any], entity_type: Optional[str] = None) -> str:
    """Extract a canonical display name from a converted plan activity/entity.

    Conversion outputs are mostly normalized to ``details.name``, but older or
    slightly different conversion prompts may emit typed keys such as
    ``restaurant_name``, ``attraction_name``, or ``hotel_name``.  Evaluators
    should use this helper so hard checks, soft checks, and diagnostics do not
    disagree on the same converted plan.
    """
    if not isinstance(activity, dict):
        return ""
    details = activity.get("details") if isinstance(activity.get("details"), dict) else {}
    direct = activity
    key_map = {
        "restaurant": ("name", "restaurant_name"),
        "attraction": ("name", "attraction_name"),
        "hotel": ("name", "hotel_name"),
    }
    if entity_type in key_map:
        keys = key_map[entity_type]
    else:
        keys = (
            "name",
            "restaurant_name",
            "attraction_name",
            "hotel_name",
            "to",
            "from",
            "activity",
            "description",
        )
    for key in keys:
        value = details.get(key)
        if value:
            return str(value).strip()
        value = direct.get(key)
        if value:
            return str(value).strip()
    return ""


def add_index_entry(index: Dict[str, Dict[str, Any]], name: str, payload: Dict[str, Any]) -> None:
    """Store both raw and normalized lookup keys for a database entity."""
    raw_name = (name or "").strip()
    if not raw_name:
        return
    payload.setdefault("_index_entity_name", raw_name)
    index[raw_name] = payload
    normalized_name = normalize_entity_name(raw_name)
    if normalized_name and normalized_name not in index:
        index[normalized_name] = payload


def add_unique_comma_prefix_aliases(index: Dict[str, Dict[str, Any]]) -> None:
    """Add aliases for converter-truncated names only when the prefix is unique."""
    candidates: Dict[str, List[Dict[str, Any]]] = {}
    seen_canonical: set[str] = set()
    for record in list(index.values()):
        canonical_name = str(record.get("_index_entity_name") or "").strip()
        if not canonical_name or canonical_name in seen_canonical:
            continue
        seen_canonical.add(canonical_name)
        for sep in (",", "，"):
            if sep not in canonical_name:
                continue
            prefix = canonical_name.split(sep, 1)[0].strip()
            prefix_norm = normalize_entity_name(prefix)
            if len(prefix_norm) < 8:
                continue
            for alias in {prefix, prefix_norm}:
                if alias:
                    candidates.setdefault(alias, []).append(record)

    for alias, records in candidates.items():
        unique_records = {
            str(record.get("_index_entity_name") or ""): record
            for record in records
        }
        if len(unique_records) == 1 and alias not in index:
            index[alias] = next(iter(unique_records.values()))


def _infer_language_from_database_path(path: Path) -> Optional[str]:
    parts = path.parts
    for idx, part in enumerate(parts):
        if part == "en":
            return part
        if part == "sample" and idx + 1 < len(parts) and parts[idx + 1] == "en":
            return parts[idx + 1]
    return None


def _infer_city_from_database_path(path: Path, language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    parts = path.parts
    for idx, part in enumerate(parts):
        if part == language and idx + 1 < len(parts):
            candidate = parts[idx + 1]
            if candidate and not candidate.startswith("id_"):
                return candidate
    return None


def _city_alias_record(category_aliases: Dict[str, Any], city_key: str, name_key: str) -> Optional[Dict[str, Any]]:
    if city_key:
        city_aliases = category_aliases.get(city_key)
        if isinstance(city_aliases, dict) and isinstance(city_aliases.get(name_key), dict):
            return city_aliases[name_key]
        compact_city_key = city_key.replace("_", "")
        for alias_city_key, city_aliases in category_aliases.items():
            if (
                isinstance(alias_city_key, str)
                and alias_city_key.replace("_", "") == compact_city_key
                and isinstance(city_aliases, dict)
                and isinstance(city_aliases.get(name_key), dict)
            ):
                return city_aliases[name_key]
    unique_matches = [
        city_aliases[name_key]
        for city_aliases in category_aliases.values()
        if isinstance(city_aliases, dict) and isinstance(city_aliases.get(name_key), dict)
    ]
    if len(unique_matches) == 1:
        return unique_matches[0]
    return None


@lru_cache(maxsize=1)
def _load_dedupe_price_aliases() -> Dict[str, Any]:
    alias_path = get_base_dir() / "evaluation" / "resources" / "dedupe_price_aliases.json"
    if not alias_path.exists():
        alias_path = get_base_dir() / "database" / "dedupe_price_aliases.json"
    if not alias_path.exists():
        return {}
    try:
        data = json.loads(alias_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    aliases = data.get("aliases")
    return aliases if isinstance(aliases, dict) else {}


def get_dedupe_allowed_prices(csv_path: str, category: str, name: str, city: Optional[str] = None) -> List[str]:
    """Return price alternatives recorded before duplicate DB rows were merged.

    This keeps evaluation stable for plans generated before database cleanup:
    if a model used a price that existed in a now-merged duplicate row, the
    price check should not become a new false negative solely due to cleanup.
    """
    lang = _infer_language_from_database_path(Path(csv_path))
    if not lang:
        return []
    category_aliases = (
        _load_dedupe_price_aliases()
        .get(lang, {})
        .get(category, {})
    )
    if not isinstance(category_aliases, dict):
        return []
    name_key = normalize_entity_key(name)
    city_key = normalize_entity_key(city) or _infer_city_from_database_path(Path(csv_path), lang)
    record = _city_alias_record(category_aliases, city_key, name_key)
    if not isinstance(record, dict):
        return []
    prices = record.get("prices")
    if not isinstance(prices, list):
        return []
    return [str(price) for price in prices if str(price).strip()]


def get_index_record(index: Dict[str, Dict[str, Any]], name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Look up a database record by raw or normalized entity name."""
    if not index or not name:
        return None
    raw_name = str(name).strip()
    if raw_name in index:
        return index[raw_name]
    record = index.get(normalize_entity_name(raw_name))
    if record:
        return record
    cleaned = re.sub(
        r"\s*\(\s*\d+\s*(?:rooms?|room)\s*\)\s*$",
        "",
        raw_name.lstrip("[").rstrip("]").strip(),
        flags=re.IGNORECASE,
    ).strip()
    if cleaned and cleaned != raw_name:
        return index.get(cleaned) or index.get(normalize_entity_name(cleaned))
    return None


def parse_lonlat_string(text: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """Parse coordinate string in format 'latitude,longitude', returns (lat, lon)."""
    if not text or not isinstance(text, str):
        return None, None
    m = re.match(r"\s*([\-0-9\.]+)\s*,\s*([\-0-9\.]+)\s*$", text)
    if not m:
        return None, None
    try:
        # Database format is "latitude,longitude"
        lat = float(m.group(1))
        lon = float(m.group(2))
        return lat, lon
    except Exception:
        return None, None


# ----------------------
# Time Parsing Utilities
# ----------------------

def parse_time_hhmm(t: Optional[str]) -> Optional[time]:
    """
    Parse time string to time object.
    Special case: "24:00" is treated as end of day and mapped to 23:59.
    """
    if not t or not isinstance(t, str):
        return None
    t = t.strip()
    if t == "24:00":
        # Map 24:00 to 23:59 (end of day)
        return time(23, 59)
    try:
        dt = datetime.strptime(t, "%H:%M")
        return time(dt.hour, dt.minute)
    except Exception:
        return None


def parse_time_slot(slot: Optional[str]) -> Tuple[Optional[time], Optional[time]]:
    """Parse time slot string (e.g., '09:00-17:00') to time objects."""
    if not slot or not isinstance(slot, str):
        return None, None
    m = re.match(r"\s*(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\s*", slot)
    if not m:
        return None, None
    start = parse_time_hhmm(m.group(1))
    end = parse_time_hhmm(m.group(2))
    return start, end


def is_within_business_hours(slot_start: time, slot_end: time, open_t: time, close_t: time) -> bool:
    """
    Check if activity time slot is within business hours.
    Handles midnight crossover (e.g., 16:30-03:00).
    """
    slot_crosses_midnight = slot_end < slot_start
    if open_t <= close_t:
        # Normal same-day interval: activity time must be fully within business hours
        if slot_crosses_midnight:
            # Activity crosses midnight but business hours don't: invalid
            return False
        return (slot_start >= open_t) and (slot_end <= close_t)
    # Crosses midnight: business hours are [open, 24:00) ∪ [00:00, close]
    if slot_crosses_midnight:
        # Both activity and business hours cross midnight: must start in night segment and end in morning segment
        return (slot_start >= open_t) and (slot_end <= close_t)
    # Activity doesn't cross midnight, but business hours do: can be in night segment or morning segment
    in_night = (slot_start >= open_t) and (slot_end >= open_t)
    in_morning = (slot_start <= close_t) and (slot_end <= close_t)
    return in_night or in_morning


def slot_to_minutes(slot: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Convert time slot to minutes since midnight."""
    start_t, end_t = parse_time_slot(slot)
    if not start_t or not end_t:
        return None, None
    start_m = start_t.hour * 60 + start_t.minute
    end_m = end_t.hour * 60 + end_t.minute
    if end_m < start_m:
        end_m += 24 * 60  # Handle midnight crossover
    return start_m, end_m


# ----------------------
# Geographic Utilities
# ----------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates in kilometers using Haversine formula."""
    from math import radians, sin, cos, asin, sqrt
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


# ----------------------
# Database Path Management
# ----------------------

def get_base_dir() -> Path:
    """Get repository root directory."""
    
    # Method 0: Read from environment variable (for evaluation with specific sample database)
    env_db_path = os.environ.get('EVAL_DATABASE_PATH')
    if env_db_path:
        db_path = Path(env_db_path)
        if db_path.exists():
            # EVAL_DATABASE_PATH points to a sample database; return the repository root.
            # Since we use _BASE_DIR / "database" / ... format later, need special handling
            return db_path.parent.parent
    
    # Method 1: Parse from __file__ (most reliable)
    try:
        current_file = Path(__file__).resolve()
        base_dir = current_file.parent.parent  # evaluation -> repository root
        if (base_dir / "database").exists():
            return base_dir
    except (NameError, AttributeError):
        pass
    
    # Method 2: Search from current working directory
    cwd = Path.cwd()
    # Check current directory
    if (cwd / "database").exists():
        return cwd
    # Check parent directory
    if (cwd.parent / "database").exists():
        return cwd.parent
    # Check parent's parent directory (if running from evaluation directory)
    if (cwd.parent.parent / "database").exists():
        return cwd.parent.parent
    
    # Method 3: If not found, try from common locations
    # Default: return result parsed from __file__ (if available)
    try:
        return Path(__file__).resolve().parent.parent
    except (NameError, AttributeError):
        # Last fallback: return current directory
        return cwd


def get_database_dir(database_dir: Optional[Path] = None) -> Path:
    """
    Get database directory, supports reading specific sample database path from environment variable or parameter.
    
    Args:
        database_dir: Directly specified database directory path (highest priority)
    """
    
    # 1. Use directly passed parameter first
    if database_dir is not None:
        if isinstance(database_dir, str):
            database_dir = Path(database_dir)
        if database_dir.exists():
            return database_dir
    
    # 2. Read from environment variable (for evaluation with specific sample database)
    env_db_path = os.environ.get('EVAL_DATABASE_PATH')
    if env_db_path:
        db_path = Path(env_db_path)
        if db_path.exists():
            # EVAL_DATABASE_PATH already points to database/id_x/, return directly
            return db_path
    
    # 3. Default to _BASE_DIR / "database" / "id_0"
    return get_base_dir() / "database" / "id_0"


# ----------------------
# Data Loading Utilities
# ----------------------

def load_restaurant_index(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """Load restaurant index from CSV file."""
    index: Dict[str, Dict[str, Any]] = {}
    path_obj = Path(csv_path)
    if not path_obj.exists():
        # If file doesn't exist, return empty index; upper layer checks will provide failure reason
        return {}
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:  # Use utf-8-sig to handle BOM
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("restaurant_name") or "").strip()
                if not name:
                    continue
                payload = {
                    "price_per_person": row.get("price_per_person"),
                    "opening_time": row.get("opening_time"),
                    "closing_time": row.get("closing_time"),
                }
                allowed_prices = get_dedupe_allowed_prices(csv_path, "restaurants", name, row.get("city"))
                if allowed_prices:
                    payload["allowed_prices"] = allowed_prices
                add_index_entry(index, name, payload)
    except Exception:
        # If loading fails, return empty index; upper layer checks will provide failure reason
        return {}
    add_unique_comma_prefix_aliases(index)
    return index


def load_hotel_index(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """Load hotel index from CSV file."""
    index: Dict[str, Dict[str, Any]] = {}
    path_obj = Path(csv_path)
    if not path_obj.exists():
        return {}
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:  # Use utf-8-sig to handle BOM
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                payload = {
                    "price_per_night": row.get("price"),
                    "city": row.get("city"),
                }
                allowed_prices = get_dedupe_allowed_prices(csv_path, "hotels", name, row.get("city"))
                if allowed_prices:
                    payload["allowed_prices"] = allowed_prices
                add_index_entry(index, name, payload)
    except Exception:
        return {}
    add_unique_comma_prefix_aliases(index)
    return index


def load_attraction_index(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """Load attraction index from CSV file."""
    index: Dict[str, Dict[str, Any]] = {}
    path_obj = Path(csv_path)
    if not path_obj.exists():
        return {}
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:  # Use utf-8-sig to handle BOM
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("attraction_name") or "").strip()
                if not name:
                    continue
                payload = {
                    "opening_time": row.get("opening_time"),
                    "closing_time": row.get("closing_time"),
                    "min_visit_hours": row.get("min_visit_hours"),
                    "max_visit_hours": row.get("max_visit_hours"),
                    "ticket_price": row.get("ticket_price"),
                    "latitude": row.get("latitude"),
                    "longitude": row.get("longitude"),
                    "closing_dates": row.get("closing_dates"),  # Add closing_dates field
                    "popularity_tags": row.get("popularity_tags"),
                    "crowd_risk": row.get("crowd_risk"),
                    "queue_risk": row.get("queue_risk"),
                    "peak_crowd_windows": row.get("peak_crowd_windows"),
                }
                allowed_prices = get_dedupe_allowed_prices(csv_path, "attractions", name, row.get("city"))
                if allowed_prices:
                    payload["allowed_prices"] = allowed_prices
                add_index_entry(index, name, payload)
    except Exception:
        return {}
    add_unique_comma_prefix_aliases(index)
    return index


def load_locations_index(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load locations_coords.csv, which contains coordinate information for all POIs (attractions, restaurants, hotels, etc.).
    
    Note: Keep coordinates in original string format to match format in distance_matrix.csv.
    """
    index: Dict[str, Dict[str, Any]] = {}
    path_obj = Path(csv_path)
    if not path_obj.exists():
        return {}

    def coord_precision(payload: Dict[str, Any]) -> int:
        precision = 0
        for key in ("latitude", "longitude"):
            raw = str(payload.get(key, ""))
            if "." in raw:
                precision = max(precision, len(raw.split(".", 1)[1]))
        return precision

    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:  # Use utf-8-sig to handle BOM
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("poi_name") or "").strip()
                if not name:
                    continue
                # Keep original string format, don't convert to float
                payload = {
                    "latitude": (row.get("latitude") or "").strip(),
                    "longitude": (row.get("longitude") or "").strip(),
                    "poi_type": (row.get("poi_type") or "").strip(),
                }
                existing = get_index_record(index, name)
                if existing and existing.get("latitude") and existing.get("longitude"):
                    # Some generated location files contain duplicate names with
                    # both 6-decimal distance-matrix coordinates and raw
                    # restaurant coordinates. Prefer the lower-precision entry,
                    # because distance_matrix.csv is keyed by 6-decimal strings.
                    if coord_precision(payload) >= coord_precision(existing):
                        continue
                    index[name] = payload
                    normalized_name = normalize_entity_name(name)
                    if normalized_name:
                        index[normalized_name] = payload
                    continue
                add_index_entry(index, name, payload)
    except Exception:
        return {}
    load_location_aliases_into_index(index, path_obj.parent / "location_aliases.csv")
    add_unique_comma_prefix_aliases(index)
    return index


def load_location_aliases_into_index(index: Dict[str, Dict[str, Any]], aliases_path: Path) -> None:
    """Add only explicit location aliases whose canonical target already exists.

    Runtime entity resolution must not guess aliases from suffixes or naming
    conventions. This loader accepts aliases only from `location_aliases.csv`,
    and only when `canonical_name` resolves to a primary row that was already
    loaded into the index snapshot.
    """
    if not aliases_path.exists():
        return
    base_index = dict(index)
    try:
        with aliases_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
            alias_targets = {
                (row.get("alias") or "").strip(): (row.get("canonical_name") or "").strip()
                for row in rows
                if (row.get("alias") or "").strip() and (row.get("canonical_name") or "").strip()
            }

            def resolve_payload(canonical_name: str) -> Optional[Dict[str, Any]]:
                current = canonical_name
                seen = set()
                while current and current not in seen:
                    seen.add(current)
                    payload = get_index_record(base_index, current)
                    if payload:
                        return payload
                    current = alias_targets.get(current, "")
                return None

            for row in rows:
                alias = (row.get("alias") or "").strip()
                canonical = (row.get("canonical_name") or "").strip()
                if not alias or not canonical or alias == canonical:
                    continue
                if get_index_record(base_index, alias):
                    # The alias is also a primary entity. Keeping the primary
                    # row is safer than silently rewriting it to a different
                    # canonical target at runtime.
                    continue
                payload = resolve_payload(canonical)
                if payload:
                    add_index_entry(index, alias, payload)
    except Exception:
        return


def _safe_price(value: Any) -> Optional[float]:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price) or price <= 0:
        return None
    return price


def _route_price_sets(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, ...], List[float]]:
    """Compute complete route prices from route/segment rows where possible."""
    grouped: Dict[Tuple[str, ...], Dict[int, set[float]]] = {}
    for row in rows:
        route_index = str(row.get("route_index") or "").strip()
        segment_index = row.get("segment_index")
        price = _safe_price(row.get("price"))
        if not route_index or segment_index in (None, "") or price is None:
            continue
        try:
            segment_number = int(segment_index)
        except (TypeError, ValueError):
            continue
        key = (
            str(row.get("origin_city") or "").strip(),
            str(row.get("destination_city") or "").strip(),
            str(row.get("dep_date") or "").strip(),
            route_index,
            str(row.get("seat_class") or "").strip(),
        )
        grouped.setdefault(key, {}).setdefault(segment_number, set()).add(price)

    prices_by_key: Dict[Tuple[str, ...], List[float]] = {}
    for key, segment_prices in grouped.items():
        ordered_segments = [sorted(segment_prices[idx]) for idx in sorted(segment_prices)]
        if not ordered_segments:
            continue
        totals = {
            round(sum(combo), 2)
            for combo in product(*ordered_segments)
        }
        prices_by_key[key] = sorted(totals)
    return prices_by_key


def _route_segment_key(row: Dict[str, Any]) -> Tuple[str, ...]:
    return (
        str(row.get("origin_city") or "").strip(),
        str(row.get("destination_city") or "").strip(),
        str(row.get("route_index") or "").strip(),
        str(row.get("seat_class") or "").strip(),
    )


def _route_segment_price_sets(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, ...], Dict[int, List[float]]]:
    grouped: Dict[Tuple[str, ...], Dict[int, set[float]]] = {}
    for row in rows:
        route_index = str(row.get("route_index") or "").strip()
        segment_index = row.get("segment_index")
        price = _safe_price(row.get("price"))
        if not route_index or segment_index in (None, "") or price is None:
            continue
        try:
            segment_number = int(segment_index)
        except (TypeError, ValueError):
            continue
        grouped.setdefault(_route_segment_key(row), {}).setdefault(segment_number, set()).add(price)
    return {
        key: {segment: sorted(prices) for segment, prices in segments.items()}
        for key, segments in grouped.items()
    }


def _route_price_key(row: Dict[str, Any]) -> Tuple[str, ...]:
    return (
        str(row.get("origin_city") or "").strip(),
        str(row.get("destination_city") or "").strip(),
        str(row.get("dep_date") or "").strip(),
        str(row.get("route_index") or "").strip(),
        str(row.get("seat_class") or "").strip(),
    )


@lru_cache(maxsize=512)
def load_flights_index(csv_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load flights index from CSV file.
    
    Returns:
        Dictionary with flight_no as key, list of flight records as value.
        (A flight number may have multiple records for different dates/segments)
    """
    index: Dict[str, List[Dict[str, Any]]] = {}
    path_obj = Path(csv_path)
    if not path_obj.exists():
        return {}
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
            route_prices = _route_price_sets(rows)
            route_segment_prices = _route_segment_price_sets(rows)
            for row in rows:
                flight_no = (row.get("flight_no") or "").strip()
                if not flight_no:
                    continue
                record = {
                    "origin_city": (row.get("origin_city") or "").strip(),
                    "destination_city": (row.get("destination_city") or "").strip(),
                    "dep_date": (row.get("dep_date") or "").strip(),
                    "dep_station_name": (row.get("dep_station_name") or "").strip(),
                    "arr_station_name": (row.get("arr_station_name") or "").strip(),
                    "dep_datetime": (row.get("dep_datetime") or "").strip(),
                    "arr_datetime": (row.get("arr_datetime") or "").strip(),
                    "price": row.get("price"),
                    "route_prices": route_prices.get(_route_price_key(row), []),
                    "route_segment_prices": route_segment_prices.get(_route_segment_key(row), {}),
                    "airline": (row.get("airline") or "").strip(),
                    "seat_class": (row.get("seat_class") or "").strip(),
                    "segment_index": row.get("segment_index"),
                    "route_index": row.get("route_index"),
                }
                if flight_no not in index:
                    index[flight_no] = []
                index[flight_no].append(record)
    except Exception:
        return {}
    return index


@lru_cache(maxsize=512)
def load_trains_index(csv_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load trains index from CSV file.
    
    Returns:
        Dictionary with train_no as key, list of train records as value.
        (A train number may have multiple records for different dates/segments)
    """
    index: Dict[str, List[Dict[str, Any]]] = {}
    path_obj = Path(csv_path)
    if not path_obj.exists():
        return {}
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
            route_prices = _route_price_sets(rows)
            route_segment_prices = _route_segment_price_sets(rows)
            for row in rows:
                train_no = (row.get("train_no") or "").strip()
                if not train_no:
                    continue
                record = {
                    "origin_city": (row.get("origin_city") or "").strip(),
                    "destination_city": (row.get("destination_city") or "").strip(),
                    "dep_date": (row.get("dep_date") or "").strip(),
                    "dep_station_name": (row.get("dep_station_name") or "").strip(),
                    "arr_station_name": (row.get("arr_station_name") or "").strip(),
                    "dep_datetime": (row.get("dep_datetime") or "").strip(),
                    "arr_datetime": (row.get("arr_datetime") or "").strip(),
                    "price": row.get("price"),
                    "route_prices": route_prices.get(_route_price_key(row), []),
                    "route_segment_prices": route_segment_prices.get(_route_segment_key(row), {}),
                    "train_type": (row.get("train_type") or "").strip(),
                    "seat_class": (row.get("seat_class") or "").strip(),
                    "segment_index": row.get("segment_index"),
                    "route_index": row.get("route_index"),
                }
                if train_no not in index:
                    index[train_no] = []
                index[train_no].append(record)
    except Exception:
        return {}
    return index


# ----------------------
# Station/Airport Mapping
# ----------------------

# Global cache: airport/station name to city mapping
_STATION_TO_CITY_MAP: Optional[Dict[str, str]] = None


def load_station_to_city_mapping(database_dir: Optional[Path] = None) -> Dict[str, str]:
    """
    Load airport/station to city mapping from flights.csv and trains.csv.
    
    Args:
        database_dir: Database directory path (if specified, will use that sample's database)
    
    Returns: Dictionary of {station_name: city_name}
    Example: {"Xiaoshan International Airport": "Hangzhou", "Hangzhou East Station": "Hangzhou"}
    """
    mapping: Dict[str, str] = {}
    raw_station_entries: List[Tuple[str, str]] = []
    
    # Determine database directory
    if database_dir is not None:
        db_dir = get_database_dir(database_dir)
    else:
        db_dir = get_database_dir()

    def store_station_mapping(station_text: str, city_text: str) -> None:
        mapping[station_text] = city_text
        normalized_station = normalize_entity_name(station_text)
        if normalized_station:
            mapping[normalized_station] = city_text

    def register_station(station: object, city: object) -> None:
        station_text = (str(station or "")).strip()
        city_text = normalize_city((str(city or "")).strip())
        if not station_text or not city_text:
            return
        store_station_mapping(station_text, city_text)
        raw_station_entries.append((station_text, city_text))

    def register_alias(alias: object, city: object) -> None:
        alias_text = (str(alias or "")).strip()
        city_text = normalize_city((str(city or "")).strip())
        if not alias_text or not city_text:
            return
        store_station_mapping(alias_text, city_text)
     
    # Load airport mapping from flights.csv
    flights_path = db_dir / "flights" / "flights.csv"
    if flights_path.exists():
        try:
            with open(str(flights_path), "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Departure airport
                    dep_station = (row.get("dep_station_name") or "").strip()
                    origin_city = (row.get("origin_city") or "").strip()
                    register_station(dep_station, origin_city)
                    
                    # Arrival airport
                    arr_station = (row.get("arr_station_name") or "").strip()
                    dest_city = (row.get("destination_city") or "").strip()
                    register_station(arr_station, dest_city)
        except Exception:
            pass
    
    # Load station mapping from trains.csv
    trains_path = db_dir / "trains" / "trains.csv"
    if trains_path.exists():
        try:
            with open(str(trains_path), "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Departure station
                    dep_station = (row.get("dep_station_name") or "").strip()
                    origin_city = (row.get("origin_city") or "").strip()
                    register_station(dep_station, origin_city)
                    
                    # Arrival station
                    arr_station = (row.get("arr_station_name") or "").strip()
                    dest_city = (row.get("destination_city") or "").strip()
                    register_station(arr_station, dest_city)
        except Exception:
            pass

    # Register deterministic aliases derived from route-table station names.
    # This is not fuzzy entity matching: each alias is backed by a station row
    # whose city is already known from flights.csv/trains.csv.
    airport_stations_by_city: Dict[str, set[str]] = {}
    rail_stations_by_city: Dict[str, set[str]] = {}
    for station, city in raw_station_entries:
        station_lower = station.lower()
        city_lower = city.lower()
        if city and not station_lower.startswith(city_lower):
            register_alias(f"{city} {station}", city)
        if "airport" in station_lower:
            airport_stations_by_city.setdefault(city, set()).add(station)
        if "station" in station_lower or "railway" in station_lower:
            rail_stations_by_city.setdefault(city, set()).add(station)

    for city, stations in airport_stations_by_city.items():
        if len(stations) == 1:
            register_alias(f"{city} Airport", city)
    for city, stations in rail_stations_by_city.items():
        if len(stations) == 1:
            register_alias(f"{city} Railway Station", city)
            register_alias(f"{city} Station", city)

    # Some sample DBs contain hub coordinate aliases whose names differ from
    # route-table station names but share the exact same coordinates. Register
    # only exact-coordinate aliases for hub-like names, keeping the matching
    # auditable and snapshot-local.
    locations_path = db_dir / "locations" / "locations_coords.csv"
    if locations_path.exists():
        try:
            with open(str(locations_path), "r", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            coord_city: Dict[Tuple[str, str], str] = {}
            for row in rows:
                name = (row.get("poi_name") or row.get("name") or "").strip()
                lat = (row.get("latitude") or "").strip()
                lon = (row.get("longitude") or "").strip()
                city = mapping.get(name) or mapping.get(normalize_entity_name(name))
                if name and lat and lon and city:
                    coord_city[(lat, lon)] = city
            for row in rows:
                name = (row.get("poi_name") or row.get("name") or "").strip()
                lat = (row.get("latitude") or "").strip()
                lon = (row.get("longitude") or "").strip()
                lower_name = name.lower()
                if not name or not lat or not lon:
                    continue
                if not any(token in lower_name for token in ("airport", "station", "railway")):
                    continue
                city = coord_city.get((lat, lon))
                if city:
                    register_alias(name, city)
        except Exception:
            pass

    aliases_path = db_dir / "locations" / "location_aliases.csv"
    if aliases_path.exists():
        try:
            with open(str(aliases_path), "r", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
                alias_targets = {
                    (row.get("alias") or "").strip(): (row.get("canonical_name") or "").strip()
                    for row in rows
                    if (row.get("alias") or "").strip() and (row.get("canonical_name") or "").strip()
                }

                def city_for_alias_target(canonical_name: str) -> Optional[str]:
                    current = canonical_name
                    seen = set()
                    while current and current not in seen:
                        seen.add(current)
                        city = mapping.get(current) or mapping.get(normalize_entity_name(current))
                        if city:
                            return city
                        current = alias_targets.get(current, "")
                    return None

                for row in rows:
                    alias = (row.get("alias") or "").strip()
                    canonical = (row.get("canonical_name") or "").strip()
                    city = city_for_alias_target(canonical)
                    if alias and city:
                        mapping[alias] = city
                        normalized_alias = normalize_entity_name(alias)
                        if normalized_alias:
                            mapping[normalized_alias] = city
        except Exception:
            pass
    
    return mapping


def get_station_to_city_map(database_dir: Optional[Path] = None) -> Dict[str, str]:
    """
    Get airport/station to city mapping.
    
    Args:
        database_dir: Database directory path (if specified, will use that sample's database)
    
    Note: In multi-threaded environments, global cache is not used; reloads on each call.
    """
    # If database_dir is specified, don't use cache (avoid multi-threading conflicts)
    if database_dir is not None:
        return load_station_to_city_mapping(database_dir)
    
    # Otherwise use global cache
    global _STATION_TO_CITY_MAP
    if _STATION_TO_CITY_MAP is None:
        _STATION_TO_CITY_MAP = load_station_to_city_mapping()
    return _STATION_TO_CITY_MAP


def extract_city_from_location(location: str, database_dir: Optional[Path] = None) -> Optional[str]:
    """
    Extract city name from airport/station name.
    
    Args:
        location: Airport/station name
        database_dir: Database directory path (if specified, will use that sample's database)
    
    Strategy:
    1. Look up directly in flights.csv and trains.csv mapping table.
    2. Look up explicit entries from locations/location_aliases.csv.

    This intentionally avoids heuristic city extraction. If a transport hub is
    not present in route data or explicit aliases, the evaluator should fail
    with an auditable unresolved-entity error instead of guessing.
    
    Examples:
    - "Xiaoshan International Airport" -> "Hangzhou" (from flights.csv)
    - "Hangzhou East Station" -> "Hangzhou" (from trains.csv)
    - "Beijing Daxing International Airport" -> "Beijing"
    """
    if not location:
        return None
    
    # Strategy 1: Look up in mapping table (most accurate)
    station_map = get_station_to_city_map(database_dir)
    if location in station_map:
        return station_map[location]
    normalized_location = normalize_entity_name(location)
    if normalized_location in station_map:
        return station_map[normalized_location]
    
    return None


# ----------------------
# Coordinate Resolution
# ----------------------

def get_location_coords(name: str, locations_index: Dict[str, Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """Get coordinates from locations_index (string format, preserving original precision)."""
    record = get_index_record(locations_index, name)
    if not record:
        return None, None
    lat_str = record.get("latitude")
    lon_str = record.get("longitude")
    # Verify if valid numbers (but return string)
    if not lat_str or not lon_str:
        return None, None
    try:
        float(lat_str)  # Verify convertible to number
        float(lon_str)
        return lat_str, lon_str
    except Exception:
        return None, None


def resolve_name_coords(name: str, locations_index: Optional[Dict[str, Dict[str, Any]]] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve location name to coordinates (string format, preserving original precision).
    
    Returns (lat_str, lon_str) or (None, None)
    """
    # 1) Look up directly in locations_coords.csv (contains all POI types: attractions, restaurants, hotels, etc.)
    if locations_index is not None:
        lat_str, lon_str = get_location_coords(name, locations_index)
        if lat_str is not None and lon_str is not None:
            return lat_str, lon_str
    # 2) Parse as "latitude,longitude" string
    lat_float, lon_float = parse_lonlat_string(name)
    if lat_float is not None and lon_float is not None:
        # Convert back to string (maintain reasonable precision)
        return str(lat_float), str(lon_float)
    return None, None


# ----------------------
# Weekday Calculation
# ----------------------

# ----------------------
# Duration Parsing
# ----------------------

def parse_duration_hours(val: Any) -> Optional[float]:
    """Parse duration value to hours."""
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def is_all_day(opening: Optional[str], closing: Optional[str]) -> bool:
    """Check if opening hours are all day."""
    opening = (opening or "").strip()
    closing = (closing or "").strip()
    all_day_patterns = ["Open 24 Hours"]
    return opening in all_day_patterns and closing in all_day_patterns


# ----------------------
# Date and Day of Week Utilities
# ----------------------

def calculate_day_of_week(depart_weekday: int, day_index: int) -> int:
    """
    Calculate the day of week for a given day in the trip.
    
    Args:
        depart_weekday: Day of week for departure day (1=Monday, 7=Sunday)
        day_index: Day index in the trip (0-based, 0 = first day)
    
    Returns:
        Day of week (1=Monday, 7=Sunday)
    
    Example:
        If departure is Wednesday (3), and we want day_index=1 (second day):
        calculate_day_of_week(3, 1) = 4 (Thursday)
    """
    result = depart_weekday + day_index
    # Handle wraparound: if result > 7, wrap to 1-7
    while result > 7:
        result -= 7
    return result


def parse_closing_dates(closing_dates_str: Optional[str]) -> List[int]:
    """
    Parse closing_dates string to list of day-of-week integers.
    
    Based on English database data:
    - Formats found: "Monday", "Tuesday"
    - Delimiter: English comma (,)
    - Only full day names (no abbreviations)
    
    Returns:
        List of integers where 1=Monday, 7=Sunday
    
    Examples:
        "Monday" -> [1]
        "Monday,Sunday" -> [1, 7]
        "" -> []
    """
    if not closing_dates_str or not isinstance(closing_dates_str, str):
        return []
    
    # Day name mappings (1=Monday, 7=Sunday)
    # Based on actual data: only full names, no abbreviations
    day_map = {
        "monday": 1,
        "tuesday": 2,
        "wednesday": 3,
        "thursday": 4,
        "friday": 5,
        "saturday": 6,
        "sunday": 7,
    }
    
    closing_days = []
    # Split by comma (only delimiter found in actual data)
    parts = closing_dates_str.split(',')
    
    for part in parts:
        part_stripped = part.strip()
        # Try case-insensitive match for English
        part_lower = part_stripped.lower()
        
        if part_lower in day_map:
            closing_days.append(day_map[part_lower])
    
    return sorted(list(set(closing_days)))  # Remove duplicates and sort


def is_attraction_closed_on_day(closing_dates: Optional[str], weekday: int) -> bool:
    """
    Check if an attraction is closed on a specific day of week.
    
    Args:
        closing_dates: Closing dates string from CSV (e.g., "Monday,Wednesday")
        weekday: Day of week to check (1=Monday, 7=Sunday)
    
    Returns:
        True if attraction is closed on that day, False otherwise
    """
    closed_days = parse_closing_dates(closing_dates)
    return weekday in closed_days


# ----------------------
# Activity Iteration Helpers
# ----------------------

def day_cities(current_city: str) -> List[str]:
    """Get list of cities for a given day."""
    a, b = extract_from_to(current_city)
    if a and b:
        return [normalize_city(a), normalize_city(b)]
    return [normalize_city(current_city)]


def iter_meal_acts(daily_plans: List[Dict[str, Any]]):
    """Iterate through all meal activities in daily plans."""
    results = []
    for day in daily_plans:
        for act in day.get("activities", []) or []:
            if act.get("type") != "meal":
                continue
            details = act.get("details") or {}
            name = (details.get("name") or "").strip()
            results.append((act, details, name))
    return results


def iter_hotel_acts(daily_plans: List[Dict[str, Any]]):
    """Iterate through all hotel activities in daily plans."""
    results = []
    for day in daily_plans:
        for act in day.get("activities", []) or []:
            if act.get("type") != "hotel":
                continue
            details = act.get("details") or {}
            name = extract_hotel_name_from_activity(details)
            results.append((act, details, name))
    return results


def iter_attraction_acts(daily_plans: List[Dict[str, Any]]):
    """Iterate through all attraction activities in daily plans."""
    results = []
    for day in daily_plans:
        for act in day.get("activities", []) or []:
            if act.get("type") != "attraction":
                continue
            details = act.get("details") or {}
            name = (details.get("name") or "").strip()
            results.append((act, details, name))
    return results


def iter_intercity_public_acts(daily_plans: List[Dict[str, Any]]):
    """Iterate through all intercity public transport activities in daily plans."""
    results = []
    for day in daily_plans:
        for act in day.get("activities", []) or []:
            if act.get("type") != "travel_intercity_public":
                continue
            details = act.get("details") or {}
            results.append((act, details))
    return results


def end_city_of_day(current_city: str) -> Optional[str]:
    """Get the ending city of a day."""
    a, b = extract_from_to(current_city)
    if a and b:
        return normalize_city(b)
    return normalize_city(current_city)


def get_day_accommodation_city(day: Dict[str, Any], hotels_index: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[str]:
    """Get the accommodation city for a given day."""
    # Priority 1: Read city from hotel activity
    for act, details, _name in iter_hotel_acts([day]):
        city = (details.get("city") or "").strip()
        if city:
            return normalize_city(city)
    # Priority 2: Read from day.accommodation field, look up city by hotel name in hotels.csv
    accom = day.get("accommodation")
    if isinstance(accom, dict):
        hotel_name = (accom.get("name") or "").strip()
        hotel_record = get_index_record(hotels_index or {}, hotel_name)
        if hotel_name and hotel_record:
            city_str = hotel_record.get("city")
            if city_str:
                return normalize_city(str(city_str).strip())
    return None


def iter_accommodation_entries(daily_plans: List[Dict[str, Any]]):
    """Iterate through all accommodation entries: hotel activities + day.accommodation."""
    for idx, day in enumerate(daily_plans):
        # Hotel activities (except last day, as it may have checkout activity)
        if idx < len(daily_plans) - 1:
            for act, details, name in iter_hotel_acts([day]):
                yield idx, {
                    "name": name,
                    "price": details.get("price") or details.get("cost"),
                    "city": (details.get("city") or "").strip(),
                    "source": "activity",
                }
        # day.accommodation field
        accom = day.get("accommodation")
        if isinstance(accom, dict):
            yield idx, {
                "name": (accom.get("name") or "").strip(),
                "price": accom.get("price") or accom.get("cost") or accom.get("price_per_night"),
                "city": (accom.get("city") or "").strip(),
                "source": "field",
            }


def get_intercity_arrival_time(day: Dict[str, Any]) -> Optional[float]:
    """Get final arrival time of intercity transportation for a day.

    A connecting route is represented as multiple travel_intercity_public
    activities on the same day.  Daily meal/attraction coverage should be based
    on the arrival at the destination city, not the first transfer hub.
    """
    arrival: Optional[float] = None
    for act in day.get("activities", []) or []:
        if act.get("type") == "travel_intercity_public":
            # Priority: use end_time, if not available extract from time_slot
            end_time = act.get("end_time", "")
            if not end_time:
                time_slot = act.get("time_slot", "")
                if time_slot and "-" in time_slot:
                    end_time = time_slot.split("-")[1]
            
            if end_time:
                try:
                    hour, minute = map(int, end_time.split(":"))
                    arrival = hour + minute / 60.0
                except:
                    pass
    return arrival


def get_intercity_departure_time(day: Dict[str, Any]) -> Optional[float]:
    """Get the departure time of intercity transportation for a given day (in hours)."""
    for act in day.get("activities", []) or []:
        if act.get("type") == "travel_intercity_public":
            # Priority: use start_time, if not available extract from time_slot
            start_time = act.get("start_time", "")
            if not start_time:
                time_slot = act.get("time_slot", "")
                if time_slot and "-" in time_slot:
                    start_time = time_slot.split("-")[0]
            
            if start_time:
                try:
                    hour, minute = map(int, start_time.split(":"))
                    return hour + minute / 60.0
                except:
                    pass
    return None
