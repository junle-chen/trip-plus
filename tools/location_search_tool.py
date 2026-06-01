"""
Location Search Tool - Query location coordinates (English-only)
"""
import csv
import difflib
import os
import re
from pathlib import Path
from typing import Dict, Optional, Union

from .base_travel_tool import BaseTravelTool, register_tool
from .city_db_access import (
    load_city_index,
    load_city_location_aliases,
    load_city_location_entities,
    load_city_locations,
    load_city_subway_stations,
)


def _compact_name(value: object) -> str:
    return (
        str(value or "")
        .strip()
        .replace(" ", "")
        .replace("　", "")
        .replace("（", "(")
        .replace("）", ")")
    )


def _alias_lookup_keys(value: object) -> list[str]:
    raw = str(value or "").strip()
    compact = _compact_name(raw)
    keys = []
    for key in (raw, compact):
        if key and key not in keys:
            keys.append(key)
    return keys


def _normalize_transport_hub_name(value: object) -> str:
    text = _compact_name(value)
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


def _can_fuzzy_match_transport_hub(row: object) -> bool:
    try:
        poi_type = str(row.get("poi_type", "") or row.get("type", "")).lower()
        source = str(row.get("source", "")).lower()
        row_name = str(row.get("poi_name", "") or row.get("place_name", "")).strip()
    except Exception:
        return False
    if any(token in poi_type for token in ("hotel", "restaurant", "attraction")):
        return False
    if re.search(r"\b(hotel|restaurant|cafe|coffee|buffet|cuisine|dining|lounge|bistro|bar)\b", row_name, re.I):
        return False
    return (
        any(token in poi_type for token in ("station", "airport", "subway", "railway", "transport"))
        or source in {"city_subway", "transport_hub", "route_station_name"}
        or bool(re.search(r"\b(airport|railway station|train station|station)\b", row_name, re.I))
    )


def _city_prefix_variants(value: object, candidate_cities: list[str]) -> list[str]:
    """Return variants with an explicit city prefix stripped.

    Tool outputs often use canonical local names while the model may prefix
    the city name. This keeps lookup deterministic by only
    stripping cities already known for the sample.
    """
    compact = _compact_name(value)
    variants: list[str] = []
    for city in candidate_cities:
        city_compact = _compact_name(city)
        if not city_compact:
            continue
        city_forms = [city_compact]
        for prefix in city_forms:
            if compact.startswith(prefix) and len(compact) > len(prefix):
                stripped = compact[len(prefix):]
                if stripped and stripped not in variants:
                    variants.append(stripped)
    return variants


def _load_location_aliases(alias_path: Path) -> dict[str, str]:
    if not alias_path.exists():
        return {}
    aliases: dict[str, str] = {}
    with alias_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            alias = str(row.get("alias", "")).strip()
            canonical_name = str(row.get("canonical_name", "")).strip()
            if not alias or not canonical_name or alias == canonical_name:
                continue
            for key in _alias_lookup_keys(alias):
                aliases.setdefault(key, canonical_name)
    return aliases


@register_tool('search_location')
class LocationSearchTool(BaseTravelTool):
    """Tool for querying location latitude and longitude coordinates (English-only)"""
    
    # English field mappings
    LANG_FIELDS = {
        'en': {
            'db_not_loaded': "Database not loaded",
            'not_found': lambda place: f"Coordinate information not found for location {place}, please check: 1. Whether the place name comes from other tool results; 2. Whether the place name is exactly consistent with tool results, no abbreviation, renaming or additional description allowed",
        }
    }
    
    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        cfg = cfg or {}
        self.database_path = cfg.get('database_path') if cfg else None
        if not self.database_path and cfg.get("sample_db_path"):
            sample_location_path = Path(cfg["sample_db_path"]) / "locations" / "locations_coords.csv"
            if sample_location_path.exists():
                self.database_path = str(sample_location_path)
        if cfg.get("city_db_root"):
            self.city_db_root = Path(cfg["city_db_root"])
        else:
            project_root = Path(__file__).resolve().parent.parent
            self.city_db_root = None
            for default_city_root in (
                project_root / "database" / self.language,
                project_root / "database" / "database_by_city" / self.language,
            ):
                if (default_city_root / "city_index.json").exists():
                    self.city_db_root = default_city_root
                    break
        self.candidate_cities = [str(city).strip() for city in cfg.get("candidate_cities", []) if str(city).strip()]
        
        # Get English fields
        self.fields = self.LANG_FIELDS.get(self.language, self.LANG_FIELDS['en'])
        
        if self.database_path and os.path.exists(self.database_path):
            self.data = self.load_csv_database(self.database_path)
            self.location_aliases = _load_location_aliases(
                Path(self.database_path).parent / "location_aliases.csv"
            )
        else:
            self.data = None
            self.location_aliases = {}
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Execute location coordinate query
        
        Args:
            params: Query parameters containing place_name
            
        Returns:
            JSON string of query results
        """
        params = self._verify_json_format_args(params)
        
        place_name = params.get('place_name')
        
        if self.data is None:
            fallback = self._search_city_level_locations(place_name)
            if fallback is not None:
                return self.format_result_as_json(fallback)
            if self.city_db_root is not None:
                return self.fields['not_found'](place_name)
            return self.fields['db_not_loaded']
        
        # Query from CSV database. Runtime lookup is exact or database-backed:
        # aliases must be present in locations/location_aliases.csv.
        col_name = 'poi_name' if 'poi_name' in self.data.columns else 'place_name'
        query_result = self.data[self.data[col_name] == place_name]
        matched_name = place_name

        if query_result.empty:
            canonical_name = None
            for key in _alias_lookup_keys(place_name):
                canonical_name = self.location_aliases.get(key)
                if canonical_name:
                    break
            if canonical_name:
                query_result = self.data[self.data[col_name] == canonical_name]
                if not query_result.empty:
                    matched_name = canonical_name

        if query_result.empty:
            compact_place = _compact_name(place_name)
            compact_names = self.data[col_name].map(_compact_name)
            query_result = self.data[compact_names == compact_place]
            if not query_result.empty:
                matched_name = query_result.iloc[0].get(col_name, place_name)

        if query_result.empty:
            compact_names = self.data[col_name].map(_compact_name)
            for stripped in _city_prefix_variants(place_name, self.candidate_cities):
                query_result = self.data[compact_names == stripped]
                if not query_result.empty:
                    matched_name = query_result.iloc[0].get(col_name, stripped)
                    break

        if query_result.empty:
            for _, candidate in self.data.iterrows():
                row_name = str(candidate.get(col_name, "")).strip()
                if _can_fuzzy_match_transport_hub(candidate) and _transport_hub_name_matches(place_name, row_name):
                    query_result = self.data[self.data[col_name] == row_name]
                    matched_name = row_name
                    break

        if query_result.empty:
            fallback = self._search_city_level_locations(place_name)
            if fallback is None:
                return self.fields['not_found'](place_name)
            return self.format_result_as_json(fallback)
        
        # Build return result
        row = query_result.iloc[0]
        result = {
            "place_name": row.get('poi_name', row.get('place_name', place_name)),
            "latitude": str(row.get('latitude', '')),
            "longitude": str(row.get('longitude', '')),
        }
        if matched_name != place_name:
            result["matched_place_name"] = str(matched_name)
        
        return self.format_result_as_json(result)

    def _search_city_level_locations(self, place_name: object) -> Optional[dict]:
        if self.city_db_root is None:
            return None

        requested_compact = _compact_name(place_name)
        candidate_cities = self.candidate_cities
        if not candidate_cities:
            try:
                city_index = load_city_index(self.city_db_root)
                candidate_cities = [
                    str(info.get("city_name") or info.get("folder_name") or key).strip()
                    for key, info in city_index.items()
                    if isinstance(info, dict) and str(info.get("city_name") or info.get("folder_name") or key).strip()
                ]
            except Exception:
                candidate_cities = []

        best: tuple[int, dict[str, str], str] | None = None
        for city in candidate_cities:
            city_rows = load_city_locations(self.city_db_root, city)
            city_rows.extend(load_city_location_entities(self.city_db_root, city))
            city_rows.extend(load_city_subway_stations(self.city_db_root, city))
            alias_map = load_city_location_aliases(self.city_db_root, city)
            canonical_name = None
            for key in _alias_lookup_keys(place_name):
                canonical_name = alias_map.get(key)
                if canonical_name:
                    break
            if canonical_name:
                for row in city_rows:
                    row_name = str(row.get("poi_name") or row.get("place_name") or "").strip()
                    if row_name == canonical_name:
                        return {
                            "place_name": row_name,
                            "latitude": str(row.get("latitude", "")),
                            "longitude": str(row.get("longitude", "")),
                            "matched_place_name": row_name,
                            "matched_city": city,
                            "source": str(row.get("source") or "city_level_aliases"),
                        }
            for row in city_rows:
                row_name = str(row.get("poi_name") or row.get("place_name") or "").strip()
                compact_row_name = _compact_name(row_name)
                if not compact_row_name:
                    continue
                stripped_variants = _city_prefix_variants(place_name, [city])
                exact = (
                    row_name == str(place_name or "").strip()
                    or compact_row_name == requested_compact
                    or compact_row_name in stripped_variants
                )
                if not exact and not (_can_fuzzy_match_transport_hub(row) and _transport_hub_name_matches(place_name, row_name)):
                    continue
                score = len(compact_row_name)
                if best is None or score > best[0]:
                    best = (score, row, city)

        if best is None:
            return None

        _, row, city = best
        row_name = str(row.get("poi_name") or row.get("place_name") or place_name)
        result = {
            "place_name": row_name,
            "latitude": str(row.get("latitude", "")),
            "longitude": str(row.get("longitude", "")),
            "matched_place_name": row_name,
            "matched_city": city,
            "source": str(row.get("source") or "city_level_locations"),
        }
        return result
