"""
Restaurant Query Tool - Recommend and query restaurant information (English-only)
"""
import csv
import math
import os
from pathlib import Path
from typing import Dict, Optional, Union

from .base_travel_tool import BaseTravelTool, register_tool


def _compact_text(value: object) -> str:
    text = _clean_text(value)
    return (
        text
        .strip()
        .lower()
        .replace(" ", "")
        .replace("　", "")
        .replace("（", "(")
        .replace("）", ")")
    )


def _is_missing_value(value: object) -> bool:
    if value is None:
        return True
    if value != value:
        return True
    text = str(value).strip()
    return text.lower() in {"", "nan", "none", "nat"}


def _clean_text(value: object, default: str = "") -> str:
    if _is_missing_value(value):
        return default
    return str(value).strip()


def _split_tags(value: object, limit: int = 8) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    tags: list[str] = []
    seen = set()
    for item in value.split(';'):
        tag = item.strip()
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
        if len(tags) >= limit:
            break
    return tags


def _safe_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _euclidean_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return math.sqrt(((lat1 - lat2) * 111_000) ** 2 + ((lon1 - lon2) * 85_000) ** 2)


@register_tool('recommend_restaurants')
class RestaurantRecommendTool(BaseTravelTool):
    """Tool for recommending nearby restaurants (English-only)"""
    
    # English field mappings
    LANG_FIELDS = {
        'en': {
            'db_not_loaded': "Database not loaded",
            'missing_anchor': "Restaurant recommendation requires nearby_attraction_name from the user request or a tool-returned place name",
            'not_found_attraction': lambda name: f"No recommended restaurants found near {name}; check whether the place name matches the tool result or user request",
        }
    }
    
    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        cfg = cfg or {}
        self.database_path = cfg.get('database_path') if cfg else None
        sample_db_path = cfg.get('sample_db_path')
        self.sample_db_path = Path(sample_db_path) if sample_db_path else None
        self.location_coords = self._load_location_coords()
        
        # Get English fields
        self.fields = self.LANG_FIELDS.get(self.language, self.LANG_FIELDS['en'])
        
        if self.database_path and os.path.exists(self.database_path):
            self.data = self.load_csv_database(self.database_path)
        else:
            self.data = None

    def _to_float(self, value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _load_location_coords(self) -> dict[str, tuple[float, float]]:
        if self.sample_db_path is None:
            return {}
        path = self.sample_db_path / "locations" / "locations_coords.csv"
        if not path.exists():
            return {}
        coords: dict[str, tuple[float, float]] = {}
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    name = _clean_text(row.get("poi_name") or row.get("name"))
                    lat = _safe_float(row.get("latitude") or row.get("lat"))
                    lon = _safe_float(row.get("longitude") or row.get("lon"))
                    if name and lat is not None and lon is not None:
                        coords[_compact_text(name)] = (lat, lon)
        except Exception:
            return {}
        return coords

    def _row_lat_lon(self, row) -> Optional[tuple[float, float]]:
        lat = _safe_float(row.get("latitude") or row.get("lat"))
        lon = _safe_float(row.get("longitude") or row.get("lon"))
        if lat is not None and lon is not None:
            return lat, lon
        restaurant_name = _clean_text(row.get("restaurant_name") or row.get("name"))
        return self.location_coords.get(_compact_text(restaurant_name))

    def _anchor_lat_lon(self, row, nearby_attraction_name: object) -> Optional[tuple[float, float]]:
        query_lat = _safe_float(row.get("query_latitude") or row.get("nearby_attraction_latitude"))
        query_lon = _safe_float(row.get("query_longitude") or row.get("nearby_attraction_longitude"))
        if query_lat is not None and query_lon is not None:
            return query_lat, query_lon

        coords = _clean_text(row.get("nearby_attraction_coords"))
        if "," in coords:
            lng_str, lat_str = coords.split(",", 1)
            lat = _safe_float(lat_str)
            lon = _safe_float(lng_str)
            if lat is not None and lon is not None:
                return lat, lon

        anchor = _clean_text(row.get("nearby_attraction_name") or nearby_attraction_name)
        return self.location_coords.get(_compact_text(anchor))

    def _distance_meters(self, row, nearby_attraction_name: object) -> Optional[int]:
        restaurant_coords = self._row_lat_lon(row)
        anchor_coords = self._anchor_lat_lon(row, nearby_attraction_name)
        if restaurant_coords is None or anchor_coords is None:
            return None
        distance = _euclidean_distance_m(
            restaurant_coords[0],
            restaurant_coords[1],
            anchor_coords[0],
            anchor_coords[1],
        )
        return int(distance)

    def _group_distance(self, rows, nearby_attraction_name: object) -> float:
        distances = [
            self._distance_meters(row, nearby_attraction_name)
            for _, row in rows.iterrows()
        ]
        distances = [value for value in distances if value is not None]
        return float(min(distances)) if distances else float("inf")

    def _normalize_sort_by(self, value: object) -> str:
        text = _compact_text(value)
        aliases = {
            "closest": "distance",
            "nearest": "distance",
            "near": "distance",
            "distance": "distance",
            "price": "price",
            "cheap": "price",
            "cheapest": "price",
            "budget": "price",
            "rating": "rating",
            "rated": "rating",
            "highest": "rating",
            "highestrated": "rating",
        }
        return aliases.get(text, "auto")

    def _ordered_groups(self, grouped_rows, sort_by: str, nearby_attraction_name: object):
        def min_price(item) -> float:
            return min(self._to_float(v, float("inf")) for v in item[1].get("price_per_person", []))

        def max_rating(item) -> float:
            return max(self._to_float(v, -1.0) for v in item[1].get("rating", []))

        def name(item) -> str:
            return str(item[1].iloc[0].get("restaurant_name", ""))

        def distance(item) -> float:
            return self._group_distance(item[1], nearby_attraction_name)

        by_distance = sorted(grouped_rows, key=lambda item: (distance(item), min_price(item), -max_rating(item), name(item)))
        by_price = sorted(grouped_rows, key=lambda item: (min_price(item), distance(item), -max_rating(item), name(item)))
        by_rating = sorted(grouped_rows, key=lambda item: (-max_rating(item), distance(item), min_price(item), name(item)))

        if sort_by == "distance":
            return by_distance[:10]
        if sort_by == "price":
            return by_price[:10]
        if sort_by == "rating":
            return by_rating[:10]

        # Default mode preserves the old price-first behavior, but makes sure
        # the closest and highest-rated leaders are not hidden by the top-10
        # cutoff. Explicit closest requests should use sort_by=distance.
        selected = list(by_price[:10])
        selected_names = {name(item) for item in selected}
        for leader in (by_distance[0] if by_distance else None, by_rating[0] if by_rating else None):
            if leader is None or name(leader) in selected_names:
                continue
            if len(selected) >= 10:
                selected[-1] = leader
            else:
                selected.append(leader)
            selected_names.add(name(leader))
        return selected[:10]

    def _merge_tags(self, rows) -> list[str]:
        tags: list[str] = []
        seen = set()
        for _, row in rows.iterrows():
            for item in _split_tags(row.get('tags', None)):
                if item and item not in seen:
                    seen.add(item)
                    tags.append(item)
                if len(tags) >= 8:
                    return tags
        return tags

    def _match_rows_for_nearby_attraction(self, attraction_name: object):
        query = _compact_text(attraction_name)
        if not query:
            return self.data.iloc[0:0], "", "empty_query"

        if 'nearby_attraction_name' in self.data.columns:
            names = self.data['nearby_attraction_name'].map(_compact_text)
            exact = self.data[names == query]
            if not exact.empty:
                return exact, _clean_text(exact.iloc[0].get("nearby_attraction_name")), "exact"

            substring = self.data[
                names.map(lambda item: bool(item) and (query in item or item in query))
            ]
            if not substring.empty:
                matched_names = {
                    _compact_text(item)
                    for item in substring.get('nearby_attraction_name', [])
                    if _compact_text(item)
                }
                if len(matched_names) == 1:
                    return substring, _clean_text(substring.iloc[0].get("nearby_attraction_name")), "substring"

        return self._match_rows_for_anchor_coordinates(attraction_name)

    def _match_rows_for_anchor_coordinates(self, attraction_name: object):
        anchor_coords = self.location_coords.get(_compact_text(attraction_name))
        if anchor_coords is None:
            return self.data.iloc[0:0], "", "not_found"

        candidates: list[tuple[float, object]] = []
        for index, row in self.data.iterrows():
            restaurant_coords = self._row_lat_lon(row)
            if restaurant_coords is None:
                continue
            distance = _euclidean_distance_m(
                restaurant_coords[0],
                restaurant_coords[1],
                anchor_coords[0],
                anchor_coords[1],
            )
            candidates.append((distance, index))

        if not candidates:
            return self.data.iloc[0:0], "", "coordinate_fallback_no_restaurant_coords"

        for radius_meters in (3_000, 5_000):
            indexes = [index for distance, index in candidates if distance <= radius_meters]
            if not indexes:
                continue
            rows = self.data.loc[indexes].copy()
            rows["query_latitude"] = anchor_coords[0]
            rows["query_longitude"] = anchor_coords[1]
            return rows, _clean_text(attraction_name), "coordinate_fallback"

        return self.data.iloc[0:0], "", "coordinate_fallback_out_of_range"

    def _rows_for_nearby_attraction(self, attraction_name: object):
        rows, _matched_name, _match_type = self._match_rows_for_nearby_attraction(attraction_name)
        return rows

    def _match_rows_for_coordinates(self, latitude: object, longitude: object):
        lat = _safe_float(latitude)
        lon = _safe_float(longitude)
        if lat is None or lon is None or self.data is None:
            return self.data.iloc[0:0], "", "invalid_coordinates"

        best_name = ""
        best_distance: Optional[float] = None
        seen = set()
        for _, row in self.data.iterrows():
            anchor_name = _clean_text(row.get("nearby_attraction_name"))
            if not anchor_name or anchor_name in seen:
                continue
            seen.add(anchor_name)
            anchor_coords = self._anchor_lat_lon(row, anchor_name)
            if anchor_coords is None:
                continue
            distance = _euclidean_distance_m(lat, lon, anchor_coords[0], anchor_coords[1])
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_name = anchor_name

        if not best_name or best_distance is None or best_distance > 3_000:
            return self.data.iloc[0:0], "", "nearest_anchor_not_found"
        rows, matched_name, _match_type = self._match_rows_for_nearby_attraction(best_name)
        return rows, matched_name or best_name, "nearest_anchor"

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Execute restaurant recommendation
        
        Args:
            params: Query parameters containing nearby_attraction_name
            
        Returns:
            JSON string of query results
        """
        params = self._verify_json_format_args(params)
        
        nearby_attraction_name = (
            params.get('nearby_attraction_name')
            or params.get('attraction_name')
            or params.get('place_name')
        )
        sort_by = self._normalize_sort_by(params.get("sort_by"))
        query_nearby_attraction_name = _clean_text(nearby_attraction_name)
        matched_nearby_attraction_name = ""
        nearby_attraction_match_type = ""
        
        if self.data is None:
            return self.fields['db_not_loaded']

        if nearby_attraction_name:
            query_result, matched_nearby_attraction_name, nearby_attraction_match_type = self._match_rows_for_nearby_attraction(nearby_attraction_name)
            if query_result.empty:
                return self.fields['not_found_attraction'](nearby_attraction_name)
        elif params.get("latitude") is not None and params.get("longitude") is not None:
            query_nearby_attraction_name = f"{_clean_text(params.get('latitude'))},{_clean_text(params.get('longitude'))}"
            query_result, matched_nearby_attraction_name, nearby_attraction_match_type = self._match_rows_for_coordinates(
                params.get("latitude"),
                params.get("longitude"),
            )
            nearby_attraction_name = matched_nearby_attraction_name
            if query_result.empty:
                return self.fields['not_found_attraction'](query_nearby_attraction_name)
        else:
            return self.fields['missing_anchor']
        
        if query_result.empty:
            return self.fields['not_found_attraction'](nearby_attraction_name)

        results = []
        grouped_rows = [
            (_, group) for _, group in query_result.groupby('restaurant_name', sort=False)
        ]
        grouped_rows = self._ordered_groups(grouped_rows, sort_by, nearby_attraction_name)

        for _, rows in grouped_rows:
            row = rows.sort_values(
                by=['rating', 'price_per_person', 'restaurant_name'],
                ascending=[False, True, True],
                na_position='last',
            ).iloc[0]
            price_text = _clean_text(row.get('price_per_person', ''))
            result = {
                "name": _clean_text(row.get('restaurant_name', '')),
                "price_per_person": price_text,
                "cuisine": _clean_text(row.get('cuisine', '')),
                "opening_time": _clean_text(row.get('opening_time', '')),
                "closing_time": _clean_text(row.get('closing_time', '')),
                "nearby_attraction_name": _clean_text(row.get('nearby_attraction_name', '')),
                "rating": _clean_text(row.get('rating', '')),
                "query_nearby_attraction_name": query_nearby_attraction_name,
                "matched_nearby_attraction_name": matched_nearby_attraction_name,
                "nearby_attraction_match_type": nearby_attraction_match_type,
            }
            distance_meters = self._distance_meters(row, nearby_attraction_name)
            if distance_meters is not None:
                result["distance_meters"] = distance_meters
            if not price_text:
                result["price_missing"] = True
            if 'tags' in rows.columns:
                tags_list = self._merge_tags(rows)
                if tags_list:
                    result['tags'] = tags_list

            results.append(result)
        
        return self.format_result_as_json(results)


@register_tool('query_restaurant_details')
class RestaurantDetailsQueryTool(BaseTravelTool):
    """Tool for querying detailed restaurant information (English-only)"""
    
    # English field mappings
    LANG_FIELDS = {
        'en': {
            'db_not_loaded': "Database not loaded",
            'not_found': lambda name: f"Detailed information not found for restaurant {name}",
        }
    }
    
    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.database_path = cfg.get('database_path') if cfg else None
        
        # Get English fields
        self.fields = self.LANG_FIELDS.get(self.language, self.LANG_FIELDS['en'])
        
        if self.database_path and os.path.exists(self.database_path):
            self.data = self.load_csv_database(self.database_path)
        else:
            self.data = None
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Execute restaurant details query
        
        Args:
            params: Query parameters containing restaurant_name
            
        Returns:
            JSON string of query results
        """
        params = self._verify_json_format_args(params)
        
        restaurant_name = params.get('restaurant_name')
        
        if self.data is None:
            return self.format_result_as_json({
                "message": self.fields['db_not_loaded'],
                "restaurant_name": restaurant_name
            })
        
        # Query from CSV database
        if 'restaurant_name' in self.data.columns:
            query = _compact_text(restaurant_name)
            names = self.data['restaurant_name'].map(_compact_text)
            query_result = self.data[names == query]
        else:
            query_result = self.data.iloc[0:0]
        
        if query_result.empty:
            return self.format_result_as_json({
                "message": self.fields['not_found'](restaurant_name),
                "restaurant_name": restaurant_name
            })
        
        # Build return result (take first row if duplicates exist)
        row = query_result.iloc[0]
        price_text = _clean_text(row.get('price_per_person', ''))
        result = {
            "name": _clean_text(row.get('restaurant_name', restaurant_name)),
            "price_per_person": price_text,
            "cuisine": _clean_text(row.get('cuisine', '')),
            "opening_time": _clean_text(row.get('opening_time', '')),
            "closing_time": _clean_text(row.get('closing_time', '')),
            "nearby_attraction_name": _clean_text(row.get('nearby_attraction_name', '')),
            "rating": _clean_text(row.get('rating', ''))
        }
        if not price_text:
            result["price_missing"] = True
        
        # If CSV has tags field, add to return result
        if 'tags' in row.index:
            tags_list = _split_tags(row.get('tags', None))
            if tags_list:
                result['tags'] = tags_list
        
        return self.format_result_as_json(result)
