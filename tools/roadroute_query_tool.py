"""
Road Route Query Tool - Query distance and duration between locations (English-only)
"""
import math
import os
from typing import Dict, Optional, Union

from .base_travel_tool import BaseTravelTool, register_tool


def _normalize_coord(coord: object) -> str:
    raw = str(coord or "").strip()
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        return raw
    try:
        return f"{float(parts[0]):.6f},{float(parts[1]):.6f}"
    except ValueError:
        return raw


def _parse_coord(coord: object) -> tuple[float, float] | None:
    raw = str(coord or "").strip()
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _haversine_meters(origin: str, destination: str) -> float | None:
    parsed_origin = _parse_coord(origin)
    parsed_destination = _parse_coord(destination)
    if parsed_origin is None or parsed_destination is None:
        return None
    lat1, lon1 = parsed_origin
    lat2, lon2 = parsed_destination
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


def _fallback_route(origin: str, destination: str, reason: str) -> dict | None:
    distance_meters = _haversine_meters(origin, destination)
    if distance_meters is None:
        return None
    if distance_meters <= 1500:
        duration_minutes = max(1, int(round((distance_meters / 1000.0) / 5.0 * 60)))
        cost = 0
        mode = "walking"
    else:
        duration_minutes = max(5, int(round((distance_meters / 1000.0) / 30.0 * 60)))
        distance_km = distance_meters / 1000.0
        cost = int(round(10 + max(0.0, distance_km - 3.0) * 2.2))
        mode = "taxi_estimate"
    return {
        "origin": _normalize_coord(origin),
        "destination": _normalize_coord(destination),
        "distance_in_meters": int(round(distance_meters)),
        "duration_in_minutes": duration_minutes,
        "cost": cost,
        "source": "deterministic_fallback",
        "mode": mode,
        "fallback_reason": reason,
    }


@register_tool('query_road_route_info')
class RoadRouteInfoQueryTool(BaseTravelTool):
    """Tool for querying distance and duration information between two locations (English-only)"""
    
    # English field mappings
    LANG_FIELDS = {
        'en': {
            'db_not_loaded': "Database not loaded",
            'not_found': lambda origin, dest: f"No transportation information found from {origin} to {dest}",
            'coord_not_in_range': lambda coord: f"Coordinate {coord} is not in query range, please check:\n1. Whether coordinate comes from valid tool query result, not manual input or fabrication;\n2. Whether coordinate precision is exactly consistent with query result, 6 decimal places",
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
        Execute road route query
        
        Args:
            params: Query parameters containing:
                - origin: Coordinate string in format "latitude,longitude"
                - destination: Coordinate string in format "latitude,longitude"
                - mode: Transportation mode (walking, driving, taxi)
            
        Returns:
            JSON string of query results
        """
        params = self._verify_json_format_args(params)
        
        origin = str(params.get('origin') or '').strip()
        destination = str(params.get('destination') or '').strip()
        
        origin = self._resolve_coordinate_alias(origin)
        destination = self._resolve_coordinate_alias(destination)

        if self.data is None:
            fallback = _fallback_route(origin, destination, "distance_matrix_not_loaded")
            if fallback is not None:
                return self.format_result_as_json(fallback)
            return self.fields['db_not_loaded']
        
        # Query directly using coordinates
        query_result = self.data[
            (self.data['origin'] == origin) &
            (self.data['destination'] == destination)
        ]
        
        if query_result.empty:
            fallback = _fallback_route(origin, destination, "coordinate_pair_not_in_distance_matrix")
            if fallback is not None:
                return self.format_result_as_json(fallback)
            return self.fields['not_found'](origin, destination)
        
        # Build return result
        row = query_result.iloc[0]
        result = {
            "origin": row.get('origin', origin),
            "destination": row.get('destination', destination),
            "distance_in_meters": int(row.get('distance_meters', 0)),
            "duration_in_minutes": int(row.get('duration_minutes', 0)),
            "cost": int(row.get('cost', 0)),
            "source": "distance_matrix",
        }
        
        return self.format_result_as_json(result)

    def _resolve_coordinate_alias(self, coord: str) -> str:
        if self.data is None:
            return coord
        all_coords = set(self.data['origin'].unique()) | set(self.data['destination'].unique())
        if coord in all_coords:
            return coord
        normalized = _normalize_coord(coord)
        return normalized if normalized in all_coords else coord
    
    def _check_coordinate_existence(self, origin: str, destination: str) -> str:
        """
        Check if coordinates exist in database
        
        Args:
            origin: Origin coordinate string
            destination: Destination coordinate string
            
        Returns:
            Error message string, empty string if no error
        """
        # Get all unique origin and destination coordinates from database
        all_origins = set(self.data['origin'].unique())
        all_destinations = set(self.data['destination'].unique())
        
        # Merge all coordinates
        all_coords = all_origins | all_destinations
        
        # Check origin coordinate
        if origin not in all_coords:
            return self.fields['coord_not_in_range'](origin)
        
        # Check destination coordinate
        if destination not in all_coords:
            return self.fields['coord_not_in_range'](destination)
        
        return ""
