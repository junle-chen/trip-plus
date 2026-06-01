"""
Attraction Query Tool - Query and recommend attractions (English-only)
"""
import os
import re
import math
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

from .base_travel_tool import BaseTravelTool, register_tool


def _clean_text(value: object) -> str:
    try:
        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        text = str(value)
        if text.lower() == "nan":
            return ""
        return text
    except Exception:
        return ""


def _alias_lookup_keys(value: object) -> list[str]:
    text = _clean_text(value).strip()
    if not text:
        return []
    compact = re.sub(r"\s+", "", text)
    normalized = compact.replace("（", "(").replace("）", ")").lower()
    keys = [text, compact, normalized]
    seen: set[str] = set()
    result: list[str] = []
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _weekday_from_visit_params(params: dict) -> Optional[int]:
    visit_weekday = _clean_text(params.get("visit_weekday", "")).strip()
    if visit_weekday:
        mapping = {
            "1": 1,
            "monday": 1,
            "2": 2,
            "tuesday": 2,
            "3": 3,
            "wednesday": 3,
            "4": 4,
            "thursday": 4,
            "5": 5,
            "friday": 5,
            "6": 6,
            "saturday": 6,
            "7": 7,
            "sunday": 7,
        }
        normalized = visit_weekday.lower()
        if normalized in mapping:
            return mapping[normalized]
        if visit_weekday in mapping:
            return mapping[visit_weekday]

    visit_date = _clean_text(params.get("visit_date", "")).strip()
    if visit_date:
        try:
            return datetime.strptime(visit_date, "%Y-%m-%d").weekday() + 1
        except ValueError:
            return None
    return None


def _closed_weekdays(closing_dates: str) -> set[int]:
    mapping = {
        "monday": 1,
        "tuesday": 2,
        "wednesday": 3,
        "thursday": 4,
        "friday": 5,
        "saturday": 6,
        "sunday": 7,
    }
    closed = set()
    for part in _clean_text(closing_dates).split(","):
        token = part.strip()
        if not token:
            continue
        closed_day = mapping.get(token.lower()) or mapping.get(token)
        if closed_day:
            closed.add(closed_day)
    return closed


@register_tool('query_attraction_details')
class AttractionDetailsQueryTool(BaseTravelTool):
    """Tool for querying detailed attraction information (English-only)"""
    
    # English field mappings
    LANG_FIELDS = {
        'en': {
            'db_not_loaded': "Database not loaded",
            'not_found': lambda name: f"Detailed information not found for attraction {name}",
            'attraction_name': "Attraction Name",
            'city': "City",
            'address': "Address",
            'coordinates': "Coordinates",
            'latitude': "Latitude",
            'longitude': "Longitude",
            'description': "Description",
            'rating': "Rating",
            'visitor_rating': "(average visitor rating)",
            'opening_hours': "Opening Hours",
            'to': "to",
            'closed_dates': "Closed Dates",
            'visit_open_status': "Open Status on Visit Day",
            'open_on_visit_day': "Open",
            'closed_on_visit_day': "Closed",
            'min_visit_hours': "Minimum Visit Duration",
            'max_visit_hours': "Maximum Visit Duration",
            'hours_unit': "hours",
            'ticket_price': "Ticket Price",
            'currency_unit': "RMB",
            'attraction_type': "Attraction Type",
            'popularity_tags': "Popularity Tags",
            'crowd_risk': "Crowd Risk",
            'queue_risk': "Queue Risk",
            'peak_crowd_windows': "Peak Crowd Windows",
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
        self.location_aliases = self._load_location_aliases()

    def _load_location_aliases(self) -> dict[str, str]:
        if not self.database_path:
            return {}
        alias_path = Path(self.database_path).parent.parent / "locations" / "location_aliases.csv"
        if not alias_path.exists():
            return {}
        aliases: dict[str, str] = {}
        with alias_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                alias = _clean_text(row.get("alias")).strip()
                canonical_name = _clean_text(row.get("canonical_name")).strip()
                if not alias or not canonical_name or alias == canonical_name:
                    continue
                for key in _alias_lookup_keys(alias):
                    aliases.setdefault(key, canonical_name)
        return aliases

    def _resolve_attraction_alias(self, attraction_name: str) -> str:
        for key in _alias_lookup_keys(attraction_name):
            canonical = self.location_aliases.get(key)
            if canonical:
                return canonical
        return attraction_name
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Execute attraction details query
        
        Args:
            params: Query parameters containing attraction_name
            
        Returns:
            Formatted text string of query results
        """
        def format_result_as_text(result: dict) -> str:
            """Format attraction details dictionary into readable text"""
            lines = []
            lines.append(f"{self.fields['attraction_name']}：{result.get('attraction_name', '')}")
            lines.append(f"{self.fields['city']}：{result.get('city', '')}")
            lines.append(f"{self.fields['address']}：{result.get('address', '')}")
            lines.append(f"{self.fields['coordinates']}：{self.fields['latitude']} {result.get('latitude', '')}，{self.fields['longitude']} {result.get('longitude', '')}")
            lines.append(f"{self.fields['description']}：{result.get('description', '')}")
            lines.append(f"{self.fields['rating']}：{result.get('rating', '')}{self.fields['visitor_rating']}")
            
            # Handle opening hours
            opening_time = result.get('opening_time', '')
            closing_time = result.get('closing_time', '')
            if opening_time == closing_time:
                lines.append(f"{self.fields['opening_hours']}：{opening_time}")
            else:
                lines.append(f"{self.fields['opening_hours']}：{opening_time} {self.fields['to']} {closing_time}")
            
            lines.append(f"{self.fields['closed_dates']}：{result.get('closing_dates', '')}")
            if result.get("visit_open_status"):
                lines.append(f"{self.fields['visit_open_status']}：{result.get('visit_open_status', '')}")
            lines.append(f"{self.fields['min_visit_hours']}：{result.get('min_visit_hours', '')} {self.fields['hours_unit']}")
            lines.append(f"{self.fields['max_visit_hours']}：{result.get('max_visit_hours', '')} {self.fields['hours_unit']}")
            lines.append(f"{self.fields['ticket_price']}：{result.get('ticket_price', 0)} {self.fields['currency_unit']}")
            lines.append(f"{self.fields['attraction_type']}：{result.get('attraction_type', '')}")
            for key in ("popularity_tags", "crowd_risk", "queue_risk", "peak_crowd_windows"):
                value = result.get(key, "")
                if value:
                    lines.append(f"{self.fields[key]}：{value}")
            
            return "\n".join(lines)

        params = self._verify_json_format_args(params)
        
        attraction_name = params.get('attraction_name')
        visit_weekday = _weekday_from_visit_params(params)
        
        # Database not loaded
        if self.data is None:
            return self.fields['db_not_loaded']
        
        # Query from CSV
        df = self.data
        rows = df[df['attraction_name'] == attraction_name]
        if rows.empty:
            canonical_name = self._resolve_attraction_alias(attraction_name)
            if canonical_name != attraction_name:
                rows = df[df['attraction_name'] == canonical_name]
        if rows.empty:
            return self.fields['not_found'](attraction_name)
        row = rows.iloc[0]
        
        # Convert numpy scalars to Python basic types
        def to_num(v):
            try:
                if v == v:  # Filter NaN
                    return float(v)
                return None
            except Exception:
                return None

        rating_val = to_num(row.get("rating", None))
        min_hours_val = to_num(row.get("min_visit_hours", None))
        max_hours_val = to_num(row.get("max_visit_hours", None))
        ticket_price_val = to_num(row.get("ticket_price", None))

        closing_dates = _clean_text(row.get("closing_dates", ""))
        visit_open_status = ""
        if visit_weekday is not None:
            if visit_weekday in _closed_weekdays(closing_dates):
                visit_open_status = self.fields["closed_on_visit_day"]
            else:
                visit_open_status = self.fields["open_on_visit_day"]

        # Build result
        result = {
            "attraction_name": _clean_text(row.get("attraction_name", attraction_name)),
            "city": _clean_text(row.get("city", "")),
            "address": _clean_text(row.get("address", "")),
            "latitude": _clean_text(row.get("latitude", "")),
            "longitude": _clean_text(row.get("longitude", "")),
            "description": _clean_text(row.get("description", "")),
            "rating": rating_val if rating_val is not None else "",
            "opening_time": _clean_text(row.get("opening_time", "")),
            "closing_time": _clean_text(row.get("closing_time", "")),
            "closing_dates": closing_dates,
            "visit_open_status": visit_open_status,
            "min_visit_hours": min_hours_val if min_hours_val is not None else "",
            "max_visit_hours": max_hours_val if max_hours_val is not None else "",
            "ticket_price": ticket_price_val if ticket_price_val is not None else "unknown",
            "attraction_type": _clean_text(row.get("attraction_type", "")),
            "popularity_tags": _clean_text(row.get("popularity_tags", "")),
            "crowd_risk": _clean_text(row.get("crowd_risk", "")),
            "queue_risk": _clean_text(row.get("queue_risk", "")),
            "peak_crowd_windows": _clean_text(row.get("peak_crowd_windows", "")),
        }
        
        return format_result_as_text(result)


@register_tool('recommend_attractions')
class AttractionRecommendTool(BaseTravelTool):
    """Tool for recommending attractions (English-only)"""
    
    # English field mappings
    LANG_FIELDS = {
        'en': {
            'db_not_loaded': "Database not loaded",
            'not_found': "No attraction recommendations found",
            'recommendations': "Recommended attractions:\n",
            'attraction_suffix': lambda name, desc, atype: f"Attraction Name: {name}; Description: {desc}; Type: {atype}",
            'popularity_tags': "Popularity Tags",
            'crowd_risk': "Crowd Risk",
            'queue_risk': "Queue Risk",
            'peak_crowd_windows': "Peak Crowd Windows",
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

    def _normalize_city_name(self, value: object) -> str:
        text = str(value or '').strip()
        if not text:
            return ''
        text = re.sub(r"[（(].*?[)）]", "", text).strip()
        return re.sub(r"\s+city$", "", text, flags=re.IGNORECASE)

    def _to_float(self, value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Execute attraction recommendation
        
        Args:
            params: Query parameters containing city and optional attraction_type
            
        Returns:
            Formatted text string of recommendations
        """
        params = self._verify_json_format_args(params)
        
        city = params.get('city')
        attraction_type = params.get('attraction_type', '')
        
        # Database not loaded
        if self.data is None:
            return self.fields['db_not_loaded']

        df = self.data
        rows = df
        if city and 'city' in rows.columns:
            normalized_city = self._normalize_city_name(city)
            if normalized_city:
                normalized_cities = rows['city'].astype(str).map(self._normalize_city_name)
                city_filtered = rows[normalized_cities == normalized_city]
                if not city_filtered.empty:
                    rows = city_filtered
        if attraction_type:
            rows = rows[rows['attraction_type'] == attraction_type]

        if rows.empty:
            return self.fields['not_found']

        deduped = {}
        for _, row in rows.iterrows():
            attraction_name = str(row.get("attraction_name", "")).strip()
            if not attraction_name:
                continue
            existing = deduped.get(attraction_name)
            if existing is None:
                deduped[attraction_name] = row
                continue

            current_key = (
                self._to_float(row.get("rating", 0), -1.0),
                -self._to_float(row.get("ticket_price", 0), 0.0),
                attraction_name,
            )
            existing_key = (
                self._to_float(existing.get("rating", 0), -1.0),
                -self._to_float(existing.get("ticket_price", 0), 0.0),
                attraction_name,
            )
            if current_key > existing_key:
                deduped[attraction_name] = row

        all_rows = sorted(
            deduped.values(),
            key=lambda row: (
                -self._to_float(row.get("rating", 0), -1.0),
                self._to_float(row.get("ticket_price", 0), 0.0),
                str(row.get("attraction_name", "")),
            ),
        )[:10]

        # Build result string
        result_lines = [self.fields['recommendations']]

        for r in all_rows:
            attraction_name = _clean_text(r.get("attraction_name", ""))
            description = _clean_text(r.get("description", ""))
            attraction_type = _clean_text(r.get("attraction_type", ""))
            crowd_notes = []
            for field in ("popularity_tags", "crowd_risk", "queue_risk", "peak_crowd_windows"):
                value = _clean_text(r.get(field, "")).strip()
                if value:
                    crowd_notes.append(f"{self.fields[field]}：{value}")
            suffix = self.fields['attraction_suffix'](attraction_name, description, attraction_type)
            if crowd_notes:
                if self.language == "en":
                    suffix += "; " + ", ".join(note.replace("：", ": ") for note in crowd_notes)
                else:
                    suffix += "；" + "，".join(crowd_notes)
            result_lines.append(suffix)
        
        return "\n".join(result_lines)
