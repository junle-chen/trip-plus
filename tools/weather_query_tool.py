from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base_travel_tool import BaseTravelTool, register_tool
from .city_db_access import load_city_weather, resolve_city_db_root
from .weather_utils import normalize_weather_condition, to_float


def classify_weather_condition(row: dict[str, str] | None) -> dict[str, Any] | None:
    if not row:
        return None

    weather_code = int(round(to_float(row.get("weather_code"), -1)))
    label = normalize_weather_condition(row)

    return {
        "weather_code": weather_code,
        "condition": label,
        "weather_condition": label,
        "temperature_min_c": to_float(row.get("temperature_min_c")),
        "temperature_max_c": to_float(row.get("temperature_max_c")),
        "precipitation_mm": to_float(row.get("precipitation_mm")),
        "precipitation_hours": to_float(row.get("precipitation_hours")),
    }


def build_weather_advisory(row: dict[str, str] | None) -> dict[str, Any] | None:
    condition = classify_weather_condition(row)
    if condition is None:
        return None

    precipitation_mm = float(condition["precipitation_mm"])
    precipitation_hours = float(condition["precipitation_hours"])
    weather_code = int(condition["weather_code"])

    if weather_code in {95, 96, 99} or precipitation_mm >= 20:
        risk_level = "high"
        advice = "Heavy rain or convective weather; avoid long outdoor walking and prefer indoor activities or taxis"
    elif weather_code in {45, 48, 61, 63, 65, 66, 67, 71, 73, 75, 77, 80, 81, 82, 85, 86} or precipitation_hours >= 2:
        risk_level = "medium"
        advice = "Weather may affect local travel; reduce long walking and prefer metro or taxi"
    else:
        risk_level = "low"
        advice = "Weather has limited travel impact; walking and outdoor activities can be planned normally"

    return {
        "risk_level": risk_level,
        "advice": advice,
    }


@register_tool("query_city_weather")
class CityWeatherQueryTool(BaseTravelTool):
    """Query city-level daily weather information for planning."""

    LANG_FIELDS = {
        "en": {
            "missing_city_db": "City-level database not found. Build a sample cache from database/{lang} before using this tool",
            "not_found": lambda city, date: f"No weather information found for {city} on {date}",
        },
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.cfg = cfg or {}
        self.fields = self.LANG_FIELDS.get(self.language, self.LANG_FIELDS["en"])
        self.city_db_root = resolve_city_db_root(self.cfg)
        sample_db_path = str(self.cfg.get("sample_db_path", "")).strip()
        self.sample_db_path = Path(sample_db_path) if sample_db_path else None

    def _load_weather_rows(self, city_name: str) -> list[dict[str, str]]:
        if self.city_db_root is not None:
            rows = list(load_city_weather(self.city_db_root, city_name))
            if rows:
                return rows
        if self.sample_db_path is None:
            return []
        weather_path = self.sample_db_path / "weather" / "weather_daily.csv"
        if not weather_path.exists():
            return []
        try:
            with weather_path.open("r", encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle))
        except Exception:
            return []

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        city_name = str(params.get("city", "")).strip()
        date = str(params.get("date", "")).strip()

        rows = self._load_weather_rows(city_name)
        if not rows:
            return self.fields["missing_city_db"]
        for row in rows:
            if str(row.get("date", "")).strip() != date:
                continue
            condition = classify_weather_condition(row)
            advisory = build_weather_advisory(row)
            result = {
                "city": city_name,
                "date": date,
                **(condition or {}),
                "travel_advisory": advisory,
            }
            return self.format_result_as_json(result)

        try:
            requested_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            requested_date = None

        if requested_date is not None:
            dated_rows: list[tuple[int, Any, dict[str, str]]] = []
            for row in rows:
                row_date_text = str(row.get("date", "")).strip()
                try:
                    row_date = datetime.strptime(row_date_text, "%Y-%m-%d").date()
                except ValueError:
                    continue
                delta_days = abs((row_date - requested_date).days)
                if delta_days <= 3:
                    dated_rows.append((delta_days, row_date, row))

            if dated_rows:
                dated_rows.sort(key=lambda item: (item[0], item[1]))
                _, fallback_date, row = dated_rows[0]
                condition = classify_weather_condition(row)
                advisory = build_weather_advisory(row)
                result = {
                    "city": city_name,
                    "date": str(fallback_date),
                    "requested_date": date,
                    "date_fallback": True,
                    "fallback_reason": "exact weather date missing; nearest available city weather within 3 days",
                    **(condition or {}),
                    "travel_advisory": advisory,
                }
                return self.format_result_as_json(result)

        return self.fields["not_found"](city_name, date)
