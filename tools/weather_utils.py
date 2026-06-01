from __future__ import annotations


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def weather_condition_label_from_code(weather_code: object) -> str:
    code = int(round(to_float(weather_code, -1)))
    if code == 0:
        return "clear"
    if code in {1, 2, 3}:
        return "cloudy"
    if code in {45, 48}:
        return "fog"
    if code in {51, 53, 55, 56, 57}:
        return "drizzle"
    if code in {61, 63, 65, 66, 67, 80, 81, 82}:
        return "rain"
    if code in {71, 73, 75, 77, 85, 86}:
        return "snow"
    if code in {95, 96, 99}:
        return "thunderstorm"
    return "moderate weather"


def normalize_weather_condition(row: dict[str, str] | None) -> str:
    if not row:
        return ""
    explicit = str(row.get("weather_condition", "")).strip()
    if explicit:
        return explicit
    return weather_condition_label_from_code(row.get("weather_code"))
