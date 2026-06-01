"""City-level environmental context derived from sample database evidence."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from query_generation.common import BASE_DIR, extract_cuisine_label, safe_float


DEFAULT_CITY_TAGS_PATH = BASE_DIR / "database" / "en" / "city_tags.json"
DEFAULT_CITY_PLANNING_TAGS_PATH = BASE_DIR / "database" / "en" / "city_planning_tags.json"

HIGH_ALTITUDE_CITY_FOLDERS = {"lhasa", "xining"}
COLD_WINTER_CITY_FOLDERS = {"harbin", "changchun", "shenyang", "hohhot", "urumqi", "xining", "lhasa"}
SUMMER_HEAT_CITY_FOLDERS = {
    "chongqing",
    "wuhan",
    "changsha",
    "nanchang",
    "nanjing",
    "fuzhou",
    "guangzhou",
    "shenzhen",
    "hong_kong",
    "sanya",
    "xiamen",
    "nanning",
    "hangzhou",
}
COASTAL_HUMID_CITY_FOLDERS = {"hong_kong", "guangzhou", "shenzhen", "xiamen", "fuzhou", "sanya", "zhuhai", "quanzhou"}
_CITY_TAG_CATALOG_CACHE: dict[str, Any] | None = None


def _season_for_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def load_city_planning_tag_catalog(path: Path | None = None) -> dict[str, Any]:
    catalog_path = path or DEFAULT_CITY_PLANNING_TAGS_PATH
    if not catalog_path.exists():
        return {"cities": {}}
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def _merge_unique_strings(left: list[Any], right: list[Any]) -> list[str]:
    values = [str(item).strip() for item in list(left or []) + list(right or []) if str(item).strip()]
    return sorted(set(values))


def _merge_city_planning_tags(base_catalog: dict[str, Any], supplement: dict[str, Any]) -> None:
    base_catalog.setdefault("planning_tag_definitions", {}).update(supplement.get("tag_definitions", {}) or {})
    base_cities = base_catalog.setdefault("cities", {})
    for folder, extra in (supplement.get("cities", {}) or {}).items():
        if not isinstance(extra, dict):
            continue
        city = base_cities.setdefault(folder, {"city_name": extra.get("city_name", folder)})
        planning_tags = list(extra.get("planning_tags", []) or [])
        city["base_tags"] = _merge_unique_strings(city.get("base_tags", []), planning_tags)
        city["planning_tags"] = _merge_unique_strings(city.get("planning_tags", []), planning_tags)
        for key in ("generation_hints", "travel_tips"):
            city[key] = list(city.get(key, []) or []) + list(extra.get(key, []) or [])
        city["evaluable_planning_checks"] = list(city.get("evaluable_planning_checks", []) or []) + list(
            extra.get("evaluable_planning_checks", []) or []
        )
        city["references"] = list(city.get("references", []) or []) + list(extra.get("references", []) or [])
        city["feedback_triggers"] = list(city.get("feedback_triggers", []) or []) + list(extra.get("feedback_triggers", []) or [])


def load_city_tag_catalog(path: Path | None = None) -> dict[str, Any]:
    global _CITY_TAG_CATALOG_CACHE
    catalog_path = path or DEFAULT_CITY_TAGS_PATH
    if _CITY_TAG_CATALOG_CACHE is not None and catalog_path == DEFAULT_CITY_TAGS_PATH:
        return _CITY_TAG_CATALOG_CACHE
    if not catalog_path.exists():
        catalog: dict[str, Any] = {"cities": {}}
    else:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog_path == DEFAULT_CITY_TAGS_PATH:
        _merge_city_planning_tags(catalog, load_city_planning_tag_catalog())
        _CITY_TAG_CATALOG_CACHE = catalog
    return catalog


def _city_tag_profile(city_folder: str | None, city_name: str | None) -> dict[str, Any] | None:
    cities = load_city_tag_catalog().get("cities", {}) or {}
    folder = str(city_folder or "").strip()
    if folder and folder in cities:
        return dict(cities[folder])
    wanted_name = str(city_name or "").strip()
    if wanted_name:
        for profile in cities.values():
            if str(profile.get("city_name", "")).strip() == wanted_name:
                return dict(profile)
    return None


def _fallback_city_tags(city_folder: str | None, city_name: str | None, month: int) -> list[str]:
    folder = str(city_folder or "").strip()
    tags = []
    if folder in HIGH_ALTITUDE_CITY_FOLDERS:
        tags.append("high_altitude_city")
    if folder in COLD_WINTER_CITY_FOLDERS:
        tags.append("cold_winter_city")
    if folder in SUMMER_HEAT_CITY_FOLDERS:
        tags.append("summer_heat_city")
    if folder in COASTAL_HUMID_CITY_FOLDERS:
        tags.append("coastal_humid_heat")
    if city_name in {"Harbin", "Changchun", "Shenyang"}:
        tags.append("ice_snow_winter_destination")
    if folder in HIGH_ALTITUDE_CITY_FOLDERS and month in {11, 12, 1, 2, 3}:
        tags.append("plateau_winter_exposure")
    return sorted(set(tags))


def _infer_city_tags(city_folder: str | None, city_name: str | None, month: int) -> list[str]:
    profile = _city_tag_profile(city_folder, city_name)
    if profile:
        return sorted(set(str(tag) for tag in profile.get("base_tags", []) or [] if str(tag).strip()))
    return _fallback_city_tags(city_folder, city_name, month)


def _catalog_seasonal_advisories(city_profile: dict[str, Any] | None, month: int) -> list[str]:
    if not city_profile:
        return []
    seasonal = city_profile.get("seasonal_advisories", {}) or {}
    season = _season_for_month(month)
    tags = list(seasonal.get("all", []) or []) + list(seasonal.get(season, []) or [])
    return sorted(set(str(tag) for tag in tags if str(tag).strip()))


def _trigger_applies(trigger: dict[str, Any], month: int) -> bool:
    season = _season_for_month(month)
    trigger_seasons = trigger.get("season", "all")
    if isinstance(trigger_seasons, str):
        trigger_seasons = [trigger_seasons]
    trigger_seasons = {str(item) for item in trigger_seasons}
    return "all" in trigger_seasons or season in trigger_seasons


def _reference_map(city_profile: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not city_profile:
        return {}
    references = city_profile.get("references", []) or []
    return {
        str(item.get("id")): {
            "id": item.get("id"),
            "title": item.get("title"),
            "url": item.get("url"),
            "evidence": item.get("evidence"),
        }
        for item in references
        if item.get("id")
    }


def _catalog_feedback_triggers(city_profile: dict[str, Any] | None, month: int) -> list[dict[str, Any]]:
    if not city_profile:
        return []
    refs = _reference_map(city_profile)
    triggers = []
    for trigger in city_profile.get("feedback_triggers", []) or []:
        if not isinstance(trigger, dict) or not _trigger_applies(trigger, month):
            continue
        trigger_refs = [refs[ref_id] for ref_id in trigger.get("reference_ids", []) or [] if ref_id in refs]
        triggers.append(
            {
                "type": trigger.get("type"),
                "season": trigger.get("season", "all"),
                "message": trigger.get("message"),
                "references": trigger_refs,
            }
        )
    return triggers


def _date_range(start_date: str, end_date: str) -> list[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    days = []
    current = start
    while current <= end:
        days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def _infer_date_tags(start_date: str, end_date: str, holiday_info: dict[str, Any]) -> list[str]:
    days = _date_range(start_date, end_date)
    months = {datetime.strptime(day, "%Y-%m-%d").month for day in days}
    tags = []
    if months & {12, 1, 2}:
        tags.append("season_winter")
    if months & {3, 4, 5}:
        tags.append("season_spring")
    if months & {6, 7, 8}:
        tags.append("season_summer")
    if months & {9, 10, 11}:
        tags.append("season_autumn")
    if holiday_info.get("weekend_days"):
        tags.append("weekend_overlap")
    for holiday_tag in holiday_info.get("holiday_tags", []) or []:
        tags.append(f"holiday_{holiday_tag}")
    return sorted(set(tags))


def infer_holiday_signals(start_date: str, end_date: str) -> dict[str, Any]:
    days = _date_range(start_date, end_date)
    weekend_days = []
    holiday_tags = []
    for day_str in days:
        dt = datetime.strptime(day_str, "%Y-%m-%d").date()
        if dt.weekday() >= 5:
            weekend_days.append(day_str)
        month_day = dt.strftime("%m-%d")
        if month_day in {"01-01"}:
            holiday_tags.append("new_year")
        if month_day in {"05-01", "05-02", "05-03", "05-04", "05-05"}:
            holiday_tags.append("labor_day_window")
        if month_day in {"10-01", "10-02", "10-03", "10-04", "10-05", "10-06", "10-07"}:
            holiday_tags.append("national_day_window")
        if dt.month in {7, 8}:
            holiday_tags.append("summer_vacation")
        if dt.month in {1, 2}:
            holiday_tags.append("winter_break")
    return {
        "weekend_days": weekend_days,
        "holiday_tags": sorted(set(holiday_tags)),
    }


def derive_city_context(
    db: dict[str, list[dict[str, str]]],
    depart_date: str,
    *,
    city_name: str | None = None,
    city_folder: str | None = None,
) -> dict[str, Any]:
    attractions = sorted(
        [
            row
            for row in db.get("attractions", [])
            if str(row.get("ticket_price_source", "")).strip() != "estimated_rule_non_ticket_poi"
        ],
        key=lambda row: (safe_float(row.get("rating")), -safe_float(row.get("ticket_price"), 999999.0)),
        reverse=True,
    )
    signature_scenery = []
    seen_attr = set()
    for row in attractions:
        name = str(row.get("attraction_name", "")).strip()
        if name and name not in seen_attr:
            signature_scenery.append(name)
            seen_attr.add(name)
        if len(signature_scenery) >= 4:
            break

    high_crowd_attractions = []
    high_queue_attractions = []
    popularity_counter: Counter[str] = Counter()
    for row in db.get("attractions", []):
        name = str(row.get("attraction_name", "")).strip()
        if not name:
            continue
        for tag in str(row.get("popularity_tags", "")).split(";"):
            tag = tag.strip()
            if tag:
                popularity_counter[tag] += 1
        crowd_risk = str(row.get("crowd_risk", "")).strip()
        queue_risk = str(row.get("queue_risk", "")).strip()
        if crowd_risk == "high" and len(high_crowd_attractions) < 8:
            high_crowd_attractions.append(name)
        if queue_risk == "high" and len(high_queue_attractions) < 8:
            high_queue_attractions.append(name)

    cuisine_counter: Counter[str] = Counter()
    for row in db.get("restaurants", []):
        cuisine = extract_cuisine_label(str(row.get("cuisine", "")))
        if cuisine:
            cuisine_counter[cuisine] += 1
    signature_cuisines = [name for name, _ in cuisine_counter.most_common(4)]

    month = int(depart_date.split("-")[1])
    city_profile = _city_tag_profile(city_folder, city_name)
    city_tags = _infer_city_tags(city_folder, city_name, month)
    travel_tips = list(city_profile.get("travel_tips", []) or []) if city_profile else []
    seasonal_advisories = _catalog_seasonal_advisories(city_profile, month)
    if month == 1 and ("cold_winter_city" in city_tags or "high_altitude_city" in city_tags):
        travel_tips.append("In winter, keep outdoor exposure short and leave enough warm indoor breaks.")
        seasonal_advisories.append("winter_outdoor_exposure")
    elif month == 1 and ("coastal_humid_heat" in city_tags or "summer_heat_city" in city_tags):
        travel_tips.append("In warm coastal cities, winter trips can use comfortable outdoor time without overpacking the day.")
        seasonal_advisories.append("winter_warm_escape")
    if month == 7:
        travel_tips.append("In summer, avoid the strongest midday sun for outdoor activities.")
        seasonal_advisories.append("extreme_heat")
    if "high_altitude_city" in city_tags:
        travel_tips.append("For high-altitude cities, keep arrival-day intensity moderate and avoid long continuous outdoor blocks.")
        seasonal_advisories.append("high_altitude_adaptation")
    if "cold_winter_city" in city_tags and month in {11, 12, 1, 2, 3}:
        travel_tips.append("Cold winter cities should avoid long continuous outdoor exposure.")
        seasonal_advisories.append("winter_outdoor_exposure")
    if "summer_heat_city" in city_tags and month in {6, 7, 8, 9}:
        travel_tips.append("Hot summer cities need lower midday outdoor intensity.")
        seasonal_advisories.append("extreme_heat")
    return {
        "city_name": city_name,
        "city_folder": city_folder,
        "city_tags": city_tags,
        "signature_scenery": signature_scenery,
        "signature_cuisines": signature_cuisines,
        "high_crowd_attractions": high_crowd_attractions,
        "high_queue_attractions": high_queue_attractions,
        "popularity_tag_counts": dict(popularity_counter.most_common(10)),
        "travel_tips": travel_tips,
        "generation_hints": list(city_profile.get("generation_hints", []) or []) if city_profile else [],
        "evaluable_planning_checks": list(city_profile.get("evaluable_planning_checks", []) or []) if city_profile else [],
        "seasonal_advisories": sorted(set(seasonal_advisories)),
        "city_tag_references": list(_reference_map(city_profile).values()),
        "feedback_triggers": _catalog_feedback_triggers(city_profile, month),
        "cultural_highlights": [],
    }


def load_weather_rows(city_db_root: Path, dest_folder: str) -> list[dict[str, str]]:
    path = city_db_root / dest_folder / "weather" / "weather_daily.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def derive_environmental_grounding(
    *,
    city_db_root: Path,
    option: Any,
    city_context: dict[str, Any],
) -> dict[str, Any]:
    weather_rows = load_weather_rows(city_db_root, option.dest_folder)
    weather_by_date = {str(row.get("date", "")).strip(): row for row in weather_rows}
    selected_dates = _date_range(option.depart_date, option.return_date)
    selected_weather = [weather_by_date[day] for day in selected_dates if day in weather_by_date]

    temp_max_values = [safe_float(row.get("temperature_max_c")) for row in selected_weather]
    temp_min_values = [safe_float(row.get("temperature_min_c")) for row in selected_weather]
    precip_values = [safe_float(row.get("precipitation_mm")) for row in selected_weather]
    conditions = [str(row.get("weather_condition", "")).strip() for row in selected_weather if str(row.get("weather_condition", "")).strip()]

    holiday_info = infer_holiday_signals(option.depart_date, option.return_date)
    date_tags = _infer_date_tags(option.depart_date, option.return_date, holiday_info)
    notes = []
    weather_tags = []
    if temp_max_values:
        avg_max = round(sum(temp_max_values) / len(temp_max_values), 1)
        avg_min = round(sum(temp_min_values) / len(temp_min_values), 1)
        notes.append(f"Expected daytime high is about {avg_max}C, and nighttime low is about {avg_min}C.")
        if avg_max >= 32:
            weather_tags.append("hot_daytime")
        if avg_max >= 35:
            weather_tags.append("extreme_heat")
        if avg_min <= 0:
            weather_tags.append("freezing_risk")
        if avg_max <= 5:
            weather_tags.append("cold_daytime")
    if max(precip_values or [0.0]) >= 5.0:
        notes.append("There is notable precipitation risk during the trip, so outdoor plans need alternatives.")
        weather_tags.append("rain_risk")
    if any("thunder" in item.lower() or "storm" in item.lower() for item in conditions):
        notes.append("The weather includes thunderstorm signals, so outdoor and transport planning should be conservative.")
        weather_tags.append("thunderstorm_risk")
    if holiday_info["weekend_days"]:
        notes.append("The trip includes weekend days, so popular areas may be more crowded.")
    if holiday_info["holiday_tags"]:
        notes.append("The dates fall near a holiday window, so price and crowding should be considered.")
    if "winter_outdoor_exposure" in set(city_context.get("seasonal_advisories", []) or []):
        notes.append("The city has winter outdoor exposure risk, so continuous outdoor activities should not be too long.")
    if "extreme_heat" in set(city_context.get("seasonal_advisories", []) or []):
        notes.append("The city has heat and sun-exposure risk, so midday outdoor intensity should be reduced.")
    if "high_altitude_adaptation" in set(city_context.get("seasonal_advisories", []) or []):
        notes.append("The city has altitude-adaptation risk, so arrival-day activities should stay moderate.")
        weather_tags.append("high_altitude_adaptation")

    return {
        "date_range": selected_dates,
        "date_tags": date_tags,
        "weather_tags": sorted(set(weather_tags)),
        "city_tags": list(city_context.get("city_tags", []) or []),
        "weather_daily": [
            {
                "date": row.get("date"),
                "condition": row.get("weather_condition"),
                "temperature_max_c": safe_float(row.get("temperature_max_c")),
                "temperature_min_c": safe_float(row.get("temperature_min_c")),
                "precipitation_mm": safe_float(row.get("precipitation_mm")),
            }
            for row in selected_weather
        ],
        "avg_temp_max_c": round(sum(temp_max_values) / len(temp_max_values), 1) if temp_max_values else None,
        "avg_temp_min_c": round(sum(temp_min_values) / len(temp_min_values), 1) if temp_min_values else None,
        "max_precipitation_mm": round(max(precip_values), 1) if precip_values else 0.0,
        "holiday_signals": holiday_info,
        "seasonal_advisories": list(city_context.get("seasonal_advisories", []) or []),
        "notes": notes,
    }
