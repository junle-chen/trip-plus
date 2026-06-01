"""Shared evidence helpers for database-backed hard constraints."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List

from query_generation.common import (
    extract_cuisine_label,
    safe_float,
    safe_int,
    split_semicolon_field,
)

Rows = List[Dict[str, str]]
Options = List[Dict[str, Any]]

# Format an inclusive hour window for visible transport constraints.
def hour_window_text(start_hour: int, end_hour: int) -> str:
    return f"{start_hour:02d}:00-{end_hour:02d}:59"


# Deduplicate strings while keeping deterministic source order.
def unique_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


# Deduplicate option dictionaries by a caller-provided stable key.
def unique_dicts_preserve_order(values: List[Dict[str, Any]], key_fn) -> List[Dict[str, Any]]:
    seen = set()
    ordered: List[Dict[str, Any]] = []
    for value in values:
        key = key_fn(value)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


LOW_QUALITY_POI_NAME_PATTERN = re.compile(
    r"under construction|managed by|engineering cognition|forestry\s*\(?garden management\)? bureau|"
    r"campus|children'?s playground|family amusement|parent-child|aquarium park|"
    r"underwater city|wuyue plaza|joy city|wanda|fish catching.*(park|playground)"
    r"|catch.*fish.*(park|playground)",
    flags=re.I,
)

ATTRACTION_TYPE_LABELS = (
    ("museum", "museum"),
    ("memorial", "memorial"),
    ("gallery", "art gallery"),
    ("art", "art venue"),
    ("park", "park"),
    ("square", "city square"),
    ("plaza", "city square"),
    ("shopping", "shopping district"),
    ("ancient town", "historic old town"),
    ("old town", "historic old town"),
    ("historic", "history and culture"),
    ("history", "history and culture"),
    ("temple", "history and culture"),
    ("scenic", "scenic nature"),
    ("nature", "scenic nature"),
    ("mountain", "scenic nature"),
    ("amusement", "theme park"),
    ("theme", "theme park"),
    ("zoo", "family attraction"),
    ("aquarium", "family attraction"),
    ("landmark", "city landmark"),
    ("sightseeing", "city landmark"),
    ("cultural", "cultural site"),
)


# Filter noisy POI names that should not become visible constraints.
def is_low_quality_poi_name(name: str) -> bool:
    return bool(LOW_QUALITY_POI_NAME_PATTERN.search(str(name or "")))


# Keep attraction rows that are usable as anchors or must-visit places.
def is_quality_attraction_row(row: Dict[str, str]) -> bool:
    name = str(row.get("attraction_name") or "")
    if not name or is_low_quality_poi_name(name):
        return False
    return True


# Convert train rows into unique evaluator-facing option payloads.
def train_options(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    return unique_dicts_preserve_order(
        [train_option(row) for row in rows],
        lambda item: (item["train_no"], item["route_index"], item["segment_index"]),
    )


# Convert flight rows into unique evaluator-facing option payloads.
def flight_options(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    return unique_dicts_preserve_order(
        [flight_option(row) for row in rows],
        lambda item: (item["flight_no"], item["route_index"], item["segment_index"]),
    )


# Normalize one train CSV row for constraint evidence.
def train_option(row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "train_no": row.get("train_no", ""),
        "route_index": safe_int(row.get("route_index")),
        "segment_index": safe_int(row.get("segment_index")),
        "train_type": row.get("train_type", ""),
        "seat_class": row.get("seat_class", ""),
        "price": safe_float(row.get("price")),
        "dep_datetime": row.get("dep_datetime", ""),
        "arr_datetime": row.get("arr_datetime", ""),
        "origin_city": row.get("origin_city", ""),
        "destination_city": row.get("destination_city", ""),
    }


# Normalize one flight CSV row for constraint evidence.
def flight_option(row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "flight_no": row.get("flight_no", ""),
        "route_index": safe_int(row.get("route_index")),
        "segment_index": safe_int(row.get("segment_index")),
        "airline": row.get("airline", ""),
        "seat_class": row.get("seat_class", ""),
        "price": safe_float(row.get("price")),
        "dep_datetime": row.get("dep_datetime", ""),
        "arr_datetime": row.get("arr_datetime", ""),
        "manufacturer": row.get("manufacturer", ""),
        "origin_city": row.get("origin_city", ""),
        "destination_city": row.get("destination_city", ""),
    }


# Normalize one hotel CSV row for constraint evidence.
def hotel_option(row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "hotel_name": row.get("name", ""),
        "hotel_price": safe_float(row.get("price")),
        "hotel_score": safe_float(row.get("score")),
        "hotel_star": safe_int(row.get("hotel_star")),
        "brand": row.get("brand", ""),
        "decoration_time": safe_int(row.get("decoration_time")),
        "services": split_semicolon_field(row.get("services", "")),
    }


# Normalize one restaurant CSV row for constraint evidence.
def restaurant_option(row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "restaurant_name": row.get("restaurant_name", ""),
        "price_per_person": safe_float(row.get("price_per_person")),
        "restaurant_rating": safe_float(row.get("rating")),
        "nearby_attraction_name": row.get("nearby_attraction_name", ""),
        "cuisine_type": extract_cuisine_label(row.get("cuisine", "")),
        "tags": split_semicolon_field(row.get("tags", "")),
        "opening_time": row.get("opening_time", ""),
        "closing_time": row.get("closing_time", ""),
    }


# Normalize one attraction CSV row for constraint evidence.
def attraction_option(row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "attraction_name": row.get("attraction_name", ""),
        "attraction_type": row.get("attraction_type", ""),
        "rating": safe_float(row.get("rating")),
        "ticket_price": safe_float(row.get("ticket_price")),
        "ticket_price_source": row.get("ticket_price_source", ""),
        "opening_time": row.get("opening_time", ""),
        "closing_time": row.get("closing_time", ""),
        "closing_dates": row.get("closing_dates", ""),
        "popularity_tags": split_semicolon_field(row.get("popularity_tags", "")),
        "crowd_risk": row.get("crowd_risk", ""),
        "queue_risk": row.get("queue_risk", ""),
    }


# Extract ordered names for evaluator acceptable lists.
def row_names(rows: Rows, field: str) -> List[str]:
    return unique_preserve_order([row[field] for row in rows])


# Convert rows into stable evaluator option payloads.
def option_payloads(
    rows: Rows,
    option_fn: Callable[[Dict[str, str]], Dict[str, Any]],
    key_field: str,
) -> Options:
    return unique_dicts_preserve_order(
        [option_fn(row) for row in rows],
        lambda item: item[key_field],
    )


# Keep rows whose numeric field matches a sampled superlative value.
def rows_with_float(rows: Rows, field: str, value: float, default: float = 0.0) -> Rows:
    return [row for row in rows if safe_float(row.get(field), default) == value]


# Keep rows that contain a semicolon-delimited label in the requested field.
def rows_with_label(rows: Rows, field: str, label: str) -> Rows:
    return [row for row in rows if label in split_semicolon_field(row.get(field, ""))]


def hotel_options(rows: Rows) -> Options:
    return option_payloads(rows, hotel_option, "hotel_name")


def restaurant_options(rows: Rows) -> Options:
    return option_payloads(rows, restaurant_option, "restaurant_name")


def attraction_options(rows: Rows) -> Options:
    return option_payloads(rows, attraction_option, "attraction_name")


# Exclude synthetic non-ticket POIs from ticket-sensitive attraction sampling.
def is_non_ticket_attraction(row: Dict[str, str]) -> bool:
    return str(row.get("ticket_price_source", "")).strip() == "estimated_rule_non_ticket_poi"


# Collapse raw attraction taxonomy strings into user-readable labels.
def friendly_attraction_type(raw_type: str) -> str:
    text = str(raw_type or "").strip()
    lower = text.lower()
    if not text or lower in {"other", "others", "misc", "miscellaneous"}:
        return ""
    for marker, label in ATTRACTION_TYPE_LABELS:
        if marker in lower:
            return label
    if ";" in text:
        return ""
    return text if len(text) <= 40 else ""
