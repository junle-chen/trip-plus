"""Database-grounding checks for itinerary feasibility."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..utils import (
    get_index_record,
    iter_hotel_acts,
    iter_intercity_public_acts,
    iter_meal_acts,
    normalize_entity_name,
    slot_to_minutes,
)


def _rounded_price(value: Any) -> Optional[int]:
    try:
        return int(round(float(str(value).strip())))
    except Exception:
        return None

def _allowed_price_values(record: Dict[str, Any], primary_field: str) -> List[int]:
    values: List[int] = []
    for value in [record.get(primary_field), *(record.get("allowed_prices") or [])]:
        rounded = _rounded_price(value)
        if rounded is not None and rounded not in values:
            values.append(rounded)
    return values

def _price_matches_record(plan_price: Any, record: Dict[str, Any], primary_field: str) -> Tuple[bool, Optional[int], List[int]]:
    plan_rounded = _rounded_price(plan_price)
    allowed = _allowed_price_values(record, primary_field)
    if plan_rounded is None:
        return False, None, allowed
    return plan_rounded in allowed, plan_rounded, allowed

def check_hotels_from_search(daily_plans: List[Dict[str, Any]], hotels_index: Dict[str, Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Check if all hotels are from search results and prices match."""
    if not hotels_index:
        return False, "Hotel database failed to load or is empty"
    not_found: List[str] = []
    price_mismatch: List[str] = []
    
    # 1. Check accommodation field: check name and price
    for idx, day in enumerate(daily_plans):
        accom = day.get("accommodation")
        if isinstance(accom, dict):
            name = (accom.get("name") or "").strip()
            # Last day's name if "-" then skip
            if idx == len(daily_plans) - 1 and name == "-":
                continue
            if not name:
                continue
            # Check if name is in database
            hotel_record = get_index_record(hotels_index, name)
            if not hotel_record:
                not_found.append(name)
                continue
            # Check price
            price_val = accom.get("price") or accom.get("cost") or accom.get("price_per_night")
            price_str = hotel_record.get("price_per_night")
            if price_str:
                if isinstance(price_val, (int, float)):
                    matches, _plan_price, allowed = _price_matches_record(price_val, hotel_record, "price_per_night")
                    if not matches:
                        expected = allowed[0] if len(allowed) == 1 else allowed
                        price_mismatch.append(f"{name}: plan has {price_val} ≠ database {expected}")
                else:
                    price_mismatch.append(f"{name}: plan missing valid price/cost")
    
    # 2. Check hotel activities: only check name (not price)
    for idx, day in enumerate(daily_plans[:-1]):  # Except last day
        for act, details, name in iter_hotel_acts([day]):
            name = (name or "").strip()
            if not name:
                continue
            # Only check if name is in database
            if not get_index_record(hotels_index, name):
                not_found.append(name)
    
    if not_found:
        return False, f"Hotels not in database: {sorted(set(not_found))}"
    if price_mismatch:
        return False, f"Hotel price mismatch: {price_mismatch}"
    return True, None

def check_attractions_from_search(daily_plans: List[Dict[str, Any]], attractions_index: Dict[str, Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Check if all attractions are from search results and prices match."""
    if not attractions_index:
        return False, "Attraction database failed to load or is empty"
    not_found: List[str] = []
    cost_mismatch: List[str] = []
    paid_attractions_by_day: Dict[int, set[str]] = {}

    for day_idx, day in enumerate(daily_plans):
        for act in day.get("activities", []) or []:
            if act.get("type") != "attraction":
                continue
            details = act.get("details") or {}
            name = (details.get("name") or "").strip()
            attraction_record = get_index_record(attractions_index, name)
            if not name or not attraction_record:
                not_found.append(name or "<empty>")
                continue
            ticket_price = attraction_record.get("ticket_price")
            plan_cost = details.get("cost")
            if ticket_price is None or ticket_price == "":
                continue
            if isinstance(plan_cost, (int, float)):
                matches, plan_price, allowed = _price_matches_record(plan_cost, attraction_record, "ticket_price")
                entity_key = normalize_entity_name(attraction_record.get("attraction_name") or name).casefold()
                if matches:
                    paid_attractions_by_day.setdefault(day_idx, set()).add(entity_key)
                    continue
                already_paid_same_day = entity_key in paid_attractions_by_day.get(day_idx, set())
                if plan_price == 0 and already_paid_same_day:
                    continue
                expected = allowed[0] if len(allowed) == 1 else allowed
                cost_mismatch.append(f"{name}: plan has {plan_cost} ≠ database {expected}")
            else:
                cost_mismatch.append(f"{name}: plan missing valid cost")
    if not_found:
        return False, f"Attractions not in database: {sorted(set(not_found))}"
    if cost_mismatch:
        return False, f"Attraction price mismatch: {cost_mismatch}"
    return True, None

def check_meals_from_search(daily_plans: List[Dict[str, Any]], restaurants_index: Dict[str, Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Check if all meals are from search results and prices match."""
    if not restaurants_index:
        return False, "Restaurant database failed to load or is empty"

    not_found: List[str] = []
    cost_mismatch: List[str] = []

    for _act, details, name in iter_meal_acts(daily_plans):
        cost_val = details.get("cost")

        restaurant_record = get_index_record(restaurants_index, name)
        if not name or not restaurant_record:
            not_found.append(name or "<empty>")
            continue

        price_str = restaurant_record.get("price_per_person")
        if not price_str:
            continue
        if isinstance(cost_val, (int, float)):
            matches, _plan_price, allowed = _price_matches_record(cost_val, restaurant_record, "price_per_person")
            if not matches:
                expected = allowed[0] if len(allowed) == 1 else allowed
                cost_mismatch.append(f"{name}: plan has {cost_val} ≠ database {expected}")
        else:
            cost_mismatch.append(f"{name}: plan missing valid cost")

    if not_found:
        return False, f"Restaurants not in database: {sorted(set(not_found))}"
    if cost_mismatch:
        return False, f"Restaurant price per person mismatch: {cost_mismatch}"
    return True, None

def check_intercity_public_from_search(
    daily_plans: List[Dict[str, Any]], 
    flights_index: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    trains_index: Optional[Dict[str, List[Dict[str, Any]]]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Check if intercity public transport data is valid and comes from database.
    
    Validates each intercity public segment against one exact database row:
    1. Required fields exist (number, from, to, cost, time_slot)
    2. Flight/train number exists in database (flights.csv or trains.csv)
    3. Departure/arrival station names and times match the same DB row
    4. Price matches the segment price or the materialized route total
    """
    intercity_missing: List[str] = []
    not_found: List[str] = []
    price_mismatch: List[str] = []
    segment_mismatch: List[str] = []
    
    required_fields = ("number", "from", "to", "cost")

    def _normalize_point(value: Any) -> str:
        return normalize_entity_name(str(value or "").strip())

    def _time_minutes_from_datetime(value: Any) -> Optional[int]:
        text = str(value or "").strip()
        if not text:
            return None
        match = re.search(r"\b(\d{1,2}):(\d{2})(?::\d{2})?\b", text)
        if not match:
            return None
        try:
            return int(match.group(1)) * 60 + int(match.group(2))
        except (TypeError, ValueError):
            return None

    def _record_times(record: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        dep = _time_minutes_from_datetime(record.get("dep_datetime"))
        arr = _time_minutes_from_datetime(record.get("arr_datetime"))
        if dep is None or arr is None:
            return dep, arr
        dep_date = str(record.get("dep_datetime") or "")[:10]
        arr_date = str(record.get("arr_datetime") or "")[:10]
        if arr_date and dep_date and arr_date > dep_date:
            arr += 24 * 60
        elif arr < dep:
            arr += 24 * 60
        return dep, arr

    def _price_matches(plan_cost: Optional[float], record: Dict[str, Any]) -> bool:
        if plan_cost is None:
            return True
        plan_cost_rounded = int(round(plan_cost))
        candidate_prices = [record.get("price")]
        candidate_prices.extend(record.get("route_prices") or [])
        for db_price in candidate_prices:
            try:
                if plan_cost_rounded == int(round(float(db_price))):
                    return True
            except (TypeError, ValueError):
                continue
        segment_prices = record.get("route_segment_prices")
        if isinstance(segment_prices, dict) and segment_prices:
            possible = {0}
            for segment_index in sorted(segment_prices):
                next_possible = {
                    base + int(round(float(price)))
                    for base in possible
                    for price in (segment_prices.get(segment_index) or [])
                    if base + int(round(float(price))) <= plan_cost_rounded
                }
                if not next_possible:
                    return False
                possible = next_possible
            return plan_cost_rounded in possible
        return False

    def _record_matches(details: Dict[str, Any], record: Dict[str, Any], plan_cost: Optional[float]) -> bool:
        start_min, end_min = slot_to_minutes(str(details.get("time_slot") or ""))
        if start_min is None or end_min is None:
            return False
        dep_min, arr_min = _record_times(record)
        if dep_min is None or arr_min is None:
            return False
        if start_min != dep_min or end_min != arr_min:
            return False
        if _normalize_point(details.get("from")) != _normalize_point(record.get("dep_station_name")):
            return False
        if _normalize_point(details.get("to")) != _normalize_point(record.get("arr_station_name")):
            return False
        return _price_matches(plan_cost, record)

    def _record_label(record: Dict[str, Any]) -> str:
        dep_min, arr_min = _record_times(record)
        def fmt(minutes: Optional[int]) -> str:
            if minutes is None:
                return "?"
            minutes = minutes % (24 * 60)
            return f"{minutes // 60:02d}:{minutes % 60:02d}"
        return (
            f"{record.get('dep_station_name')} {fmt(dep_min)} -> "
            f"{record.get('arr_station_name')} {fmt(arr_min)} @ {record.get('price')}"
        )
    
    for act, details in iter_intercity_public_acts(daily_plans):
        # Step 1: Check required fields exist
        missing = [k for k in required_fields if details.get(k) in (None, "")]
        if not act.get("time_slot"):
            missing.append("time_slot")
        if missing:
            intercity_missing.append(f"{act.get('time_slot') or '<no time_slot>'}: missing {missing}")
            continue
        
        number = str(details.get("number")).strip()
        
        try:
            plan_cost = float(details.get("cost"))
        except (ValueError, TypeError):
            plan_cost = None
        
        details_with_time = {**details, "time_slot": act.get("time_slot")}

        # Step 2: Check if number exists in database (if indices provided)
        if flights_index is None and trains_index is None:
            # No database provided, skip database validation
            continue
        
        found_in_flights = flights_index and number in flights_index
        found_in_trains = trains_index and number in trains_index
        
        if not found_in_flights and not found_in_trains:
            not_found.append(number)
            continue

        # Step 3: Require a single same-number row to match stations, times, and price.
        records: List[Dict[str, Any]] = []
        if found_in_flights:
            records.extend(flights_index[number])
        if found_in_trains:
            records.extend(trains_index[number])
        if not any(_record_matches(details_with_time, record, plan_cost) for record in records):
            same_station_time = [
                record for record in records
                if _normalize_point(details.get("from")) == _normalize_point(record.get("dep_station_name"))
                and _normalize_point(details.get("to")) == _normalize_point(record.get("arr_station_name"))
                and slot_to_minutes(str(act.get("time_slot") or "")) == _record_times(record)
            ]
            if same_station_time and plan_cost is not None:
                price_mismatch.append(
                    f"{number}: station/time match but plan cost {plan_cost} does not match DB prices"
                )
            else:
                candidates = "; ".join(_record_label(record) for record in records[:5])
                segment_mismatch.append(
                    f"{number} {details.get('from')} {act.get('time_slot')} -> {details.get('to')} "
                    f"does not match any DB segment; candidates: {candidates}"
                )

    # Compile error message
    error_parts = []
    if intercity_missing:
        error_parts.append(f"Missing fields: {intercity_missing}")
    if not_found:
        error_parts.append(f"Not found in database: {sorted(set(not_found))}")
    if price_mismatch:
        error_parts.append(f"Price mismatch: {price_mismatch}")
    if segment_mismatch:
        error_parts.append(f"Segment mismatch: {segment_mismatch[:10]}")
    
    if error_parts:
        return False, "; ".join(error_parts)
    return True, None


# ==============================================================================
# DIMENSION 3: Daily Stay Traceability
# Checks: traceable_accommodation, ends_with_accommodation
# ==============================================================================
