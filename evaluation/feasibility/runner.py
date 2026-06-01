"""Runner for deterministic itinerary-feasibility evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .budget import check_budget_accuracy
from .config import ALL_FEASIBILITY_CHECKS, FEASIBILITY_CHECK_DIMENSIONS, FEASIBILITY_DIMENSIONS
from .evidence import (
    check_attractions_from_search,
    check_hotels_from_search,
    check_intercity_public_from_search,
    check_meals_from_search,
)
from .structure import (
    check_accommodation_traceable,
    check_attraction_necessity,
    check_day_boundary_continuity,
    check_intercity_transportation_consistency,
    check_last_activity_is_hotel,
    check_meal_necessity,
    check_route_closed_loop,
    check_valid_days,
)
from .timing import (
    check_attractions_duration_reasonable,
    check_attractions_in_opening_hours,
    check_attractions_not_closed,
    check_intercity_buffer_adequacy,
    check_local_move_sanity,
    check_meal_duration_reasonable,
    check_meals_in_business_hours,
    check_time_no_overlap,
    check_transfer_time_reasonable,
)
from ..utils import (
    add_index_entry,
    get_base_dir,
    get_database_dir,
    load_attraction_index,
    load_flights_index,
    load_hotel_index,
    load_location_aliases_into_index,
    load_locations_index,
    load_restaurant_index,
    load_trains_index,
)


def _augment_locations_with_transport_hubs(
    locations_index: Dict[str, Dict[str, Any]],
    database_dir: Optional[Path],
    meta: Dict[str, Any],
) -> None:
    """Add city subway stations to evaluation location index when available.

    Sample-level locations sometimes omit airport/railway hubs while the
    city-level subway file contains terminal/station coordinates. The tools can
    resolve those hubs, so evaluation should use the same deterministic source.
    """
    if database_dir is None:
        return

    city_db_roots: list[Path] = []
    meta_path = get_database_dir(database_dir) / ".build_meta.json"
    if meta_path.exists():
        try:
            build_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            city_db_root_raw = str(build_meta.get("city_db_root", "")).strip()
            if city_db_root_raw:
                city_db_roots.append(Path(city_db_root_raw))
        except Exception:
            pass

    base_dir = get_base_dir()
    for root in (
        base_dir / "database" / "en",
        base_dir / "database" / "database_by_city" / "en",
    ):
        if root not in city_db_roots:
            city_db_roots.append(root)

    try:
        from tools.city_db_access import load_city_subway_stations
    except Exception:
        return

    candidate_cities = []
    for value in [meta.get("org"), *(meta.get("dest") or [])]:
        city = str(value or "").strip()
        if city and city not in candidate_cities:
            candidate_cities.append(city)

    for city in candidate_cities:
        station_rows = []
        for city_db_root in city_db_roots:
            if (city_db_root / "city_index.json").exists():
                station_rows = load_city_subway_stations(city_db_root, city)
                if station_rows:
                    break
        for row in station_rows:
            name = str(row.get("poi_name") or "").strip()
            if not name:
                continue
            payload = {
                "latitude": str(row.get("latitude", "")).strip(),
                "longitude": str(row.get("longitude", "")).strip(),
                "poi_type": str(row.get("poi_type", "subway_station")).strip(),
            }
            add_index_entry(locations_index, name, payload)
    aliases_path = get_database_dir(database_dir) / "locations" / "location_aliases.csv"
    load_location_aliases_into_index(locations_index, aliases_path)


# ----------------------
# Database Path Setup
# ----------------------

_DATABASE_DIR = get_database_dir()

RESTAURANTS_CSV_PATH = str(_DATABASE_DIR / "restaurants" / "restaurants.csv")
HOTELS_CSV_PATH = str(_DATABASE_DIR / "hotels" / "hotels.csv")
ATTRACTIONS_CSV_PATH = str(_DATABASE_DIR / "attractions" / "attractions.csv")
LOCATIONS_COORDS_CSV_PATH = str(_DATABASE_DIR / "locations" / "locations_coords.csv")

# Note: Path validation removed - actual database paths are passed during evaluation

_DATABASE_DIR = get_database_dir()

RESTAURANTS_CSV_PATH = str(_DATABASE_DIR / "restaurants" / "restaurants.csv")
HOTELS_CSV_PATH = str(_DATABASE_DIR / "hotels" / "hotels.csv")
ATTRACTIONS_CSV_PATH = str(_DATABASE_DIR / "attractions" / "attractions.csv")
LOCATIONS_COORDS_CSV_PATH = str(_DATABASE_DIR / "locations" / "locations_coords.csv")

def _score_group(check_names: List[str], check_results: Dict[str, Tuple[bool, Optional[str]]]) -> Dict[str, Any]:
    """Average a set of binary checks and keep failed-check evidence."""
    checks: List[Dict[str, Any]] = []
    passed = 0
    for check_name in check_names:
        ok, message = check_results.get(check_name, (False, "Check not evaluated"))
        checks.append({
            "name": check_name,
            "passed": bool(ok),
            "message": message,
        })
        if ok:
            passed += 1
    total = len(check_names)
    score = (passed / total) if total > 0 else 0.0
    return {
        "score": score,
        "passed": passed,
        "total": total,
        "checks": checks,
    }


# ==============================================================================
# DIMENSION 1: Trip Skeleton and Route Continuity
# Checks: valid_trip_duration, closed_loop_route_structure, seamless_intercity_transfers
# ==============================================================================

def eval_itinerary_feasibility(plan: Dict[str, Any], meta: Dict[str, Any], database_dir: Optional[Path] = None) -> Dict[str, Tuple[bool, Optional[str]]]:
    """
    Evaluate itinerary feasibility checks.
    
    Args:
        plan: Travel plan dictionary
        meta: Metadata dictionary
        database_dir: Database directory path (if specified, will use that sample's database)
    """
    res: Dict[str, Tuple[bool, Optional[str]]] = {}
    daily_plans: List[Dict[str, Any]] = plan.get("daily_plans", []) or []
    
    # If daily_plans is missing, all checks that depend on itinerary will be False with unified reason
    if not daily_plans:
        reason = "Missing daily_plans"
        for check_name in ALL_FEASIBILITY_CHECKS:
            res[check_name] = (False, reason)
        return res
    
    # ==================== Load all database indices ====================
    if database_dir is not None:
        db_dir = get_database_dir(database_dir)
        hotels_csv_path = str(db_dir / "hotels" / "hotels.csv")
        attractions_csv_path = str(db_dir / "attractions" / "attractions.csv")
        restaurants_csv_path = str(db_dir / "restaurants" / "restaurants.csv")
        flights_csv_path = str(db_dir / "flights" / "flights.csv")
        trains_csv_path = str(db_dir / "trains" / "trains.csv")
        locations_coords_path = str(db_dir / "locations" / "locations_coords.csv")
    else:
        hotels_csv_path = HOTELS_CSV_PATH
        attractions_csv_path = ATTRACTIONS_CSV_PATH
        restaurants_csv_path = RESTAURANTS_CSV_PATH
        flights_csv_path = str(_DATABASE_DIR / "flights" / "flights.csv")
        trains_csv_path = str(_DATABASE_DIR / "trains" / "trains.csv")
        locations_coords_path = LOCATIONS_COORDS_CSV_PATH
    
    hotels_index = load_hotel_index(hotels_csv_path)
    attractions_index = load_attraction_index(attractions_csv_path)
    restaurants_index = load_restaurant_index(restaurants_csv_path)
    flights_index = load_flights_index(flights_csv_path)
    trains_index = load_trains_index(trains_csv_path)
    locations_index = load_locations_index(locations_coords_path)
    _augment_locations_with_transport_hubs(locations_index, database_dir, meta)

    # ==================== Structure and route atomic checks ====================
    res["valid_trip_duration"] = check_valid_days(daily_plans, meta)
    res["closed_loop_route_structure"] = check_route_closed_loop(daily_plans, meta)
    res["seamless_intercity_transfers"] = check_intercity_transportation_consistency(daily_plans, meta, database_dir)
    res["day_boundary_continuity"] = check_day_boundary_continuity(daily_plans, hotels_index)

    # ==================== Entity and transport evidence atomic checks ====================
    res["validated_accommodation"] = check_hotels_from_search(daily_plans, hotels_index)
    res["validated_attractions"] = check_attractions_from_search(daily_plans, attractions_index)
    res["validated_meals"] = check_meals_from_search(daily_plans, restaurants_index)
    res["validated_transportation"] = check_intercity_public_from_search(daily_plans, flights_index, trains_index)

    # ==================== Stay traceability and quality-support atomic checks ====================
    res["traceable_accommodation"] = check_accommodation_traceable(daily_plans)
    res["ends_with_accommodation"] = check_last_activity_is_hotel(daily_plans)
    res["essential_meal_coverage"] = check_meal_necessity(daily_plans, meta)
    res["essential_attraction_coverage"] = check_attraction_necessity(daily_plans, meta)

    # ==================== Time and local transfer atomic checks ====================
    res["no_time_overlaps"] = check_time_no_overlap(daily_plans)
    res["reasonable_transfer_time"] = check_transfer_time_reasonable(daily_plans, locations_index, database_dir)
    res["local_move_sanity"] = check_local_move_sanity(daily_plans)

    # ==================== Venue availability atomic checks ====================
    res["attraction_visit_within_opening_hours"] = check_attractions_in_opening_hours(daily_plans, attractions_index)
    res["dining_within_service_hours"] = check_meals_in_business_hours(daily_plans, restaurants_index)
    res["avoidance_of_closure_days"] = check_attractions_not_closed(daily_plans, attractions_index, meta)

    # ==================== Duration and buffer atomic checks ====================
    res["reasonable_duration_at_attractions"] = check_attractions_duration_reasonable(daily_plans, attractions_index)
    res["reasonable_meal_duration"] = check_meal_duration_reasonable(daily_plans)
    res["intercity_buffer_adequacy"] = check_intercity_buffer_adequacy(daily_plans)

    # ==================== Cost arithmetic atomic check ====================
    res["cost_calculation_correctness"] = check_budget_accuracy(plan, daily_plans, meta)

    return res

def calculate_feasibility(check_results: Dict[str, Tuple[bool, Optional[str]]]) -> Dict[str, Any]:
    """Aggregate existing atomic checks into the compact feasibility structure."""
    dimensions: Dict[str, Any] = {}
    subdimensions: Dict[str, Any] = {}
    feasibility_score = 0.0

    for dim_name, dim_config in FEASIBILITY_DIMENSIONS.items():
        subdim_details: Dict[str, Any] = {}
        dim_score_sum = 0.0
        subdim_items = dim_config["subdimensions"]

        for subdim_name, check_names in subdim_items.items():
            detail = _score_group(check_names, check_results)
            subdim_details[subdim_name] = detail
            subdimensions[subdim_name] = detail
            dim_score_sum += detail["score"]

        dim_score = dim_score_sum / len(subdim_items) if subdim_items else 0.0
        weighted_score = dim_score * dim_config["weight"]
        dimensions[dim_name] = {
            "weight": dim_config["weight"],
            "score": dim_score,
            "weighted_score": weighted_score,
            "subdimensions": subdim_details,
        }
        feasibility_score += weighted_score

    checks = {
        name: {
            "passed": bool(result[0]),
            "message": result[1],
        }
        for name, result in check_results.items()
    }
    expected_checks = list(
        dict.fromkeys(
            check_name
            for dim_config in FEASIBILITY_CHECK_DIMENSIONS.values()
            for check_name in dim_config["checks"]
        )
    )
    evaluated = [
        bool(check_results.get(check_name, (False, "Check not evaluated"))[0])
        for check_name in expected_checks
    ]
    strict_feasibility = 1.0 if evaluated and all(evaluated) else 0.0
    return {
        "score": feasibility_score,
        "strict_feasibility": strict_feasibility,
        "passed_checks": sum(1 for passed in evaluated if passed),
        "total_checks": len(evaluated),
        "dimensions": dimensions,
        "subdimensions": subdimensions,
        "checks": checks,
    }
