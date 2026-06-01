"""Hotel, restaurant, and attraction hard-constraint checks."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from ..utils import normalize_entity_name
from .common import (
    _extract_attractions_from_plan,
    _extract_hotels_from_plan,
    _extract_restaurants_from_plan,
    _normalized_name_set,
    _resolve_acceptables,
    _string_list,
)


def _eval_hotel_constraint(
    constraint_key: str, constraint_data: Dict, plan: Dict, meta: Dict
) -> Tuple[bool, Optional[str]]:
    """
    Evaluate hotel-related constraints (unified logic)

    All hotel constraints check if the required hotel name is in the plan.
    This unified approach:
    - Extracts hotel list only once (performance optimization)
    - Uses set lookup for O(1) search instead of O(n)
    - Eliminates code duplication across multiple similar functions

    Supported constraints:
    - hotel_cheapest_brand: Check if cheapest hotel of specified brand is used
    - hotel_highest_rated: Check if highest rated hotel is used
    - hotel_cheapest_star: Check if cheapest hotel of specified star rating is used
    - hotel_newest_decoration: Check if hotel with newest decoration is used (NEW)
    - hotel_brand_highest_rated: Check if highest rated hotel within brand is used (NEW)
    - hotel_star_highest_rated: Check if highest rated hotel within star level is used (NEW)
    - hotel_price_range: Check if hotel within price range is used (NEW)
    - hotel_star_service_required: Check if hotel with specified star and service is used (NEW)
    """
    # Extract hotels from plan once (performance optimization)
    hotels = _extract_hotels_from_plan(plan)
    hotel_names = {
        normalize_entity_name(hotel["name"]) for hotel in hotels if hotel.get("name")
    }

    acceptable_hotel_names = _resolve_acceptables(
        constraint_data, "acceptable_hotel_names", "hotel_name"
    )
    if not acceptable_hotel_names:
        return (False, "No hotel name specified in constraint data")

    acceptable_hotel_names_normalized = _normalized_name_set(acceptable_hotel_names)
    if not hotel_names.intersection(acceptable_hotel_names_normalized):
        # Generate appropriate error message based on constraint type
        if constraint_key == "hotel_cheapest_brand":
            brand = constraint_data.get("brand", "specified")
            return (
                False,
                f"Required {brand} brand hotel not found in acceptable set: {acceptable_hotel_names}",
            )
        elif constraint_key == "hotel_highest_rated":
            return (
                False,
                f"Required highest rated hotel not found in acceptable set: {acceptable_hotel_names}",
            )
        elif constraint_key == "hotel_cheapest_star":
            star = constraint_data.get("hotel_star", "specified")
            return (
                False,
                f"Required {star}-star hotel not found in acceptable set: {acceptable_hotel_names}",
            )
        elif constraint_key == "hotel_newest_decoration":
            return (
                False,
                f"Required hotel with newest decoration not found in acceptable set: {acceptable_hotel_names}",
            )
        elif constraint_key == "hotel_brand_highest_rated":
            brand = constraint_data.get("brand", "specified")
            return (
                False,
                f"Required highest rated {brand} hotel not found in acceptable set: {acceptable_hotel_names}",
            )
        elif constraint_key == "hotel_star_highest_rated":
            star = constraint_data.get("hotel_star", "specified")
            return (
                False,
                f"Required highest rated {star}-star hotel not found in acceptable set: {acceptable_hotel_names}",
            )
        elif constraint_key == "hotel_price_range":
            price_range = constraint_data.get("price_range", "specified")
            return (
                False,
                f"Required hotel in price range {price_range} not found in acceptable set: {acceptable_hotel_names}",
            )
        elif constraint_key == "hotel_star_service_required":
            star = constraint_data.get("hotel_star", "specified")
            service = constraint_data.get("required_service_label", "specified service")
            return (
                False,
                f"Required {star}-star hotel with {service} not found in acceptable set: {acceptable_hotel_names}",
            )
        else:
            return (
                False,
                f"Required hotel not found in acceptable set: {acceptable_hotel_names}",
            )

    return (True, None)


# ============================================================================
# Restaurant Constraints
# ============================================================================
def _eval_restaurant_constraint(
    constraint_key: str, constraint_data: Dict, plan: Dict, meta: Dict
) -> Tuple[bool, Optional[str]]:
    """
    Evaluate restaurant-related constraints (unified logic)

    All restaurant constraints check if the required restaurant name is in the plan.
    This unified approach:
    - Extracts restaurant list only once (performance optimization)
    - Uses set lookup for O(1) search instead of O(n)
    - Eliminates code duplication across multiple similar functions

    Supported constraints:
    - restaurant_cheapest_nearby_attraction: Check if cheapest restaurant near attraction is used
    - restaurant_highest_rated: Check if highest rated restaurant near attraction is used
    - restaurant_must_eat_named: Check if must-eat named restaurant is used
    - restaurant_closest_to_attraction: Check if closest restaurant to attraction is used
    - restaurant_specific_cuisine_nearby: Check if specific cuisine restaurant near attraction is used (NEW)
    - restaurant_specific_tag_nearby: Check if restaurant with specific tag near attraction is used (NEW)
    """
    # Extract restaurants from plan once (performance optimization)
    restaurants = _extract_restaurants_from_plan(plan)
    restaurant_names = {
        normalize_entity_name(restaurant["name"])
        for restaurant in restaurants
        if restaurant.get("name")
    }

    acceptable_restaurant_names = _resolve_acceptables(
        constraint_data, "acceptable_restaurant_names", "restaurant_name"
    )
    if not acceptable_restaurant_names:
        return (False, "No restaurant name specified in constraint data")

    acceptable_restaurant_names_normalized = _normalized_name_set(
        acceptable_restaurant_names
    )
    if not restaurant_names.intersection(acceptable_restaurant_names_normalized):
        # Generate appropriate error message based on constraint type
        if constraint_key == "restaurant_cheapest_nearby_attraction":
            attraction = constraint_data.get("attraction_name", "specified attraction")
            return (
                False,
                f"Required restaurant near {attraction} not found in acceptable set: {acceptable_restaurant_names}",
            )
        elif constraint_key == "restaurant_highest_rated":
            attraction = constraint_data.get("attraction_name", "specified attraction")
            return (
                False,
                f"Required highly rated restaurant near {attraction} not found in acceptable set: {acceptable_restaurant_names}",
            )
        elif constraint_key == "restaurant_must_eat_named":
            return (
                False,
                f"Required must-eat restaurant not found in acceptable set: {acceptable_restaurant_names}",
            )
        elif constraint_key == "restaurant_closest_to_attraction":
            attraction = constraint_data.get("attraction_name", "specified attraction")
            return (
                False,
                f"Required closest restaurant to {attraction} not found in acceptable set: {acceptable_restaurant_names}",
            )
        elif constraint_key == "restaurant_specific_cuisine_nearby":
            attraction = constraint_data.get("attraction_name", "specified attraction")
            cuisine = constraint_data.get("cuisine_type", "specified cuisine")
            return (
                False,
                f"Required {cuisine} restaurant near {attraction} not found in acceptable set: {acceptable_restaurant_names}",
            )
        elif constraint_key == "restaurant_specific_tag_nearby":
            attraction = constraint_data.get("attraction_name", "specified attraction")
            tag = constraint_data.get("required_tag_label", "specified tag")
            return (
                False,
                f"Required restaurant with {tag} near {attraction} not found in acceptable set: {acceptable_restaurant_names}",
            )
        else:
            return (
                False,
                f"Required restaurant not found in acceptable set: {acceptable_restaurant_names}",
            )

    return (True, None)


# ============================================================================
# Attraction Constraints
# ============================================================================
def _eval_attraction_constraint(
    constraint_key: str, constraint_data: Dict, plan: Dict, meta: Dict
) -> Tuple[bool, Optional[str]]:
    """
    Evaluate attraction-related constraints (unified logic)

    All attraction constraints check if required attraction names are in the plan.
    This unified approach:
    - Extracts attraction list only once (performance optimization)
    - Uses set lookup for O(1) search instead of O(n)
    - Eliminates code duplication across multiple similar functions
    - All constraints use 'attraction_names' list format (even single-item constraints)

    Supported constraints:
    - attraction_must_visit_named: Check if all must-visit named attractions are included
    - attraction_all_of_type: Check if all attractions of specified type are included
    - attraction_top_rated_must_visit: Check if top 3 rated attractions are included (NEW)
    - attraction_all_free_attractions: Check if all free attractions are included (NEW)
    - attraction_type_highest_rated: Check if highest rated attraction of specific type is included (NEW)

    Note: All constraints now return 'attraction_names' as a list, even if only one attraction.
    """
    # Extract attractions from plan once (performance optimization)
    attractions = _extract_attractions_from_plan(plan)
    attraction_names = {
        normalize_entity_name(attr["name"]) for attr in attractions if attr.get("name")
    }

    banned_attractions = _string_list(constraint_data.get("banned_attraction_names"))
    if banned_attractions:
        violated = sorted(
            original
            for original in banned_attractions
            if normalize_entity_name(original) in attraction_names
        )
        if violated:
            return (
                False,
                f"Plan includes banned high-queue attractions: {', '.join(violated)}",
            )
        return (True, None)

    acceptable_attractions = _string_list(
        constraint_data.get("acceptable_attraction_names")
    )
    if acceptable_attractions:
        acceptable_attractions_normalized = _normalized_name_set(acceptable_attractions)
        if not attraction_names.intersection(acceptable_attractions_normalized):
            attraction_type = (
                constraint_data.get("attraction_type") or "specified/popular"
            )
            if constraint_key == "attraction_require_popular_hotspot":
                return (
                    False,
                    f"Required popular/check-in attraction not found in acceptable set: {acceptable_attractions}",
                )
            return (
                False,
                f"Required acceptable attraction set not found for {attraction_type}: {acceptable_attractions}",
            )
        return (True, None)

    required_attractions = constraint_data.get("attraction_names", [])
    if not required_attractions:
        return (False, "No attraction names specified in constraint data")

    # Check if all required attractions are in the plan
    missing_attractions = []
    for required_attraction in required_attractions:
        if normalize_entity_name(required_attraction) not in attraction_names:
            missing_attractions.append(required_attraction)

    if missing_attractions:
        # Generate appropriate error message based on constraint type
        if constraint_key == "attraction_must_visit_named":
            return (
                False,
                f"Missing required attractions: {', '.join(missing_attractions)}",
            )
        elif constraint_key == "attraction_all_of_type":
            attraction_type = constraint_data.get("attraction_type", "specified")
            return (
                False,
                f"Missing {attraction_type} type attractions: {', '.join(missing_attractions)}",
            )
        elif constraint_key == "attraction_top_rated_must_visit":
            return (
                False,
                f"Missing top rated attractions: {', '.join(missing_attractions)}",
            )
        elif constraint_key == "attraction_all_free_attractions":
            return (
                False,
                f"Missing free attractions: {', '.join(missing_attractions)}",
            )
        elif constraint_key == "attraction_type_highest_rated":
            attraction_type = constraint_data.get("attraction_type", "specified")
            return (
                False,
                f"Required highest rated {attraction_type} attraction not found: {', '.join(missing_attractions)}",
            )
        else:
            return (False, f"Missing attractions: {', '.join(missing_attractions)}")

    return (True, None)


# ============================================================================
# Helper Functions - Extract information from plan
# ============================================================================
