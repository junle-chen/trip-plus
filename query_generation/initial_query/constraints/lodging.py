"""Hotel hard constraint builders."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List

from query_generation.common import (
    ConstraintSpec,
    SERVICE_KEY_MAP,
    safe_float,
    safe_int,
    split_semicolon_field,
)
from query_generation.initial_query.constraints.evidence import (
    hotel_options,
    row_names,
    rows_with_float,
    rows_with_label,
)

# Sample one lodging hard constraint from hotel evidence.
def build_hotel_constraint(db: Dict[str, List[Dict[str, str]]], rng: random.Random) -> ConstraintSpec:
    hotels = db["hotels"]
    candidates: List[ConstraintSpec] = []

    if hotels:
        highest = max(hotels, key=lambda x: (safe_float(x["score"]), -safe_float(x["price"], 999999)))
        highest_score = safe_float(highest["score"])
        highest_hotels = rows_with_float(hotels, "score", highest_score)
        candidates.append(
            ConstraintSpec(
                "hotel_highest_rated",
                {
                    "constraint_context": "Choose the highest-rated hotel in the city",
                    "constraint_type": "superlative_highest_rated",
                    "hotel_name": highest["name"],
                    "hotel_score": safe_float(highest["score"]),
                    "hotel_price": safe_float(highest["price"]),
                    "acceptable_hotel_names": row_names(highest_hotels, "name"),
                    "acceptable_hotel_options": hotel_options(highest_hotels),
                },
                "For lodging, I want the highest-rated hotel in the city.",
                "Choose the highest-rated hotel in the city.",
            )
        )

    by_brand: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    by_star: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for hotel in hotels:
        if hotel.get("brand"):
            by_brand[hotel["brand"]].append(hotel)
        star = safe_int(hotel.get("hotel_star"), -1)
        if star >= 0:
            by_star[star].append(hotel)

    if by_brand:
        brand = rng.choice(sorted(by_brand))
        picked = min(by_brand[brand], key=lambda x: safe_float(x["price"], 999999))
        picked_price = safe_float(picked["price"], 999999)
        cheapest_brand_hotels = rows_with_float(by_brand[brand], "price", picked_price, 999999)
        candidates.append(
            ConstraintSpec(
                "hotel_cheapest_brand",
                {
                    "constraint_context": f"Choose the cheapest hotel under the {brand} brand",
                    "constraint_type": "superlative_cheapest_brand",
                    "brand": brand,
                    "hotel_name": picked["name"],
                    "hotel_price": safe_float(picked["price"]),
                    "acceptable_hotel_names": row_names(cheapest_brand_hotels, "name"),
                    "acceptable_hotel_options": hotel_options(cheapest_brand_hotels),
                },
                f"I prefer the {brand} brand, and I want the cheapest hotel within that brand.",
                f"Choose the cheapest hotel under the {brand} brand.",
            )
        )

    if by_star:
        star = rng.choice(sorted(by_star))
        cheapest = min(by_star[star], key=lambda x: safe_float(x["price"], 999999))
        top_rated = max(by_star[star], key=lambda x: (safe_float(x["score"]), -safe_float(x["price"], 999999)))
        cheapest_price = safe_float(cheapest["price"], 999999)
        top_rated_score = safe_float(top_rated["score"])
        cheapest_star_hotels = rows_with_float(by_star[star], "price", cheapest_price, 999999)
        top_rated_star_hotels = rows_with_float(by_star[star], "score", top_rated_score)
        candidates.extend(
            [
                ConstraintSpec(
                    "hotel_cheapest_star",
                    {
                        "constraint_context": f"Choose the cheapest {star}-star hotel",
                        "constraint_type": "superlative_cheapest_star",
                        "hotel_star": star,
                        "hotel_name": cheapest["name"],
                        "hotel_price": safe_float(cheapest["price"]),
                        "acceptable_hotel_names": row_names(cheapest_star_hotels, "name"),
                        "acceptable_hotel_options": hotel_options(cheapest_star_hotels),
                    },
                    f"I want a {star}-star hotel, and please choose the cheapest option within that star level.",
                    f"Choose the cheapest {star}-star hotel.",
                ),
                ConstraintSpec(
                    "hotel_star_highest_rated",
                    {
                        "constraint_context": f"Choose the highest-rated {star}-star hotel",
                        "constraint_type": "superlative_star_highest_rated",
                        "hotel_star": star,
                        "hotel_name": top_rated["name"],
                        "hotel_score": safe_float(top_rated["score"]),
                        "hotel_price": safe_float(top_rated["price"]),
                        "acceptable_hotel_names": row_names(top_rated_star_hotels, "name"),
                        "acceptable_hotel_options": hotel_options(top_rated_star_hotels),
                    },
                    f"I want a {star}-star hotel, and please choose the highest-rated option within that star level.",
                    f"Choose the highest-rated {star}-star hotel.",
                ),
            ]
        )
        service_hotels = [(hotel, service) for hotel in by_star[star] for service in split_semicolon_field(hotel.get("services", "")) if service in SERVICE_KEY_MAP]
        if service_hotels:
            hotel, service_label = rng.choice(service_hotels)
            service_matches = rows_with_label(by_star[star], "services", service_label)
            candidates.append(
                ConstraintSpec(
                    "hotel_star_service_required",
                    {
                        "constraint_context": f"Choose a {star}-star hotel that provides {service_label}",
                        "constraint_type": "superlative_star_service_required",
                        "hotel_star": star,
                        "required_service": SERVICE_KEY_MAP[service_label],
                        "required_service_label": service_label,
                        "hotel_name": hotel["name"],
                        "hotel_price": safe_float(hotel["price"]),
                        "acceptable_hotel_names": row_names(service_matches, "name"),
                        "acceptable_hotel_options": hotel_options(service_matches),
                    },
                    f"I want a {star}-star hotel that also provides {service_label}.",
                    f"Choose a {star}-star hotel that provides {service_label}",
                )
            )

    if hotels:
        hotels_with_decoration_year = [
            hotel
            for hotel in hotels
            if 1990 <= safe_int(hotel.get("decoration_time"), 0) <= 2035
        ]
        newest = (
            max(hotels_with_decoration_year, key=lambda x: (safe_int(x["decoration_time"]), safe_float(x["score"])))
            if hotels_with_decoration_year
            else None
        )
        year = safe_int(newest["decoration_time"]) if newest else 0
        priced_hotels = [hotel for hotel in hotels if safe_float(hotel.get("price"), 0) > 0]
        picked = rng.choice(priced_hotels) if priced_hotels else None
        price_range_constraint = None
        if picked is not None:
            price = safe_float(picked["price"])
            width = 15 if price < 200 else 20 if price < 400 else 30
            min_price = int(max(50, price - width))
            max_price = int(price + width)
            if min_price <= max_price:
                price_matches = [
                    hotel
                    for hotel in hotels
                    if min_price <= safe_float(hotel.get("price"), -1) <= max_price
                ]
                price_range_constraint = ConstraintSpec(
                    "hotel_price_range",
                    {
                        "constraint_context": f"The accommodation price must be between {min_price} and {max_price} RMB per night",
                        "constraint_type": "price_range_hotel",
                        "price_range": f"{min_price}-{max_price}",
                        "min_price": min_price,
                        "max_price": max_price,
                        "hotel_name": picked["name"],
                        "hotel_price": price,
                        "acceptable_hotel_names": row_names(price_matches, "name"),
                        "acceptable_hotel_options": hotel_options(price_matches),
                    },
                    f"I would like the hotel price to stay between {min_price} and {max_price} RMB per night.",
                    f"The accommodation price must be between {min_price} and {max_price} RMB per night.",
                )
        if newest is not None:
            newest_matches = [
                hotel
                for hotel in hotels_with_decoration_year
                if safe_int(hotel.get("decoration_time")) >= year
            ]
            candidates.append(
                ConstraintSpec(
                    "hotel_newest_decoration",
                    {
                        "constraint_context": f"Choose a hotel renovated in or after {year}",
                        "constraint_type": "superlative_newest_decoration",
                        "hotel_name": newest["name"],
                        "decoration_time": year,
                        "year_threshold": year,
                        "hotel_price": safe_float(newest["price"]),
                        "hotel_score": safe_float(newest["score"]),
                        "acceptable_hotel_names": row_names(newest_matches, "name"),
                        "acceptable_hotel_options": hotel_options(newest_matches),
                    },
                    f"I prefer a newer hotel, ideally renovated in or after {year}.",
                    f"Choose a hotel renovated in or after {year}.",
                ),
            )
        if price_range_constraint is not None:
            candidates.append(price_range_constraint)

    if not candidates:
        raise RuntimeError("Unable to sample hotel constraint")
    return rng.choice(candidates)
