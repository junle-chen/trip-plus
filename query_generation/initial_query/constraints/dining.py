"""Restaurant hard constraint builders."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List

from query_generation.common import (
    ConstraintSpec,
    TAG_KEY_MAP,
    distance_to_nearby_attraction,
    extract_cuisine_label,
    safe_float,
    split_semicolon_field,
)
from query_generation.initial_query.constraints.evidence import (
    is_quality_attraction_row,
    restaurant_options,
    row_names,
    rows_with_float,
)

# Sample one dining hard constraint anchored to restaurants and nearby attractions.
def build_restaurant_constraint(db: Dict[str, List[Dict[str, str]]], rng: random.Random) -> ConstraintSpec:
    restaurants = db["restaurants"]
    if not restaurants:
        raise RuntimeError("No restaurants available")

    candidates: List[ConstraintSpec] = []
    # Keep "nearby" restaurant constraints anchored to real attractions only.
    # Some restaurant rows may reference hotel or other POI names in nearby_attraction_name.
    valid_attraction_names = {
        row["attraction_name"]
        for row in db.get("attractions", [])
        if is_quality_attraction_row(row)
    }
    by_attraction: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in restaurants:
        nearby_name = row.get("nearby_attraction_name")
        if nearby_name and nearby_name in valid_attraction_names:
            by_attraction[row["nearby_attraction_name"]].append(row)

    for attraction_name, rows in by_attraction.items():
        cheapest = min(rows, key=lambda x: safe_float(x["price_per_person"], 999999))
        highest = max(rows, key=lambda x: (safe_float(x["rating"]), -safe_float(x["price_per_person"], 999999)))
        closest = min(rows, key=distance_to_nearby_attraction)
        cheapest_price = safe_float(cheapest["price_per_person"], 999999)
        highest_rating = safe_float(highest["rating"])
        closest_distance = distance_to_nearby_attraction(closest)
        cheapest_matches = rows_with_float(rows, "price_per_person", cheapest_price, 999999)
        highest_matches = rows_with_float(rows, "rating", highest_rating)
        closest_matches = [row for row in rows if distance_to_nearby_attraction(row) == closest_distance]
        candidates.extend(
            [
                ConstraintSpec(
                    "restaurant_cheapest_nearby_attraction",
                    {
                        "constraint_context": f"Arrange one meal at the cheapest per-person restaurant near '{attraction_name}'",
                        "constraint_type": "superlative_cheapest_nearby_attraction",
                        "attraction_name": attraction_name,
                        "restaurant_name": cheapest["restaurant_name"],
                        "price_per_person": safe_float(cheapest["price_per_person"]),
                        "acceptable_restaurant_names": row_names(cheapest_matches, "restaurant_name"),
                        "acceptable_restaurant_options": restaurant_options(cheapest_matches),
                    },
                    f"For one meal, I want the cheapest per-person restaurant near {attraction_name}.",
                    f"Arrange one meal at the cheapest per-person restaurant near {attraction_name}.",
                ),
                ConstraintSpec(
                    "restaurant_highest_rated",
                    {
                        "constraint_context": f"Arrange one meal at the highest-rated restaurant near '{attraction_name}'",
                        "constraint_type": "superlative_highest_rated_restaurant",
                        "attraction_name": attraction_name,
                        "restaurant_name": highest["restaurant_name"],
                        "restaurant_rating": safe_float(highest["rating"]),
                        "price_per_person": safe_float(highest["price_per_person"]),
                        "acceptable_restaurant_names": row_names(highest_matches, "restaurant_name"),
                        "acceptable_restaurant_options": restaurant_options(highest_matches),
                    },
                    f"For one meal, I want the highest-rated restaurant near {attraction_name}.",
                    f"Arrange one meal at the highest-rated restaurant near {attraction_name}.",
                ),
                ConstraintSpec(
                    "restaurant_closest_to_attraction",
                    {
                        "constraint_context": f"Arrange one meal at the restaurant closest to '{attraction_name}'",
                        "constraint_type": "superlative_closest_to_attraction",
                        "attraction_name": attraction_name,
                        "restaurant_name": closest["restaurant_name"],
                        "distance_meters": int(distance_to_nearby_attraction(closest)),
                        "price_per_person": safe_float(closest["price_per_person"]),
                        "acceptable_restaurant_names": row_names(closest_matches, "restaurant_name"),
                        "acceptable_restaurant_options": restaurant_options(closest_matches),
                    },
                    f"For one meal, choose the restaurant closest to {attraction_name}.",
                    f"Arrange one meal at the restaurant closest to {attraction_name}.",
                ),
            ]
        )

    tag_rows = [
        (row, tag)
        for row in restaurants
        for tag in split_semicolon_field(row.get("tags", ""))
        if tag in TAG_KEY_MAP
        and row.get("nearby_attraction_name") in valid_attraction_names
    ]
    if tag_rows:
        row, tag_label = rng.choice(tag_rows)
        tag_matches = [
            restaurant
            for restaurant in restaurants
            if restaurant.get("nearby_attraction_name") == row["nearby_attraction_name"]
            and tag_label in split_semicolon_field(restaurant.get("tags", ""))
        ]
        candidates.append(
            ConstraintSpec(
                "restaurant_specific_tag_nearby",
                {
                    "constraint_context": f"Arrange one meal near '{row['nearby_attraction_name']}' at a restaurant with {tag_label}",
                    "constraint_type": "superlative_specific_tag_nearby",
                    "attraction_name": row["nearby_attraction_name"],
                    "required_tag": TAG_KEY_MAP[tag_label],
                    "required_tag_label": tag_label,
                    "restaurant_name": row["restaurant_name"],
                    "price_per_person": safe_float(row["price_per_person"]),
                    "restaurant_rating": safe_float(row["rating"]),
                    "acceptable_restaurant_names": row_names(tag_matches, "restaurant_name"),
                    "acceptable_restaurant_options": restaurant_options(tag_matches),
                },
                f"For one meal near {row['nearby_attraction_name']}, I want a restaurant with {tag_label}.",
                f"Arrange one meal near {row['nearby_attraction_name']} at a restaurant with {tag_label}.",
            )
        )

    cuisine_rows = [
        (row, extract_cuisine_label(row.get("cuisine", "")))
        for row in restaurants
        if row.get("nearby_attraction_name") in valid_attraction_names
    ]
    cuisine_rows = [(row, cuisine) for row, cuisine in cuisine_rows if cuisine]
    if cuisine_rows:
        row, cuisine_label = rng.choice(cuisine_rows)
        cuisine_matches = [
            restaurant
            for restaurant, cuisine in cuisine_rows
            if restaurant.get("nearby_attraction_name") == row["nearby_attraction_name"]
            and cuisine == cuisine_label
        ]
        candidates.append(
            ConstraintSpec(
                "restaurant_specific_cuisine_nearby",
                {
                    "constraint_context": f"Arrange one meal near '{row['nearby_attraction_name']}' with cuisine type {cuisine_label}",
                    "constraint_type": "superlative_specific_cuisine_nearby",
                    "attraction_name": row["nearby_attraction_name"],
                    "cuisine_type": cuisine_label,
                    "restaurant_name": row["restaurant_name"],
                    "price_per_person": safe_float(row["price_per_person"]),
                    "restaurant_rating": safe_float(row["rating"]),
                    "acceptable_restaurant_names": row_names(cuisine_matches, "restaurant_name"),
                    "acceptable_restaurant_options": restaurant_options(cuisine_matches),
                },
                f"For one meal near {row['nearby_attraction_name']}, I prefer {cuisine_label} cuisine.",
                f"Arrange one meal near {row['nearby_attraction_name']} with cuisine type {cuisine_label}.",
            )
        )

    named = max(restaurants, key=lambda x: (safe_float(x["rating"]), -safe_float(x["price_per_person"], 999999)))
    candidates.append(
        ConstraintSpec(
            "restaurant_must_eat_named",
            {
                "constraint_context": f"Arrange at least one meal at '{named['restaurant_name']}'",
                "constraint_type": "superlative_must_eat_named",
                "restaurant_name": named["restaurant_name"],
                "restaurant_rating": safe_float(named["rating"]),
            },
            f"I want to lock in {named['restaurant_name']} for at least one meal.",
            f"Arrange at least one meal at {named['restaurant_name']}.",
        )
    )
    return rng.choice(candidates)
