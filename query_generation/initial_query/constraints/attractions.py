"""Attraction hard constraint builders."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List

from query_generation.common import ConstraintSpec, safe_float, split_semicolon_field
from query_generation.initial_query.constraints.evidence import (
    attraction_options,
    friendly_attraction_type,
    is_non_ticket_attraction,
    is_quality_attraction_row,
    row_names,
    rows_with_float,
    unique_preserve_order,
)

# Sample one attraction hard constraint from quality attraction evidence.
def build_attraction_constraint(db: Dict[str, List[Dict[str, str]]], rng: random.Random) -> ConstraintSpec:
    quality_attractions = [row for row in db["attractions"] if is_quality_attraction_row(row)]
    attractions = [row for row in quality_attractions if not is_non_ticket_attraction(row)]
    if not attractions:
        attractions = quality_attractions or db["attractions"]
    if not attractions:
        raise RuntimeError("No attractions available")
    candidates: List[ConstraintSpec] = []
    by_type: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in attractions:
        if row.get("attraction_type"):
            by_type[row["attraction_type"]].append(row)

    popular_rows = [
        row
        for row in attractions
        if {"classic_hotspot", "viral", "photo_spot", "night_hotspot"}
        & set(split_semicolon_field(row.get("popularity_tags", "")))
    ]
    if popular_rows:
        ordered = sorted(
            popular_rows,
            key=lambda x: (
                str(x.get("crowd_risk", "")) != "high",
                -safe_float(x.get("rating")),
                safe_float(x.get("ticket_price"), 999999),
                x["attraction_name"],
            ),
        )
        acceptable = unique_preserve_order([row["attraction_name"] for row in ordered])
        candidates.append(
            ConstraintSpec(
                "attraction_require_popular_hotspot",
                {
                    "constraint_context": "Include at least one attraction tagged as popular, viral, photogenic, or a night hotspot in the database",
                    "constraint_type": "popularity_tag_required",
                    "required_tags": ["classic_hotspot", "viral", "photo_spot", "night_hotspot"],
                    "acceptable_attraction_names": acceptable,
                    "acceptable_attraction_options": attraction_options(ordered[:20]),
                },
                "I still want to keep one representative popular check-in spot instead of making everything obscure.",
                "Include at least one representative popular check-in attraction.",
            )
        )

    high_queue_rows = [row for row in attractions if str(row.get("queue_risk", "")).strip() == "high"]
    if high_queue_rows:
        ordered = sorted(
            high_queue_rows,
            key=lambda x: (-safe_float(x.get("rating")), x["attraction_name"]),
        )
        banned = unique_preserve_order([row["attraction_name"] for row in ordered])
        candidates.append(
            ConstraintSpec(
                "attraction_avoid_high_queue",
                {
                    "constraint_context": "Do not include attractions tagged as high queue risk in the database",
                    "constraint_type": "avoid_high_queue_attractions",
                    "banned_attraction_names": banned,
                    "banned_attraction_options": attraction_options(ordered[:20]),
                },
                f"I do not want to spend time in long lines, so avoid high-queue-risk places like {banned[0]}.",
                f"Avoid high-queue-risk attractions such as {banned[0]}.",
            )
        )

    top_overall = sorted(attractions, key=lambda x: (-safe_float(x["rating"]), safe_float(x["ticket_price"], 999999), x["attraction_name"]))
    if len(top_overall) >= 2:
        picked = top_overall[:2]
        candidates.append(
            ConstraintSpec(
                "attraction_must_visit_named",
                {
                    "constraint_context": f"The itinerary must visit '{picked[0]['attraction_name']}' and '{picked[1]['attraction_name']}'",
                    "constraint_type": "superlative_must_visit_named",
                    "attraction_names": [picked[0]["attraction_name"], picked[1]["attraction_name"]],
                    "attraction_ratings": [safe_float(picked[0]["rating"]), safe_float(picked[1]["rating"])],
                },
                f"There are two places I definitely want to visit: {picked[0]['attraction_name']} and {picked[1]['attraction_name']}.",
                f"The itinerary must visit {picked[0]['attraction_name']} and {picked[1]['attraction_name']}.",
            )
        )
    if len(top_overall) >= 3:
        picked = top_overall[:3]
        candidates.append(
            ConstraintSpec(
                "attraction_top_rated_must_visit",
                {
                    "constraint_context": "The itinerary must visit the three highest-rated attractions returned by the attraction recommendation tool",
                    "constraint_type": "superlative_top_rated_must_visit",
                    "attraction_names": [row["attraction_name"] for row in picked],
                    "attraction_ratings": [safe_float(row["rating"]) for row in picked],
                },
                "For a first visit, please directly arrange the three highest-rated recommended attractions.",
                "The itinerary must visit the three highest-rated attractions returned by the attraction recommendation tool.",
            )
        )

    free_rows = [row for row in attractions if safe_float(row.get("ticket_price")) == 0.0]
    if 2 <= len(free_rows) <= 5:
        ordered = sorted(free_rows, key=lambda x: (-safe_float(x["rating"]), x["attraction_name"]))
        candidates.append(
            ConstraintSpec(
                "attraction_all_free_attractions",
                {
                    "constraint_context": "The itinerary must include all free attractions returned by the attraction recommendation tool",
                    "constraint_type": "superlative_all_free_attractions",
                    "attraction_names": row_names(ordered, "attraction_name"),
                    "attraction_ratings": [safe_float(row["rating"]) for row in ordered],
                    "ticket_prices": [safe_float(row["ticket_price"]) for row in ordered],
                },
                "If the recommendations include free attractions, I want to include all of them if feasible.",
                "The itinerary must include all free attractions returned by the attraction recommendation tool.",
            )
        )

    type_options = [
        atype
        for atype, rows in by_type.items()
        if 2 <= len(rows) <= 5 and friendly_attraction_type(atype)
    ]
    if type_options:
        picked_type = rng.choice(sorted(type_options))
        picked_type_label = friendly_attraction_type(picked_type) or picked_type
        ordered = sorted(by_type[picked_type], key=lambda x: (-safe_float(x["rating"]), x["attraction_name"]))
        candidates.append(
            ConstraintSpec(
                "attraction_all_of_type",
                {
                    "constraint_context": f"The itinerary must include all recommended attractions of type '{picked_type_label}'",
                    "constraint_type": "superlative_all_of_type",
                    "attraction_type": picked_type,
                    "attraction_type_label": picked_type_label,
                    "attraction_names": row_names(ordered, "attraction_name"),
                    "attraction_ratings": [safe_float(row["rating"]) for row in ordered],
                },
                f"I am interested in {picked_type_label} places; if several are recommended, include all of them if feasible.",
                f"The itinerary must include all recommended attractions of type {picked_type_label}.",
            )
        )

    if by_type:
        rated_type_options = [atype for atype in by_type if friendly_attraction_type(atype)]
        picked_type = rng.choice(sorted(rated_type_options or by_type))
        picked_type_label = friendly_attraction_type(picked_type) or picked_type
        best = max(by_type[picked_type], key=lambda x: (safe_float(x["rating"]), -safe_float(x["ticket_price"], 999999)))
        best_score = safe_float(best["rating"])
        best_matches = rows_with_float(by_type[picked_type], "rating", best_score)
        candidates.append(
            ConstraintSpec(
                "attraction_type_highest_rated",
                {
                    "constraint_context": f"The itinerary must include the highest-rated attraction of type '{picked_type_label}'",
                    "constraint_type": "superlative_type_highest_rated",
                    "attraction_type": picked_type,
                    "attraction_type_label": picked_type_label,
                    "attraction_names": [best["attraction_name"]],
                    "acceptable_attraction_names": row_names(best_matches, "attraction_name"),
                    "acceptable_attraction_options": attraction_options(best_matches),
                    "attraction_ratings": [safe_float(best["rating"])],
                    "ticket_prices": [safe_float(best["ticket_price"])],
                },
                f"I am interested in {picked_type_label} places, and I want the highest-rated one of that type.",
                f"The itinerary must include the highest-rated attraction of type {picked_type_label}.",
            )
        )

    if not candidates:
        raise RuntimeError("Unable to sample attraction constraint")
    return rng.choice(candidates)
