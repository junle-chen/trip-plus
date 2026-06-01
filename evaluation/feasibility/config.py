"""Feasibility dimensions and check grouping used by deterministic evaluation."""

FEASIBILITY_CHECK_DIMENSIONS = {
    "structure_completeness": {
        "weight": 1 / 3,
        "checks": [
            "valid_trip_duration",
            "closed_loop_route_structure",
            "seamless_intercity_transfers",
            "day_boundary_continuity",
            "traceable_accommodation",
            "ends_with_accommodation",
            "essential_meal_coverage",
            "essential_attraction_coverage",
        ],
    },
    "evidence_validity": {
        "weight": 1 / 3,
        "checks": [
            "validated_accommodation",
            "validated_attractions",
            "validated_meals",
            "validated_transportation",
            "local_move_sanity",
        ],
    },
    "execution_operability": {
        "weight": 1 / 3,
        "checks": [
            "no_time_overlaps",
            "reasonable_transfer_time",
            "attraction_visit_within_opening_hours",
            "dining_within_service_hours",
            "avoidance_of_closure_days",
            "reasonable_duration_at_attractions",
            "reasonable_meal_duration",
            "intercity_buffer_adequacy",
            "cost_calculation_correctness",
        ],
    },
}

FEASIBILITY_DIMENSIONS = {
    "structure_completeness": {
        "weight": 1 / 3,
        "subdimensions": {
            "valid_trip_duration": [
                "valid_trip_duration",
            ],
            "route_and_stay_continuity": [
                "closed_loop_route_structure",
                "seamless_intercity_transfers",
                "day_boundary_continuity",
                "traceable_accommodation",
                "ends_with_accommodation",
            ],
            "daily_content_coverage": [
                "essential_meal_coverage",
                "essential_attraction_coverage",
            ],
        },
    },
    "evidence_validity": {
        "weight": 1 / 3,
        "subdimensions": {
            "poi_grounding_valid": [
                "validated_accommodation",
                "validated_attractions",
                "validated_meals",
            ],
            "transport_grounding_valid": [
                "validated_transportation",
                "local_move_sanity",
            ],
        },
    },
    "execution_operability": {
        "weight": 1 / 3,
        "subdimensions": {
            "time_and_transfer_feasible": [
                "no_time_overlaps",
                "reasonable_transfer_time",
            ],
            "venue_and_duration_feasible": [
                "attraction_visit_within_opening_hours",
                "dining_within_service_hours",
                "avoidance_of_closure_days",
                "reasonable_duration_at_attractions",
                "reasonable_meal_duration",
            ],
            "intercity_buffer_feasible": [
                "intercity_buffer_adequacy",
            ],
            "budget_arithmetic_feasible": [
                "cost_calculation_correctness",
            ],
        },
    },
}

ALL_FEASIBILITY_CHECKS = tuple(
    dict.fromkeys(
        check_name
        for dim_config in FEASIBILITY_CHECK_DIMENSIONS.values()
        for check_name in dim_config["checks"]
    )
)
