"""Multi-turn fulfillment key groups."""

NON_ITINERARY_MUST_UPDATE_KEYS = {
    "clarification_or_ranked_options",
    "conflict_detection_or_clarification",
    "infeasibility_detection",
    "prior_request_conflict_detection",
    "priority_clarification",
    "prior_constraint_clarification",
}

EVALUATED_MUST_UPDATE_KEYS = {
    "initial_plan",
    *NON_ITINERARY_MUST_UPDATE_KEYS,
    "add_attraction",
    "party_update",
    "restaurant_requirement",
    "hotel_requirement",
    "budget_update",
    "late_start_request",
    "schedule_update",
    "duration_update",
    "dietary_update",
    "explain_unsolved",
    "profile_preference",
    "environment_aware_replanning",
    "resolved_priority",
    "resolved_pacing_limit",
    "final_integrated_plan",
    "route_preference",
    "transport_preference",
    "hotel_preference",
    "restaurant_preference",
    "resolved_restaurant_preference",
    "apply_relaxed_constraint",
}
