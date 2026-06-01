"""Flight and train hard-constraint checks."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from .common import (
    _extract_flights_from_plan,
    _extract_trains_from_plan,
    _transport_direction_satisfied,
)


def _eval_flight_constraint(
    constraint_key: str, constraint_data: Dict, plan: Dict, meta: Dict
) -> Tuple[bool, Optional[str]]:
    """
    Evaluate flight-related constraints (unified logic)

    All flight constraints check if required flight numbers are in the plan.
    This unified approach:
    - Extracts flight list only once (performance optimization)
    - Uses set lookup for O(1) search instead of O(n)
    - Eliminates code duplication across multiple similar functions

    Supported constraints:
    - flight_seat_class: Check both outbound and inbound flights
    - flight_cheapest_airline_direct: Check outbound flight only
    - flight_cheapest_direct: Check outbound flight only
    - flight_earliest_departure_direct: Check outbound flight only
    - flight_cheapest_manufacturer_direct: Check outbound flight only
    - flight_shortest_duration_direct: Check outbound or inbound flight (NEW)
    - flight_earliest_airline_direct: Check outbound flight only (NEW)
    - flight_departure_time_range: Check outbound flight only (NEW)
    - flight_arrival_time_range: Check inbound flight only (NEW)
    """
    # Extract flights from plan once (performance optimization)
    flights = _extract_flights_from_plan(plan)
    required_flights = [
        (
            "outbound",
            "acceptable_outbound_flight_nos",
            "outbound_flight_no",
            "acceptable_outbound_flight_options",
        ),
        (
            "inbound",
            "acceptable_inbound_flight_nos",
            "inbound_flight_no",
            "acceptable_inbound_flight_options",
        ),
    ]

    for direction, numbers_key, singular_key, options_key in required_flights:
        ok, message = _transport_direction_satisfied(
            flights,
            constraint_data,
            direction=direction,
            transport_label="flight",
            number_key="flight_no",
            acceptable_numbers_key=numbers_key,
            singular_number_key=singular_key,
            acceptable_options_key=options_key,
        )
        if not ok:
            return (False, message)

    return (True, None)


# ============================================================================
# Train Constraints
# ============================================================================
def _eval_train_constraint(
    constraint_key: str, constraint_data: Dict, plan: Dict, meta: Dict
) -> Tuple[bool, Optional[str]]:
    """
    Evaluate train-related constraints (unified logic)

    All train constraints check if required train numbers are in the plan.
    This unified approach:
    - Extracts train list only once (performance optimization)
    - Uses set lookup for O(1) search instead of O(n)
    - Eliminates code duplication across multiple similar functions

    Supported constraints:
    - train_seat_class: Check both outbound and inbound trains
    - train_shortest_duration_direct: Check outbound or inbound train (NEW)
    - train_cheapest_direct: Check outbound or inbound train (NEW)
    - train_earliest_departure_direct: Check outbound train only (NEW)
    - train_latest_arrival_direct: Check inbound train only (NEW)
    - train_cheapest_train_type: Check outbound train only (NEW)
    - train_departure_time_range: Check outbound train only (NEW)
    """
    # Extract trains from plan once (performance optimization)
    trains = _extract_trains_from_plan(plan)
    required_trains = [
        (
            "outbound",
            "acceptable_outbound_train_nos",
            "outbound_train_no",
            "acceptable_outbound_train_options",
        ),
        (
            "inbound",
            "acceptable_inbound_train_nos",
            "inbound_train_no",
            "acceptable_inbound_train_options",
        ),
    ]

    for direction, numbers_key, singular_key, options_key in required_trains:
        ok, message = _transport_direction_satisfied(
            trains,
            constraint_data,
            direction=direction,
            transport_label="train",
            number_key="train_no",
            acceptable_numbers_key=numbers_key,
            singular_number_key=singular_key,
            acceptable_options_key=options_key,
        )
        if not ok:
            return (False, message)

    return (True, None)


# ============================================================================
# Hotel Constraints
# ============================================================================
