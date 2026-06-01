"""Structure-completeness checks for itinerary feasibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils import (
    extract_city_from_location,
    extract_from_to,
    get_day_accommodation_city,
    get_intercity_arrival_time,
    get_intercity_departure_time,
    iter_hotel_acts,
    normalize_city,
    slot_to_minutes,
)


def check_valid_days(daily_plans: List[Dict[str, Any]], meta: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Check if the number of days matches expected."""
    expected_days = int(meta.get("days") or 0)
    is_days_valid = len(daily_plans) == expected_days and expected_days > 0
    return is_days_valid, None if is_days_valid else f"Plan has {len(daily_plans)} days, expected {expected_days}"

def check_route_closed_loop(daily_plans: List[Dict[str, Any]], meta: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Check if first day starts from org and last day returns to org (for intercity days)."""
    org = normalize_city(meta.get("org"))
    start_from, start_to = extract_from_to(daily_plans[0].get("current_city", "")) if daily_plans else (None, None)
    end_from, end_to = extract_from_to(daily_plans[-1].get("current_city", "")) if daily_plans else (None, None)

    is_closed_loop = True
    reason = None
    if start_from and normalize_city(start_from) != org:
        is_closed_loop = False
        reason = f"First day departure should be from {org}"
    if is_closed_loop and end_to and normalize_city(end_to) != org:
        is_closed_loop = False
        reason = f"Last day destination should return to {org}"
    return is_closed_loop, reason

def check_intercity_transportation_consistency(daily_plans: List[Dict[str, Any]], meta: Dict[str, Any], database_dir: Optional[Path] = None) -> Tuple[bool, Optional[str]]:
    """
    Track location changes by time order, check intercity transportation completeness.
    
    Logic:
    1. Initial location = org
    2. Iterate through each day:
       - If current_city = "from A to B":
         * Check if A equals current location
         * Check if there's corresponding travel_intercity_public activity
         * Update current location = B
       - If current_city = "some city":
         * Check if this city equals current location
         * If not equal, indicates missing intercity transportation info
    """
    violations: List[str] = []
    
    # Initial location
    current_location = normalize_city(meta.get("org"))
    if not current_location:
        return False, "Missing org info, cannot track location"
    
    for day_idx, day in enumerate(daily_plans, start=1):
        current_city = day.get("current_city", "")
        from_city, to_city = extract_from_to(current_city)
        
        if from_city and to_city:
            # Case 1: current_city = "from A to B"
            from_city_norm = normalize_city(from_city)
            to_city_norm = normalize_city(to_city)
            
            # Check if from equals current location
            if from_city_norm != current_location:
                violations.append(
                    f"D{day_idx}: current_city shows 'from {from_city} to {to_city}', "
                    f"but from city ({from_city}) does not match current location ({current_location})"
                )
            
            # Check if there's corresponding intercity transportation activity
            intercity_acts = []
            for act in day.get("activities", []) or []:
                if act.get("type") == "travel_intercity_public":
                    intercity_acts.append(act)
            
            if not intercity_acts:
                violations.append(
                    f"D{day_idx}: current_city shows '{from_city}→{to_city}' but missing travel_intercity_public activity"
                )
            else:
                # Check if intercity transportation route matches
                matched = False
                route_city_pairs: List[tuple[str, str]] = []
                unresolved_routes: List[str] = []
                for act in intercity_acts:
                    details = act.get("details") or {}
                    act_from = (details.get("from") or "").strip()
                    act_to = (details.get("to") or "").strip()
                    
                    if not act_from or not act_to:
                        unresolved_bits = []
                        if not act_from:
                            unresolved_bits.append("from=<missing>")
                        if not act_to:
                            unresolved_bits.append("to=<missing>")
                        unresolved_routes.append(" / ".join(unresolved_bits))
                        continue
                    
                    # Extract city name from airport/station
                    act_from_city = extract_city_from_location(act_from, database_dir)
                    act_to_city = extract_city_from_location(act_to, database_dir)
                    
                    # Check if matches
                    if act_from_city and act_to_city:
                        act_from_city_norm = normalize_city(act_from_city)
                        act_to_city_norm = normalize_city(act_to_city)
                        route_city_pairs.append((act_from_city_norm, act_to_city_norm))
                    else:
                        unresolved_bits = []
                        if not act_from_city:
                            unresolved_bits.append(f"from={act_from}")
                        if not act_to_city:
                            unresolved_bits.append(f"to={act_to}")
                        unresolved_routes.append(" / ".join(unresolved_bits))
                    if (act_from_city and act_to_city and
                        act_from_city_norm == from_city_norm and
                        act_to_city_norm == to_city_norm):
                        matched = True
                        break

                if not matched and route_city_pairs:
                    # Multi-leg intercity routes can be valid for a current_city
                    # label such as "from A to C" when the legs connect A -> B -> C.
                    # form a continuous city chain with the right endpoints.
                    chain_matches = (
                        route_city_pairs[0][0] == from_city_norm
                        and route_city_pairs[-1][1] == to_city_norm
                    )
                    if chain_matches:
                        for (_, prev_to), (next_from, _) in zip(route_city_pairs, route_city_pairs[1:]):
                            if prev_to != next_from:
                                chain_matches = False
                                break
                    matched = chain_matches
                
                if not matched:
                    # List all intercity transportation routes
                    routes = []
                    for act in intercity_acts:
                        details = act.get("details") or {}
                        act_from = details.get("from", "")
                        act_to = details.get("to", "")
                        routes.append(f"{act_from}→{act_to}")
                    
                    if unresolved_routes:
                        violations.append(
                            f"D{day_idx}: current_city is '{from_city}→{to_city}' but intercity station names could not be resolved to cities "
                            f"(unresolved: {unresolved_routes}, actual: {routes})"
                        )
                    else:
                        violations.append(
                            f"D{day_idx}: current_city is '{from_city}→{to_city}' but intercity transportation route does not match (actual: {routes})"
                        )
            
            # Update current location
            current_location = to_city_norm
            
        else:
            # Case 2: current_city = "some city" (single city)
            city_norm = normalize_city(current_city)
            
            if not city_norm:
                violations.append(f"D{day_idx}: current_city is empty or invalid")
                continue
            
            # Check if this city equals current location
            if city_norm != current_location:
                violations.append(
                    f"D{day_idx}: current_city is '{current_city}' but current location should be '{current_location}', "
                    f"missing intercity transportation info (should be written as 'from {current_location} to {current_city}')"
                )
                # Note: Don't update current_location here, as this is an error state

    if current_location != normalize_city(meta.get("org")):
        violations.append(
            f"Trip ends in '{current_location}', expected return to origin '{normalize_city(meta.get('org'))}'"
        )
    
    if violations:
        return False, f"Location tracking inconsistent: {violations}"
    return True, None


# ==============================================================================
# DIMENSION 2: Database Grounding
# Checks: validated_accommodation, validated_attractions, validated_meals, validated_transportation
# ==============================================================================

def check_accommodation_traceable(daily_plans: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Check if accommodation is traceable (both hotel activity and accommodation field present)."""
    if not daily_plans:
        return False, "Missing daily_plans"
    missing_days: List[int] = []
    for i, day in enumerate(daily_plans[:-1]):  # Except last day, must have accommodation
        has_hotel_act = any(True for _ in iter_hotel_acts([day]))
        accom = day.get("accommodation")
        has_accom_field = bool(accom)
        if not (has_hotel_act and has_accom_field):
            missing_days.append(i + 1)
    # Last day: allow accommodation field, but name must be "-" (indicating no accommodation) or empty
    last_day = daily_plans[-1]
    last_accom = last_day.get("accommodation")
    if last_accom:
        # If accommodation is a dict, check if name is "-" or empty
        if isinstance(last_accom, dict):
            last_accom_name = (last_accom.get("name") or "").strip()
            # Only report error when name exists and is not "-"
            if last_accom_name and last_accom_name != "-":
                if missing_days:
                    return False, f"Accommodation not traceable on days: {missing_days}; last day accommodation.name should be '-' or empty, actual '{last_accom_name}'"
                return False, f"Last day accommodation.name should be '-' or empty, actual '{last_accom_name}'"
        else:
            # If accommodation is not a dict, consider it invalid
            if missing_days:
                return False, f"Accommodation not traceable on days: {missing_days}; last day accommodation should be empty or name '-'"
            return False, "Last day accommodation should be empty or name '-'"

    if missing_days:
        return False, f"Accommodation not traceable on days: {missing_days}"
    return True, None

def check_last_activity_is_hotel(daily_plans: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Check if last activity of each day (except last day) is hotel."""
    if not daily_plans:
        return False, "Missing daily_plans"
    invalid_days: List[int] = []
    for i, day in enumerate(daily_plans[:-1]):  # Except last day
        activities = day.get("activities", []) or []
        if not activities:
            invalid_days.append(i + 1)
            continue
        last_act = activities[-1]
        if last_act.get("type") != "hotel":
            invalid_days.append(i + 1)
    if invalid_days:
        return False, f"Last activity not hotel on days: {invalid_days}"
    return True, None

def check_day_boundary_continuity(
    daily_plans: List[Dict[str, Any]],
    hotels_index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[bool, Optional[str]]:
    """Check whether the next day starts from the previous day's accommodation city."""
    if len(daily_plans) <= 1:
        return True, None

    violations: List[str] = []
    for day_idx in range(len(daily_plans) - 1):
        current_day = daily_plans[day_idx]
        next_day = daily_plans[day_idx + 1]
        stay_city = get_day_accommodation_city(current_day, hotels_index)
        if not stay_city:
            continue

        next_current_city = str(next_day.get("current_city", "")).strip()
        next_from_city, next_to_city = extract_from_to(next_current_city)
        if next_from_city and next_to_city:
            expected_city = normalize_city(next_from_city)
        else:
            expected_city = normalize_city(next_current_city)

        if expected_city and normalize_city(stay_city) != expected_city:
            violations.append(
                f"D{day_idx + 1}->D{day_idx + 2}: previous night stays in '{stay_city}' but next day starts from '{expected_city}'"
            )

    if violations:
        return False, f"Day boundary continuity mismatch: {violations}"
    return True, None

def check_meal_necessity(daily_plans: List[Dict[str, Any]], meta: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Check if daily meal arrangements comply with meal rules.
    Returns ratio of days with correct meal arrangements (scored per day).

    Rules:
    Breakfast is not required and does not count toward essential meal coverage.
    1. Non-intercity days: Must arrange lunch and dinner with gap >= 120 minutes
    2. Intercity day arriving at tourist destination (non-org):
       - Arrive < 10:00: must have lunch and dinner, gap >= 120 minutes
       - Arrive 10:00-15:00: must have dinner; lunch optional
       - Arrive > 15:00: no meal or dinner only
    3. Intercity day leaving tourist city (non-org):
       - Leave < 09:00: 0 meals
       - Leave 09:00–15:00: lunch optional; no dinner
       - Leave 15:00+: at least lunch; dinner optional
    """
    violations: List[str] = []
    total_days = len(daily_plans)
    correct_days = 0

    def _get_start_time_minutes(act: Dict[str, Any]) -> Optional[int]:
        """Get activity start time (in minutes)."""
        start_time = act.get("start_time")
        if not start_time:
            ts = act.get("time_slot", "")
            if ts and "-" in ts:
                start_time = ts.split("-")[0]
        if not start_time:
            return None
        try:
            h, m = map(int, start_time.split(":"))
            return h * 60 + m
        except:
            return None

    def _get_end_time_minutes(act: Dict[str, Any]) -> Optional[int]:
        """Get activity end time (in minutes)."""
        end_time = act.get("end_time")
        if not end_time:
            ts = act.get("time_slot", "")
            if ts and "-" in ts:
                end_time = ts.split("-")[1]
        if not end_time:
            return None
        try:
            h, m = map(int, end_time.split(":"))
            return h * 60 + m
        except:
            return None

    def _meal_label(act: Dict[str, Any]) -> str:
        details = act.get("details") or {}
        return str(details.get("meal_type") or details.get("type") or details.get("label") or "").lower()

    def _looks_like_lunch(label: str, st_min: Optional[int]) -> bool:
        if "lunch" in label:
            return True
        if "breakfast" in label or "dinner" in label:
            return False
        return st_min is not None and 11 * 60 <= st_min <= 13 * 60 + 45

    def _looks_like_dinner(label: str, st_min: Optional[int]) -> bool:
        if "dinner" in label:
            return True
        if "breakfast" in label or "lunch" in label:
            return False
        return st_min is not None and 17 * 60 <= st_min <= 20 * 60 + 45

    def _check_lunch_dinner_gap(
        lunch_times: List[Tuple[int, int]],
        dinner_times: List[Tuple[int, int]],
        day_idx: int,
    ) -> None:
        """Check if lunch end to dinner start gap is >= 120 minutes."""
        if lunch_times and dinner_times:
            lunch_end = max(item[1] for item in lunch_times)
            dinner_start = min(item[0] for item in dinner_times)
            gap = dinner_start - lunch_end
            if gap < 120:
                violations.append(
                    f"D{day_idx}: Gap between lunch end and dinner start less than 2 hours (gap {gap} minutes)"
                )

    def _before(times: List[Tuple[int, int]], cutoff_minutes: int) -> List[Tuple[int, int]]:
        return [item for item in times if item[0] < cutoff_minutes]

    def _after(times: List[Tuple[int, int]], cutoff_minutes: int) -> List[Tuple[int, int]]:
        return [item for item in times if item[0] >= cutoff_minutes]

    # Get org city
    org_city = normalize_city(meta.get("org"))
    if not org_city:
        return False, "Missing org info, cannot determine meal necessity"

    current_location = org_city

    # Check each day
    for day_idx, day in enumerate(daily_plans, start=1):
        current_city = day.get("current_city", "")
        from_city, to_city = extract_from_to(current_city)
        
        # Collect lunch/dinner meals for the day. Breakfast is intentionally
        # ignored because the benchmark assumes it is handled at the hotel or
        # before departure and should not satisfy essential meal coverage.
        lunch_times: List[Tuple[int, int]] = []
        dinner_times: List[Tuple[int, int]] = []
        for act in day.get("activities", []) or []:
            if act.get("type") == "meal":
                st_min = _get_start_time_minutes(act)
                ed_min = _get_end_time_minutes(act)
                if st_min is not None and ed_min is not None:
                    label = _meal_label(act)
                    if _looks_like_lunch(label, st_min):
                        lunch_times.append((st_min, ed_min))
                    elif _looks_like_dinner(label, st_min):
                        dinner_times.append((st_min, ed_min))
        is_intercity_day = bool(from_city and to_city)
        day_violations_before = len(violations)
        
        if is_intercity_day:
            from_city_norm = normalize_city(from_city)
            to_city_norm = normalize_city(to_city)
            is_departure = (from_city_norm == current_location)
            is_from_org = (from_city_norm == org_city)
            is_to_org = (to_city_norm == org_city)
            
            # Leaving tourist city (non-org)
            if is_departure and not is_from_org:
                departure_time = get_intercity_departure_time(day)
                if departure_time is not None:
                    departure_minutes = int(round(departure_time * 60))
                    pre_lunch_times = _before(lunch_times, departure_minutes)
                    pre_dinner_times = _before(dinner_times, departure_minutes)
                    pre_meal_times = pre_lunch_times + pre_dinner_times
                    if departure_time < 9:
                        if pre_meal_times:
                            violations.append(f"D{day_idx}: Departure <09:00, should not arrange meals before departure")
                    elif departure_time < 15.0:
                        if pre_dinner_times:
                            violations.append(f"D{day_idx}: Departure 09:00-15:00, should not arrange dinner before departure")
                    else:
                        if not pre_lunch_times:
                            violations.append(f"D{day_idx}: Departure >15:00, must arrange at least lunch before departure")
                        _check_lunch_dinner_gap(pre_lunch_times, pre_dinner_times, day_idx)
            
            # Arriving at tourist destination (non-org)
            if not is_to_org:
                arrival_time = get_intercity_arrival_time(day)
                if arrival_time is not None:
                    arrival_minutes = int(round(arrival_time * 60))
                    post_lunch_times = _after(lunch_times, arrival_minutes)
                    post_dinner_times = _after(dinner_times, arrival_minutes)
                    if arrival_time < 10.0:
                        if not post_lunch_times or not post_dinner_times:
                            violations.append(f"D{day_idx}: Arrival <10:00, must arrange lunch and dinner after arrival")
                        _check_lunch_dinner_gap(post_lunch_times, post_dinner_times, day_idx)
                    elif arrival_time <= 15.0:
                        if not post_dinner_times:
                            violations.append(f"D{day_idx}: Arrival 10:00-15:00, must arrange dinner after arrival")
                        _check_lunch_dinner_gap(post_lunch_times, post_dinner_times, day_idx)
                    else:
                        if post_lunch_times:
                            violations.append(f"D{day_idx}: Arrival >15:00, should not arrange lunch after arrival")
                        if len(post_dinner_times) > 1:
                            violations.append(f"D{day_idx}: Arrival >15:00, should arrange at most one dinner")
            
            current_location = to_city_norm
        else:
            # Non-intercity day
            if not lunch_times or not dinner_times:
                violations.append(f"D{day_idx}: Non-intercity day must arrange lunch and dinner")
            _check_lunch_dinner_gap(lunch_times, dinner_times, day_idx)
        
        # Check if this day is correct (no new violations)
        if len(violations) == day_violations_before:
            correct_days += 1

    # Calculate score
    if total_days == 0:
        return True, None
    
    ratio = correct_days / total_days
    
    if ratio == 1.0:
        return True, None
    
    error_msg = f"Meal necessity: {correct_days}/{total_days} days correct; Violations: {'; '.join(violations)}"
    return False, error_msg

def check_attraction_necessity(daily_plans: List[Dict[str, Any]], meta: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Check if daily attraction arrangements are reasonable.
    
    Rules:
    1. Calculate total duration of attraction-related activities (including attraction visits, travel to/from attractions)
    2. Judge based on available time in destination city:
       - Non-intercity days (full day available): attraction-related duration ≥ 4 hours or ≥2 attractions
       - Intercity day arrival (arrival < 12:00): ≥1 attraction
       - Intercity day arrival (arrival ≥ 12:00): no mandatory requirement
       - Intercity day departure (departure > 16:00): must have at least 1 attraction
       - Other cases: no mandatory requirement
    
    Note: Intercity days departing from or returning to org only check tourist destination city's attractions.
    """
    violations: List[str] = []
    
    def _parse_time_to_hours(time_str: str) -> Optional[float]:
        """Convert time string to hours (float)."""
        if not time_str:
            return None
        try:
            hour, minute = map(int, time_str.split(":"))
            return hour + minute / 60.0
        except:
            return None
    
    def _calculate_duration_minutes(start_str: str, end_str: str) -> int:
        """Calculate duration between two times (in minutes)."""
        start_h = _parse_time_to_hours(start_str)
        end_h = _parse_time_to_hours(end_str)
        if start_h is None or end_h is None:
            return 0
        duration_hours = end_h - start_h
        if duration_hours < 0:
            duration_hours += 24  # Handle day crossover
        return int(duration_hours * 60)
    
    def _get_activity_duration(act: Dict[str, Any]) -> int:
        """Get activity duration (in minutes)."""
        # Priority: use start_time and end_time
        start_time = act.get("start_time", "")
        end_time = act.get("end_time", "")
        
        if not start_time or not end_time:
            # Try to extract from time_slot
            time_slot = act.get("time_slot", "")
            if time_slot and "-" in time_slot:
                parts = time_slot.split("-")
                start_time = parts[0]
                end_time = parts[1] if len(parts) > 1 else ""
        
        if start_time and end_time:
            return _calculate_duration_minutes(start_time, end_time)
        return 0
    
    def _get_attraction_related_duration(day: Dict[str, Any]) -> int:
        """
        Calculate total duration of attraction-related activities for the day (in minutes).
        Includes:
        1. Attraction visit time (type="attraction")
        2. Travel to/from attractions (type="travel_city", from or to is attraction name)
        """
        total_minutes = 0
        activities = day.get("activities", []) or []
        
        # Collect all attraction names
        attraction_names = set()
        for act in activities:
            if act.get("type") == "attraction":
                details = act.get("details") or {}
                name = (details.get("name") or "").strip()
                if name:
                    attraction_names.add(name)
        
        # Calculate attraction-related duration
        for act in activities:
            act_type = act.get("type")
            
            if act_type == "attraction":
                # Attraction visit time
                total_minutes += _get_activity_duration(act)
            
            elif act_type == "travel_city":
                # Check if it's travel to/from attraction
                details = act.get("details") or {}
                from_loc = (details.get("from") or "").strip()
                to_loc = (details.get("to") or "").strip()
                
                # If from or to is an attraction, count in attraction-related time
                if from_loc in attraction_names or to_loc in attraction_names:
                    total_minutes += _get_activity_duration(act)
        
        return total_minutes

    def _attraction_count_in_window(
        day: Dict[str, Any],
        *,
        start_minutes: Optional[int] = None,
        end_minutes: Optional[int] = None,
    ) -> int:
        count = 0
        for act in day.get("activities", []) or []:
            if act.get("type") != "attraction":
                continue
            act_start, _act_end = slot_to_minutes(act.get("time_slot"))
            if act_start is None:
                continue
            if start_minutes is not None and act_start < start_minutes:
                continue
            if end_minutes is not None and act_start >= end_minutes:
                continue
            count += 1
        return count
    
    # Initial location (departure city)
    org_city = normalize_city(meta.get("org"))
    if not org_city:
        return False, "Missing org info, cannot determine attraction necessity"
    
    current_location = org_city
    
    for day_idx, day in enumerate(daily_plans, start=1):
        current_city = day.get("current_city", "")
        from_city, to_city = extract_from_to(current_city)
        
        # Calculate attraction-related duration for the day
        attraction_minutes = _get_attraction_related_duration(day)
        attraction_hours = attraction_minutes / 60.0
        
        # Count number of attractions for the day
        attraction_count = sum(1 for act in day.get("activities", []) or [] if act.get("type") == "attraction")
        
        # Determine if it's an intercity day
        is_intercity_day = bool(from_city and to_city)
        
        if is_intercity_day:
            from_city_norm = normalize_city(from_city)
            to_city_norm = normalize_city(to_city)
            
            is_departure = (from_city_norm == current_location)
            is_from_org = (from_city_norm == org_city)
            is_to_org = (to_city_norm == org_city)
            
            # Departing from org: only check after arrival
            # Returning to org: only check before departure
            # Between tourist cities: need to check both before departure and after arrival
            
            if is_departure and not is_from_org:
                # Leaving tourist city (non-org)
                departure_time = get_intercity_departure_time(day)
                if departure_time is not None:
                    if departure_time > 16.0:
                        # Departure > 16:00: must have at least 1 attraction
                        pre_departure_attraction_count = _attraction_count_in_window(
                            day,
                            end_minutes=int(round(departure_time * 60)),
                        )
                        if pre_departure_attraction_count < 1:
                            violations.append(
                                f"D{day_idx}: Departure time later than 16:00, must arrange at least 1 attraction before departure (current: {pre_departure_attraction_count})"
                            )
            
            if not is_to_org:
                # Arriving at tourist destination city (non-org)
                arrival_time = get_intercity_arrival_time(day)
                if arrival_time is not None:
                    if arrival_time < 12.0:
                        # Arrival < 12:00: ≥1 attraction
                        post_arrival_attraction_count = _attraction_count_in_window(
                            day,
                            start_minutes=int(round(arrival_time * 60)),
                        )
                        if post_arrival_attraction_count < 1:
                            violations.append(
                                f"D{day_idx}: Arrival time earlier than 12:00, must arrange at least 1 attraction after arrival (current: {post_arrival_attraction_count})"
                            )
                    # Arrival ≥ 12:00: no mandatory requirement
            
            # Update current location
            current_location = to_city_norm
            
        else:
            # Non-intercity day: attraction-related duration ≥ 4 hours or ≥2 attractions
            if attraction_hours < 4.0 and attraction_count < 2:
                violations.append(
                    f"D{day_idx}: Non-intercity day requires attraction-related duration ≥ 4 hours or ≥2 attractions (current: {attraction_hours:.1f} hours, {attraction_count} attractions)"
                )
    
    if violations:
        return False, f"Attraction arrangements unreasonable: {'; '.join(violations)}"
    return True, None


# ==============================================================================
# DIMENSION 4: Temporal Consistency
# Checks: no_time_overlaps, reasonable_transfer_time
# ==============================================================================
