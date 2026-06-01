"""Timing, transfer, and venue-availability checks for itinerary feasibility."""

from __future__ import annotations

import csv
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..scoring_config import get_intercity_buffer_requirements
from ..utils import (
    calculate_day_of_week,
    extract_hotel_name_from_activity,
    get_database_dir,
    get_index_record,
    haversine_km,
    is_all_day,
    is_attraction_closed_on_day,
    is_within_business_hours,
    iter_attraction_acts,
    iter_meal_acts,
    normalize_entity_name,
    parse_duration_hours,
    parse_time_hhmm,
    parse_time_slot,
    resolve_name_coords,
    slot_to_minutes,
)


_DATABASE_DIR = get_database_dir()


def _extract_numeric_value(value: Any) -> Optional[float]:
    """Extract the first numeric token from strings like '31km' or '48min'."""
    if isinstance(value, (int, float)):
        return float(value)
    if value in (None, ""):
        return None
    match = re.search(r"(-?\d+(?:\.\d+)?)", str(value))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None

def check_time_no_overlap(daily_plans: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Check if activities have time overlaps."""
    conflicts: List[str] = []
    for day_idx, day in enumerate(daily_plans, start=1):
        ranges: List[Tuple[int, int, str]] = []
        for act in day.get("activities", []) or []:
            slot = act.get("time_slot")
            if not slot:
                continue
            s, e = slot_to_minutes(slot)
            if s is None or e is None:
                continue
            ranges.append((s, e, act.get("type") or ""))
        ranges.sort(key=lambda x: x[0])
        for i in range(1, len(ranges)):
            prev = ranges[i - 1]
            curr = ranges[i]
            if curr[0] < prev[1]:
                conflicts.append(f"D{day_idx}: {prev[2]} and {curr[2]} have time overlap")
    if conflicts:
        return False, f"Time overlaps exist: {conflicts}"
    return True, None

def check_local_move_sanity(daily_plans: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Check obviously impossible local travel records.

    This is the direct local-transport evidence check used when trajectory tool
    results are unavailable: each `travel_city` record should expose the route,
    mode, duration, and cost used by the plan. Distance remains optional, but
    when present it must be compatible with the stated duration.
    """
    violations: List[str] = []
    for day_idx, day in enumerate(daily_plans, start=1):
        for act in day.get("activities", []) or []:
            if act.get("type") != "travel_city":
                continue
            details = act.get("details") or {}
            src = str(details.get("from") or "").strip()
            dst = str(details.get("to") or "").strip()
            mode = str(
                details.get("mode")
                or details.get("transport_mode")
                or details.get("recommended_mode")
                or ""
            ).strip()
            raw_duration = details.get("duration")
            raw_cost = details.get("cost")
            missing_fields: List[str] = []
            if not src:
                missing_fields.append("from")
            if not dst:
                missing_fields.append("to")
            if not mode:
                missing_fields.append("mode")
            if raw_duration in (None, ""):
                missing_fields.append("duration")
            if raw_cost in (None, ""):
                missing_fields.append("cost")
            if missing_fields:
                violations.append(
                    f"D{day_idx}: {src or '<missing>'}->{dst or '<missing>'} missing local transport fields: {missing_fields}"
                )

            duration_min = _extract_numeric_value(raw_duration)
            cost_value = _extract_numeric_value(raw_cost)
            if raw_duration not in (None, "") and duration_min is None:
                violations.append(
                    f"D{day_idx}: {src or '<missing>'}->{dst or '<missing>'} has invalid local transport duration: {raw_duration}"
                )
            if raw_cost not in (None, "") and cost_value is None:
                violations.append(
                    f"D{day_idx}: {src or '<missing>'}->{dst or '<missing>'} has invalid local transport cost: {raw_cost}"
                )
            elif cost_value is not None and cost_value < 0:
                violations.append(
                    f"D{day_idx}: {src or '<missing>'}->{dst or '<missing>'} has negative local transport cost: {raw_cost}"
                )
            if duration_min is None:
                continue

            distance_km = _extract_numeric_value(details.get("distance"))
            if distance_km is None:
                continue
            if distance_km > 0.5 and duration_min <= 0:
                violations.append(
                    f"D{day_idx}: {src}->{dst} distance {distance_km:.1f}km but duration {duration_min:.0f}min"
                )
            elif distance_km >= 1 and duration_min < max(5.0, distance_km):
                violations.append(
                    f"D{day_idx}: {src}->{dst} distance {distance_km:.1f}km but duration {duration_min:.0f}min"
                )
            elif distance_km >= 5 and duration_min < 5:
                violations.append(
                    f"D{day_idx}: {src}->{dst} distance {distance_km:.1f}km but duration {duration_min:.0f}min"
                )
            elif distance_km >= 20 and duration_min < 15:
                violations.append(
                    f"D{day_idx}: {src}->{dst} distance {distance_km:.1f}km but duration {duration_min:.0f}min"
                )
            elif distance_km >= 50 and duration_min < 30:
                violations.append(
                    f"D{day_idx}: {src}->{dst} distance {distance_km:.1f}km but duration {duration_min:.0f}min"
                )

    if violations:
        return False, f"Local travel records implausible: {violations}"
    return True, None


@lru_cache(maxsize=512)

def _load_distance_duration_minutes(distance_matrix_path: str) -> Dict[Tuple[str, str], float]:
    durations: Dict[Tuple[str, str], float] = {}
    try:
        with open(distance_matrix_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                origin = row.get("origin")
                destination = row.get("destination")
                if not origin or not destination:
                    continue
                try:
                    durations[(origin, destination)] = float(row.get("duration_minutes") or "")
                except Exception:
                    continue
    except Exception:
        return {}
    return durations

def check_transfer_time_reasonable(daily_plans: List[Dict[str, Any]], locations_index: Optional[Dict[str, Dict[str, Any]]] = None, database_dir: Optional[Path] = None) -> Tuple[bool, Optional[str]]:
    """Check if transfer times between anchor activities are reasonable."""
    violations: List[str] = []
    skipped: List[str] = []
    evaluated_pairs = 0
    anchor_types = {"hotel", "attraction", "meal", "travel_intercity_public"}

    def _coord_key(lat_str: str, lon_str: str) -> str:
        """Generate coordinate key, format 'latitude,longitude', directly use string concatenation to preserve original precision."""
        return f"{lat_str},{lon_str}"

    def _lookup_duration_minutes_in_matrix(olon_str: str, olat_str: str, dlon_str: str, dlat_str: str, mode: str) -> Optional[float]:
        # Note: Database format is "latitude,longitude"
        key_o = _coord_key(olat_str, olon_str)
        key_d = _coord_key(dlat_str, dlon_str)
        
        # Use passed database_dir (if any), otherwise use default global path
        if database_dir is not None:
            db_dir = get_database_dir(database_dir)
            distance_matrix_path = db_dir / "transportation" / "distance_matrix.csv"
        else:
            distance_matrix_path = _DATABASE_DIR / "transportation" / "distance_matrix.csv"

        return _load_distance_duration_minutes(str(distance_matrix_path)).get((key_o, key_d))

    def _resolved_coord_key(value: Any) -> Optional[Tuple[str, str]]:
        if locations_index is None:
            return None
        lat, lon = resolve_name_coords(str(value or ""), locations_index)
        if lat is None or lon is None:
            return None
        return (str(lat), str(lon))

    def _places_match(left: Any, right: Any) -> bool:
        a = normalize_entity_name(str(left or "")).lower()
        b = normalize_entity_name(str(right or "")).lower()
        if not a or not b:
            return False
        if a == b:
            return True
        # Allow only explicit alias/canonical equivalence represented in the
        # sample DB location index. Do not suffix-match station or airport
        # names, because that can silently turn an unresolved entity into a pass.
        left_coords = _resolved_coord_key(left)
        right_coords = _resolved_coord_key(right)
        return left_coords is not None and left_coords == right_coords

    def _explicit_transfer_covers_gap(
        transfer_acts: List[Dict[str, Any]],
        prev_name: str,
        curr_name: str,
        gap_min_without_buffer: float,
    ) -> bool:
        """Accept explicit city-transfer rows as the transfer evidence when they form a location chain."""
        if not transfer_acts or gap_min_without_buffer <= 0:
            return False
        sorted_transfers = sorted(
            transfer_acts,
            key=lambda act: (slot_to_minutes(act.get("time_slot"))[0] or 0),
        )
        current = prev_name
        total_minutes = 0.0
        for act in sorted_transfers:
            details = act.get("details") or {}
            src = (details.get("from") or "").strip()
            dst = (details.get("to") or "").strip()
            if current and src and not _places_match(current, src):
                return False
            s, e = slot_to_minutes(act.get("time_slot"))
            if s is None or e is None or e <= s:
                return False
            total_minutes += e - s
            current = dst or current
        if curr_name and current and not _places_match(current, curr_name):
            return False
        return abs(total_minutes - gap_min_without_buffer) <= 2.0
    
    for day_idx, day in enumerate(daily_plans, start=1):
        anchors: List[Tuple[int, int, Dict[str, Any]]] = []
        activities = day.get("activities", []) or []
        for act in activities:
            if act.get("type") not in anchor_types:
                continue
            s, e = slot_to_minutes(act.get("time_slot"))
            if s is None or e is None:
                continue
            anchors.append((s, e, act))
        
        for i in range(1, len(anchors)):
            prev_s, prev_e, prev_act = anchors[i - 1]
            curr_s, curr_e, curr_act = anchors[i]
            gap_min = curr_s - prev_e

            if prev_e > curr_s:
                # This case includes both time overlap and day crossover, we need to distinguish
                if (prev_e - curr_s) > 12 * 60:  # If time difference exceeds 12 hours, consider it day crossover
                    gap_min += 24 * 60
                else:  # Otherwise consider it time overlap, handled by check_time_no_overlap
                    # We can also ignore here, as another function will check
                    continue

            if gap_min < 0:
                # Non-overlap already handled by check_time_no_overlap, ignore here
                continue
            
            # Calculate buffer time and subtract from gap
            buffer_duration = 0.0
            for act_buf in activities:
                act_type = act_buf.get("type", "").strip()
                if act_type == "buffer":
                    s_buf, e_buf = slot_to_minutes(act_buf.get("time_slot"))

                    if s_buf is None or e_buf is None:
                        continue

                    # ====== Handle day crossover ======
                    # If buffer start time is less than previous anchor end time, it's next day
                    if s_buf < prev_e:
                        s_buf += 1440
                        e_buf += 1440
                    # Similarly, if buffer end time is less than previous anchor end time, add a day
                    elif e_buf < prev_e:
                        e_buf += 1440

                    # If current anchor is early morning next day, also add offset
                    if curr_s < prev_e:
                        curr_s += 1440

                    # Check if buffer is between the two anchors
                    if prev_e <= s_buf and e_buf <= curr_s:
                        buffer_duration += (e_buf - s_buf)

            
            # Subtract buffer time from gap, get actual time interval to verify
            gap_min_without_buffer = gap_min - buffer_duration
            
            # Anchor location names:
            # - Normal anchor (hotel/attraction/meal): use details.name
            # - Intercity anchor (travel_intercity_public):
            #   * As previous anchor, take arrival airport (details.to)
            #   * As next anchor, take departure airport (details.from)
            prev_details = (prev_act.get("details") or {})
            curr_details = (curr_act.get("details") or {})
            if prev_act.get("type") == "travel_intercity_public":
                prev_name = (prev_details.get("to") or prev_act.get("type") or "").strip()
            elif prev_act.get("type") == "hotel":
                prev_name = extract_hotel_name_from_activity(prev_details)
            else:
                prev_name = (prev_details.get("name") or prev_act.get("type") or "").strip()
            if curr_act.get("type") == "travel_intercity_public":
                curr_name = (curr_details.get("from") or curr_act.get("type") or "").strip()
            elif curr_act.get("type") == "hotel":
                curr_name = extract_hotel_name_from_activity(curr_details)
            else:
                curr_name = (curr_details.get("name") or curr_act.get("type") or "").strip()

            # Only when both names can be resolved to coordinates, look up distance_matrix; otherwise skip (record)
            if prev_name and curr_name:
                transfer_acts = [
                    act for act in activities
                    if act.get("type") == "travel_city"
                    and (slot := slot_to_minutes(act.get("time_slot"))) != (None, None)
                    and prev_e <= slot[0] and slot[1] <= curr_s
                ]
                if _explicit_transfer_covers_gap(transfer_acts, prev_name, curr_name, gap_min_without_buffer):
                    evaluated_pairs += 1
                    continue

                if prev_name == curr_name:
                    # Same name, skip directly, don't record as error or skipped item, corresponds to transit type
                    continue
                lat1, lon1 = resolve_name_coords(prev_name, locations_index)
                lat2, lon2 = resolve_name_coords(curr_name, locations_index)
                if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
                    skipped.append(f"D{day_idx}:{prev_name}->{curr_name}")
                    continue
                taxi_min = _lookup_duration_minutes_in_matrix(lon1, lat1, lon2, lat2, "taxi")
                if taxi_min is None:
                    try:
                        distance_km = haversine_km(float(lat1), float(lon1), float(lat2), float(lon2))
                        speed_kmh = 5.0 if distance_km <= 1.5 else 30.0
                        taxi_min = max(1.0, distance_km / speed_kmh * 60.0)
                    except Exception:
                        skipped.append(f"D{day_idx}:{prev_name}->{curr_name}")
                        continue
                evaluated_pairs += 1
                
                # Calculate allowed time range (no longer add extra buffer time, as already excluded from gap)
                # min round down to multiple of 10, max round up to multiple of 10
                min_allowed = min(max(0.0, taxi_min-5),(taxi_min // 10) * 10)
                max_allowed = max(taxi_min+5,math.ceil(taxi_min / 10) * 10)
                if any(
                    (_extract_numeric_value((act.get("details") or {}).get("cost")) or 0) <= 5
                    and (_extract_numeric_value((act.get("details") or {}).get("distance")) or 0) >= 1.5
                    for act in transfer_acts
                ):
                    # query_city_transport_plan may choose subway for low-cost
                    # transfers; the distance matrix is taxi-oriented, so allow
                    # modest extra time for station access and transfers.
                    max_allowed += 20
                
                # Feasibility should reject impossible transfers, not extra slack.
                # A longer-than-direct gap may represent waiting, route choice, or
                # conservative pacing; those belong to quality/efficiency rather
                # than a hard feasibility veto.
                if gap_min_without_buffer < min_allowed:
                    violations.append(
                        f"D{day_idx}:{prev_name}->{curr_name} query got commute time {taxi_min:.0f}min, plan shows gap {gap_min_without_buffer:.0f}min (after excluding buffer {buffer_duration:.0f}min), below minimum {min_allowed:.0f}min\nD{day_idx}:{prev_name}({lat1},{lon1})->{curr_name}({lat2},{lon2}) "
                    )
            else:
                skipped.append(f"D{day_idx}:{prev_name}->{curr_name}")

    if violations:
        reason = f"Anchor transfer time unreasonable: {violations}"
        if skipped:
            reason += f"; Unable to evaluate pairs: {skipped}"
        return False, reason
    if skipped:
        return False, (
            "Transfer time not fully evaluable due to missing coordinates: "
            f"{len(skipped)} skipped, {evaluated_pairs} evaluated, skipped_pairs={skipped}"
        )
    return True, None

def check_intercity_buffer_adequacy(daily_plans: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Check whether intercity segments have enough pre/post-transfer buffer."""
    violations: List[str] = []

    for day_idx, day in enumerate(daily_plans, start=1):
        activities = day.get("activities", []) or []
        for act_idx, act in enumerate(activities):
            if act.get("type") != "travel_intercity_public":
                continue

            details = act.get("details") or {}
            mode = details.get("mode")
            buffer_requirements = get_intercity_buffer_requirements(mode)
            before_required = buffer_requirements["before_minutes"]
            after_required = buffer_requirements["after_minutes"]

            act_start, act_end = slot_to_minutes(act.get("time_slot"))
            if act_start is None or act_end is None:
                continue

            before_buffer = 0
            prev_idx = act_idx - 1
            while prev_idx >= 0 and activities[prev_idx].get("type") == "buffer":
                buf_start, buf_end = slot_to_minutes(activities[prev_idx].get("time_slot"))
                if buf_start is not None and buf_end is not None:
                    before_buffer += max(0, buf_end - buf_start)
                prev_idx -= 1

            after_buffer = 0
            next_idx = act_idx + 1
            while next_idx < len(activities) and activities[next_idx].get("type") == "buffer":
                buf_start, buf_end = slot_to_minutes(activities[next_idx].get("time_slot"))
                if buf_start is not None and buf_end is not None:
                    after_buffer += max(0, buf_end - buf_start)
                next_idx += 1

            if before_buffer < before_required:
                violations.append(
                    f"D{day_idx}: intercity {mode or 'transport'} before-buffer {before_buffer}min < required {before_required}min"
                )
            if after_buffer < after_required:
                violations.append(
                    f"D{day_idx}: intercity {mode or 'transport'} after-buffer {after_buffer}min < required {after_required}min"
                )

    if violations:
        return False, f"Intercity buffer inadequate: {violations}"
    return True, None


# ==============================================================================
# DIMENSION 5: Operating Hours
# Checks: attraction_visit_within_opening_hours, dining_within_service_hours, avoidance_of_closure_days
# ==============================================================================

def check_attractions_in_opening_hours(daily_plans: List[Dict[str, Any]], attractions_index: Dict[str, Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Check if attractions are visited within opening hours."""
    if not attractions_index:
        return False, "Attraction database failed to load or is empty"
    out_of_hours: List[str] = []
    missing_slot: List[str] = []
    for act, _details, name in iter_attraction_acts(daily_plans):
        idx = get_index_record(attractions_index, name)
        if not name or not idx:
            # Handled by authenticity validation
            continue
        slot = act.get("time_slot")
        slot_start, slot_end = parse_time_slot(slot)
        if not slot_start or not slot_end:
            missing_slot.append(name)
            continue

        open_str = (idx.get("opening_time") or "").strip()
        close_str = (idx.get("closing_time") or "").strip()
        # Missing or malformed business hours are data incompleteness, not
        # itinerary infeasibility. Other validators still check entity grounding,
        # duration, continuity, and closure days when those fields are available.
        if not open_str or not close_str:
            continue
        if is_all_day(open_str, close_str):
            continue
        open_t = parse_time_hhmm(open_str)
        close_t = parse_time_hhmm(close_str)
        if not open_t or not close_t:
            continue
        if not is_within_business_hours(slot_start, slot_end, open_t, close_t):
            out_of_hours.append(f"{name}({slot} not within {open_str}-{close_str})")
    if missing_slot:
        return False, f"Missing time_slot: {sorted(set(missing_slot))}"
    if out_of_hours:
        return False, f"Attraction opening hours mismatch: {out_of_hours}"
    return True, None

def check_meals_in_business_hours(daily_plans: List[Dict[str, Any]], restaurants_index: Dict[str, Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Check if meals are scheduled within restaurant business hours."""
    if not restaurants_index:
        return False, "Restaurant database failed to load or is empty"

    out_of_hours: List[str] = []
    missing_slot: List[str] = []

    for act, _details, name in iter_meal_acts(daily_plans):
        restaurant_record = get_index_record(restaurants_index, name)
        if not name or not restaurant_record:
            # Name not in database, handled by source validation, skip here
            continue

        slot = act.get("time_slot")
        slot_start, slot_end = parse_time_slot(slot)
        open_str = (restaurant_record.get("opening_time") or "").strip()
        close_str = (restaurant_record.get("closing_time") or "").strip()
        open_t = parse_time_hhmm(open_str)
        close_t = parse_time_hhmm(close_str)

        # If time_slot is missing, record as error
        if not slot_start or not slot_end:
            missing_slot.append(name)
            continue

        # If business hours are missing, skip this restaurant's check
        if not open_t or not close_t:
            continue

        if not is_within_business_hours(slot_start, slot_end, open_t, close_t):
            out_of_hours.append(f"{name}({slot} not within {open_str}-{close_str})")

    if missing_slot:
        return False, f"Missing time_slot: {sorted(set(missing_slot))}"
    if out_of_hours:
        return False, f"Meal time not within business hours: {out_of_hours}"
    return True, None

def check_attractions_not_closed(
    daily_plans: List[Dict[str, Any]], 
    attractions_index: Dict[str, Dict[str, Any]],
    meta: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Check if attractions are not visited on their closing dates (e.g., closed on Mondays).
    
    Args:
        daily_plans: Daily plan list
        attractions_index: Attraction database index
        meta: Metadata containing depart_weekday
    
    Returns:
        (True, None) if all attractions are visited on open days
        (False, error_message) if any attraction is visited on a closed day
    """
    if not attractions_index:
        return False, "Attraction database failed to load or is empty"
    
    # Get departure weekday from meta (1=Monday, 7=Sunday)
    depart_weekday = meta.get("depart_weekday")
    if not depart_weekday:
        # If depart_weekday is not provided, skip this check
        return True, None
    
    try:
        depart_weekday = int(depart_weekday)
    except (ValueError, TypeError):
        return False, f"Invalid depart_weekday value: {depart_weekday}"
    
    closed_attractions = []
    
    for day_index, day in enumerate(daily_plans):
        # Calculate the weekday for this day
        current_weekday = calculate_day_of_week(depart_weekday, day_index)
        
        # Check all attractions in this day
        for act in day.get("activities", []) or []:
            if act.get("type") != "attraction":
                continue
            
            details = act.get("details") or {}
            name = (details.get("name") or "").strip()
            
            attraction_info = get_index_record(attractions_index, name)
            if not name or not attraction_info:
                # Not in database, handled by authenticity validation
                continue
            closing_dates_str = attraction_info.get("closing_dates")
            
            # Check if attraction is closed on this weekday
            if is_attraction_closed_on_day(closing_dates_str, current_weekday):
                # Map weekday number to name for error message
                weekday_names = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 
                                4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}
                weekday_name = weekday_names.get(current_weekday, str(current_weekday))
                
                closed_attractions.append(
                    f"{name} on Day {day_index + 1} ({weekday_name}), "
                    f"but closed on: {closing_dates_str}"
                )
    
    if closed_attractions:
        return False, f"Attractions visited on closing dates: {'; '.join(closed_attractions)}"
    
    return True, None


# ==============================================================================
# DIMENSION 6: Duration and Buffer Rationality
# Checks: reasonable_duration_at_attractions, reasonable_meal_duration
# ==============================================================================

def check_attractions_duration_reasonable(daily_plans: List[Dict[str, Any]], attractions_index: Dict[str, Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Check if attraction visit durations are within reasonable ranges."""
    if not attractions_index:
        return False, "Attraction database failed to load or is empty"
    duration_invalid: List[str] = []
    for _act, details, name in iter_attraction_acts(daily_plans):
        idx = get_index_record(attractions_index, name)
        if not name or not idx:
            # Handled by authenticity validation
            continue
        # Use activity's time_slot to parse actual visit duration
        time_slot = _act.get("time_slot")
        start_m, end_m = slot_to_minutes(time_slot)
        plan_duration = None
        if start_m is not None and end_m is not None and end_m >= start_m:
            plan_duration = (end_m - start_m) / 60.0
        min_hours = parse_duration_hours(idx.get("min_visit_hours"))
        max_hours = parse_duration_hours(idx.get("max_visit_hours"))
        if plan_duration is None or min_hours is None or max_hours is None:
            duration_invalid.append(f"{name}: Missing duration")
            continue
        if not (min_hours <= plan_duration <= max_hours):
            duration_invalid.append(f"{name}: plan has {plan_duration}h not in recommended {min_hours}-{max_hours}h")
    if duration_invalid:
        return False, f"Attraction visit duration unreasonable: {duration_invalid}"
    return True, None

def check_meal_duration_reasonable(daily_plans: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """
    Check if meal durations are within reasonable ranges.
    
    Args:
        daily_plans: List of daily plans
    
    Rules:
    - A meal should take at least 30 minutes (too short is unrealistic)
    - A meal should not exceed 150 minutes (too long blocks other activities)
    """
    # Duration constraints (in minutes)
    MIN_MEAL_MINUTES = 30
    MAX_MEAL_MINUTES = 150
    
    duration_invalid: List[str] = []
    
    for act, details, name in iter_meal_acts(daily_plans):
        if not name:
            continue
        
        # Use activity's time_slot to parse actual meal duration
        time_slot = act.get("time_slot")
        start_m, end_m = slot_to_minutes(time_slot)
        
        if start_m is None or end_m is None:
            duration_invalid.append(f"{name}: Missing time_slot or invalid format")
            continue
        
        # Calculate duration in minutes
        duration_minutes = end_m - start_m
        if duration_minutes < 0:
            duration_minutes += 24 * 60  # Handle midnight crossover
        
        if duration_minutes < MIN_MEAL_MINUTES:
            duration_invalid.append(
                f"{name}: meal duration {duration_minutes}min < minimum {MIN_MEAL_MINUTES}min"
            )
        elif duration_minutes > MAX_MEAL_MINUTES:
            duration_invalid.append(
                f"{name}: meal duration {duration_minutes}min > maximum {MAX_MEAL_MINUTES}min"
            )
    
    if duration_invalid:
        return False, f"Meal duration unreasonable: {duration_invalid}"
    return True, None


# ==============================================================================
# DIMENSION 7: Cost Arithmetic Consistency
# Checks: cost_calculation_correctness
# ==============================================================================
