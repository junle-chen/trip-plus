"""Transport hard constraint builders."""

from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Optional

from query_generation.common import (
    ConstraintSpec,
    SampleContext,
    build_direct_routes,
    filtered_rows_by_direction,
    parse_dt,
    safe_float,
    safe_int,
)
from query_generation.initial_query.constraints.evidence import (
    flight_option,
    flight_options,
    hour_window_text,
    train_option,
    train_options,
    unique_dicts_preserve_order,
    unique_preserve_order,
)

# Sample a transport hard constraint, preferring the requested route mode.
def build_transport_constraint(ctx: SampleContext, db: Dict[str, List[Dict[str, str]]], rng: random.Random, mode: str) -> ConstraintSpec:
    if mode == "flight":
        spec = build_flight_constraint(ctx, db["flights"], rng)
        if spec is not None:
            return spec
    spec = build_train_constraint(ctx, db["trains"], rng)
    if spec is not None:
        return spec
    if mode == "train":
        spec = build_flight_constraint(ctx, db["flights"], rng)
        if spec is not None:
            return spec
    raise RuntimeError(f"Unable to sample transport constraint for sample {ctx.sample_id}")


# Build one train constraint from direct and date-matched route evidence.
def build_train_constraint(ctx: SampleContext, rows: List[Dict[str, str]], rng: random.Random) -> Optional[ConstraintSpec]:
    direct = build_direct_routes(rows)
    raw_direct_outbound = [group[0] for group in direct.values() if filtered_rows_by_direction(group, ctx, "outbound")]
    raw_direct_inbound = [group[0] for group in direct.values() if filtered_rows_by_direction(group, ctx, "inbound")]
    all_outbound = filtered_rows_by_direction(rows, ctx, "outbound")
    all_inbound = filtered_rows_by_direction(rows, ctx, "inbound")
    direct_outbound = raw_direct_outbound
    direct_inbound = raw_direct_inbound
    candidates: List[ConstraintSpec] = []

    common_classes = sorted(
        {
            out["seat_class"]
            for out in all_outbound
            for inn in all_inbound
            if out["seat_class"] == inn["seat_class"] and out["seat_class"] not in {"", "No Seat", "No seat"}
        }
    )
    if common_classes:
        seat_class = rng.choice(common_classes)
        outbound_pool = [row for row in all_outbound if row["seat_class"] == seat_class]
        inbound_pool = [row for row in all_inbound if row["seat_class"] == seat_class]
        outbound = min(outbound_pool, key=lambda x: safe_float(x["price"], 999999))
        inbound = min(inbound_pool, key=lambda x: safe_float(x["price"], 999999))
        candidates.append(
            ConstraintSpec(
                key="train_seat_class",
                data={
                    "constraint_context": f"Use train transport for the round trip, with {seat_class} seats in both directions",
                    "constraint_type": "transport_seat_class_required",
                    "seat_class": seat_class,
                    "outbound_train_no": outbound["train_no"],
                    "inbound_train_no": inbound["train_no"],
                    "acceptable_outbound_train_nos": unique_preserve_order([row["train_no"] for row in outbound_pool]),
                    "acceptable_inbound_train_nos": unique_preserve_order([row["train_no"] for row in inbound_pool]),
                    "acceptable_outbound_train_options": train_options(outbound_pool),
                    "acceptable_inbound_train_options": train_options(inbound_pool),
                },
                visible_hint=f"I prefer train for both directions, with {seat_class} seats.",
                query_bullet=f"Use train transport with {seat_class} seats for both directions.",
            )
        )

    if direct_outbound:
        earliest = min(direct_outbound, key=lambda x: parse_dt(x["dep_datetime"]))
        earliest_dep = parse_dt(earliest["dep_datetime"])
        candidates.append(
            ConstraintSpec(
                key="train_earliest_departure_direct",
                data={
                    "constraint_context": "Choose the earliest direct outbound train",
                    "constraint_type": "superlative_earliest_departure",
                    "outbound_train_no": earliest["train_no"],
                    "outbound_route_index": safe_int(earliest["route_index"]),
                    "outbound_train_type": earliest["train_type"],
                    "outbound_dep_time": earliest["dep_datetime"],
                    "outbound_price": safe_float(earliest["price"]),
                    "is_direct": True,
                    "acceptable_outbound_train_nos": unique_preserve_order(
                        [row["train_no"] for row in direct_outbound if parse_dt(row["dep_datetime"]) == earliest_dep]
                    ),
                    "acceptable_outbound_train_options": train_options(
                        [row for row in direct_outbound if parse_dt(row["dep_datetime"]) == earliest_dep]
                    ),
                },
                visible_hint="For the outbound trip, I want the earliest direct train.",
                query_bullet="Choose the earliest direct outbound train.",
            )
        )

        cheapest = min(direct_outbound, key=lambda x: safe_float(x["price"], 999999))
        cheapest_price = safe_float(cheapest["price"], 999999)
        candidates.append(
            ConstraintSpec(
                key="train_cheapest_direct",
                data={
                    "constraint_context": "For the outbound trip, choose train and select the cheapest direct train",
                    "constraint_type": "superlative_cheapest_direct",
                    "direction": "outbound",
                    "outbound_train_no": cheapest["train_no"],
                    "outbound_route_index": safe_int(cheapest["route_index"]),
                    "outbound_train_type": cheapest["train_type"],
                    "outbound_price": safe_float(cheapest["price"]),
                    "outbound_dep_time": cheapest["dep_datetime"],
                    "is_direct": True,
                    "acceptable_outbound_train_nos": unique_preserve_order(
                        [row["train_no"] for row in direct_outbound if safe_float(row["price"], 999999) == cheapest_price]
                    ),
                    "acceptable_outbound_train_options": train_options(
                        [row for row in direct_outbound if safe_float(row["price"], 999999) == cheapest_price]
                    ),
                },
                visible_hint="For the outbound trip, I prefer train and want the cheapest direct train.",
                query_bullet="Choose the cheapest direct outbound train.",
            )
        )

        shortest = min(direct_outbound, key=lambda x: safe_int(x["duration"], 999999))
        shortest_duration = safe_int(shortest["duration"], 999999)
        candidates.append(
            ConstraintSpec(
                key="train_shortest_duration_direct",
                data={
                    "constraint_context": "For the outbound trip, choose train and select the shortest-duration direct train",
                    "constraint_type": "superlative_shortest_duration",
                    "direction": "outbound",
                    "outbound_train_no": shortest["train_no"],
                    "outbound_route_index": safe_int(shortest["route_index"]),
                    "outbound_train_type": shortest["train_type"],
                    "outbound_duration": safe_int(shortest["duration"]),
                    "outbound_dep_time": shortest["dep_datetime"],
                    "is_direct": True,
                    "acceptable_outbound_train_nos": unique_preserve_order(
                        [row["train_no"] for row in direct_outbound if safe_int(row["duration"], 999999) == shortest_duration]
                    ),
                    "acceptable_outbound_train_options": train_options(
                        [row for row in direct_outbound if safe_int(row["duration"], 999999) == shortest_duration]
                    ),
                },
                visible_hint="For the outbound trip, I prefer train and care most about the shortest direct travel time.",
                query_bullet="Choose the shortest-duration direct outbound train.",
            )
        )

        train_types = [row["train_type"] for row in direct_outbound if row["train_type"]]
        if train_types:
            picked_type = rng.choice(sorted(Counter(train_types).keys()))
            type_rows = [row for row in direct_outbound if row["train_type"] == picked_type]
            type_cheapest = min(type_rows, key=lambda x: safe_float(x["price"], 999999))
            type_cheapest_price = safe_float(type_cheapest["price"], 999999)
            candidates.append(
                ConstraintSpec(
                    key="train_cheapest_train_type",
                    data={
                        "constraint_context": f"For the outbound trip, choose train and select the cheapest direct {picked_type} train",
                        "constraint_type": "superlative_cheapest_train_type",
                        "train_type": picked_type,
                        "outbound_train_no": type_cheapest["train_no"],
                        "outbound_route_index": safe_int(type_cheapest["route_index"]),
                        "outbound_price": safe_float(type_cheapest["price"]),
                        "outbound_dep_time": type_cheapest["dep_datetime"],
                        "is_direct": True,
                        "acceptable_outbound_train_nos": unique_preserve_order(
                            [row["train_no"] for row in type_rows if safe_float(row["price"], 999999) == type_cheapest_price]
                        ),
                        "acceptable_outbound_train_options": train_options(
                            [row for row in type_rows if safe_float(row["price"], 999999) == type_cheapest_price]
                        ),
                    },
                    visible_hint=f"For the outbound trip, I want train and the cheapest direct option within the {picked_type} type.",
                    query_bullet=f"Choose the cheapest direct outbound {picked_type} train.",
                )
            )

        dep_pick = rng.choice(direct_outbound)
        start_hour = parse_dt(dep_pick["dep_datetime"]).hour
        end_hour = min(23, start_hour + 1)
        time_range = hour_window_text(start_hour, end_hour)
        candidates.append(
            ConstraintSpec(
                key="train_departure_time_range",
                data={
                    "constraint_context": f"The outbound train must depart within {time_range}",
                    "constraint_type": "time_window_departure",
                    "time_range": time_range,
                    "start_hour": start_hour,
                    "end_hour": end_hour,
                    "outbound_train_no": dep_pick["train_no"],
                    "outbound_dep_time": dep_pick["dep_datetime"],
                    "acceptable_outbound_train_nos": unique_preserve_order(
                        [
                            row["train_no"]
                            for row in direct_outbound
                            if start_hour <= parse_dt(row["dep_datetime"]).hour <= end_hour
                        ]
                    ),
                    "acceptable_outbound_train_options": unique_dicts_preserve_order(
                        [
                            train_option(row)
                            for row in direct_outbound
                            if start_hour <= parse_dt(row["dep_datetime"]).hour <= end_hour
                        ],
                        lambda item: (item["train_no"], item["route_index"], item["segment_index"]),
                    ),
                },
                visible_hint=f"For the outbound train, I need departure within {time_range}.",
                query_bullet=f"The outbound train must depart within {time_range}.",
            )
        )

    if direct_inbound:
        latest = max(direct_inbound, key=lambda x: parse_dt(x["arr_datetime"]))
        latest_arr = parse_dt(latest["arr_datetime"])
        candidates.append(
            ConstraintSpec(
                key="train_latest_arrival_direct",
                data={
                    "constraint_context": "Choose the latest-arriving direct return train",
                    "constraint_type": "superlative_latest_arrival",
                    "inbound_train_no": latest["train_no"],
                    "inbound_route_index": safe_int(latest["route_index"]),
                    "inbound_train_type": latest["train_type"],
                    "inbound_arr_time": latest["arr_datetime"],
                    "inbound_price": safe_float(latest["price"]),
                    "is_direct": True,
                    "acceptable_inbound_train_nos": unique_preserve_order(
                        [row["train_no"] for row in direct_inbound if parse_dt(row["arr_datetime"]) == latest_arr]
                    ),
                    "acceptable_inbound_train_options": train_options(
                        [row for row in direct_inbound if parse_dt(row["arr_datetime"]) == latest_arr]
                    ),
                },
                visible_hint="For the return trip, I want to arrive as late as feasible so I can spend more time at the destination.",
                query_bullet="Choose the latest-arriving direct return train.",
            )
        )

    return rng.choice(candidates) if candidates else None


# Build one flight constraint from direct and date-matched route evidence.
def build_flight_constraint(ctx: SampleContext, rows: List[Dict[str, str]], rng: random.Random) -> Optional[ConstraintSpec]:
    direct = build_direct_routes(rows)
    raw_direct_outbound = [group[0] for group in direct.values() if filtered_rows_by_direction(group, ctx, "outbound")]
    raw_direct_inbound = [group[0] for group in direct.values() if filtered_rows_by_direction(group, ctx, "inbound")]
    all_outbound = filtered_rows_by_direction(rows, ctx, "outbound")
    all_inbound = filtered_rows_by_direction(rows, ctx, "inbound")
    direct_outbound = raw_direct_outbound
    direct_inbound = raw_direct_inbound
    candidates: List[ConstraintSpec] = []

    common_classes = sorted(
        {
            out["seat_class"]
            for out in all_outbound
            for inn in all_inbound
            if out["seat_class"] == inn["seat_class"] and out["seat_class"]
        }
    )
    if common_classes:
        seat_class = rng.choice(common_classes)
        outbound_pool = [row for row in all_outbound if row["seat_class"] == seat_class]
        inbound_pool = [row for row in all_inbound if row["seat_class"] == seat_class]
        outbound = min(outbound_pool, key=lambda x: safe_float(x["price"], 999999))
        inbound = min(inbound_pool, key=lambda x: safe_float(x["price"], 999999))
        candidates.append(
            ConstraintSpec(
                key="flight_seat_class",
                data={
                    "constraint_context": f"Use flights for the round trip, with {seat_class} in both directions",
                    "constraint_type": "transport_seat_class_required",
                    "seat_class": seat_class,
                    "outbound_flight_no": outbound["flight_no"],
                    "inbound_flight_no": inbound["flight_no"],
                    "acceptable_outbound_flight_nos": unique_preserve_order([row["flight_no"] for row in outbound_pool]),
                    "acceptable_inbound_flight_nos": unique_preserve_order([row["flight_no"] for row in inbound_pool]),
                    "acceptable_outbound_flight_options": flight_options(outbound_pool),
                    "acceptable_inbound_flight_options": flight_options(inbound_pool),
                },
                visible_hint=f"I prefer flying for both directions, with {seat_class}.",
                query_bullet=f"Use flights with {seat_class} for both directions.",
            )
        )

    if direct_outbound:
        cheapest = min(direct_outbound, key=lambda x: safe_float(x["price"], 999999))
        cheapest_price = safe_float(cheapest["price"], 999999)
        candidates.append(
            ConstraintSpec(
                key="flight_cheapest_direct",
                data={
                    "constraint_context": "For the outbound trip, choose the cheapest direct flight",
                    "constraint_type": "superlative_cheapest_direct",
                    "direction": "outbound",
                    "outbound_flight_no": cheapest["flight_no"],
                    "outbound_route_index": safe_int(cheapest["route_index"]),
                    "outbound_airline": cheapest["airline"],
                    "outbound_price": safe_float(cheapest["price"]),
                    "outbound_dep_time": cheapest["dep_datetime"],
                    "is_direct": True,
                    "acceptable_outbound_flight_nos": unique_preserve_order(
                        [row["flight_no"] for row in direct_outbound if safe_float(row["price"], 999999) == cheapest_price]
                    ),
                    "acceptable_outbound_flight_options": flight_options(
                        [row for row in direct_outbound if safe_float(row["price"], 999999) == cheapest_price]
                    ),
                },
                visible_hint="For the outbound trip, please arrange the cheapest direct flight.",
                query_bullet="Choose the cheapest direct outbound flight.",
            )
        )

        shortest = min(direct_outbound, key=lambda x: safe_int(x["duration"], 999999))
        shortest_duration = safe_int(shortest["duration"], 999999)
        candidates.append(
            ConstraintSpec(
                key="flight_shortest_duration_direct",
                data={
                    "constraint_context": "For the outbound trip, choose the shortest-duration direct flight",
                    "constraint_type": "superlative_shortest_duration",
                    "direction": "outbound",
                    "outbound_flight_no": shortest["flight_no"],
                    "outbound_route_index": safe_int(shortest["route_index"]),
                    "outbound_airline": shortest["airline"],
                    "outbound_duration": safe_int(shortest["duration"]),
                    "outbound_dep_time": shortest["dep_datetime"],
                    "is_direct": True,
                    "acceptable_outbound_flight_nos": unique_preserve_order(
                        [row["flight_no"] for row in direct_outbound if safe_int(row["duration"], 999999) == shortest_duration]
                    ),
                    "acceptable_outbound_flight_options": flight_options(
                        [row for row in direct_outbound if safe_int(row["duration"], 999999) == shortest_duration]
                    ),
                },
                visible_hint="For the outbound trip, I care most about flight duration, so please choose the shortest direct flight.",
                query_bullet="Choose the shortest-duration direct outbound flight.",
            )
        )

    if direct_inbound:
        latest = max(direct_inbound, key=lambda x: parse_dt(x["arr_datetime"]))
        arr_hour = parse_dt(latest["arr_datetime"]).hour
        start_hour = max(0, arr_hour - 1)
        end_hour = min(23, arr_hour + 1)
        time_range = hour_window_text(start_hour, end_hour)
        candidates.append(
            ConstraintSpec(
                key="flight_arrival_time_range",
                data={
                    "constraint_context": f"The return flight must arrive within {time_range}",
                    "constraint_type": "time_window_arrival",
                    "time_range": time_range,
                    "start_hour": start_hour,
                    "end_hour": end_hour,
                    "inbound_flight_no": latest["flight_no"],
                    "inbound_arr_time": latest["arr_datetime"],
                    "acceptable_inbound_flight_nos": unique_preserve_order(
                        [
                            row["flight_no"]
                            for row in direct_inbound
                            if start_hour <= parse_dt(row["arr_datetime"]).hour <= end_hour
                        ]
                    ),
                    "acceptable_inbound_flight_options": unique_dicts_preserve_order(
                        [
                            flight_option(row)
                            for row in direct_inbound
                            if start_hour <= parse_dt(row["arr_datetime"]).hour <= end_hour
                        ],
                        lambda item: (item["flight_no"], item["route_index"], item["segment_index"]),
                    ),
                },
                visible_hint=f"For the return flight, I need arrival within {time_range}.",
                query_bullet=f"The return flight must arrive within {time_range}.",
            )
        )

    return rng.choice(candidates) if candidates else None
