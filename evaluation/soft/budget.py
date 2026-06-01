"""Hotel value and cost-sensitive soft-preference checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..costing import compute_total_cost
from ..hard import iter_normalized_hard_constraints
from .common import (
    _count_violation,
    _iter_activities,
    _level_result,
    _not_applicable,
    _numeric_field_quantiles,
    _rule_variants,
    _safe_average,
    _safe_float,
    _source_rule_ids,
)


def _accommodations(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    values: List[Dict[str, Any]] = []
    for day in plan.get("daily_plans", []) or []:
        accom = day.get("accommodation") if isinstance(day, dict) else None
        if isinstance(accom, dict):
            values.append(accom)
    return values


def _hotel_price_thresholds(
    database_dir: Optional[Path | str],
) -> tuple[Dict[str, float], str]:
    thresholds, source = _numeric_field_quantiles(
        database_dir,
        "hotels/hotels.csv",
        ("price_per_night", "price", "cost"),
        {
            "city_price_p25": 0.25,
            "city_price_p50": 0.50,
            "city_price_p75": 0.75,
            "city_price_p90": 0.90,
        },
    )
    if not thresholds:
        thresholds = {
            "city_price_p25": 350.0,
            "city_price_p50": 600.0,
            "city_price_p75": 900.0,
            "city_price_p90": 1200.0,
        }
    return thresholds, source


def _score_hotel(
    plan: Dict[str, Any],
    rule_id: str,
    database_dir: Optional[Path | str] = None,
    source_rule_ids: Optional[Iterable[str]] = None,
    rule_metadata: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Score value-oriented hotel choices for price-sensitive lodging profiles."""
    accommodations = _accommodations(plan)
    prices = [
        _safe_float(
            item.get("price_per_night") or item.get("price") or item.get("cost")
        )
        for item in accommodations
    ]
    prices = [price for price in prices if price is not None]
    has_named_hotel = any(
        str(item.get("name") or "").strip() for item in accommodations
    )
    source_rule_ids = _source_rule_ids(source_rule_ids or [], rule_metadata)
    variants = set(_rule_variants(rule_metadata))

    if rule_id != "hotel_value_first":
        return _not_applicable(
            rule_id,
            "No deterministic hotel soft check is active for this rule.",
            {"has_named_hotel": has_named_hotel, "prices_per_night": prices},
        )

    price_thresholds, price_source = _hotel_price_thresholds(database_dir)
    thresholds = price_thresholds
    threshold_source = price_source
    avg_price = _safe_average([float(price) for price in prices], default=0.0)
    if avg_price and avg_price <= price_thresholds["city_price_p50"]:
        severity = "none"
    elif avg_price and avg_price <= price_thresholds["city_price_p75"]:
        severity = "minor"
    elif avg_price:
        severity = "major"
    else:
        severity = "minor"
    violations = [
        {
            "name": "average_hotel_price_per_night",
            "value": round(float(avg_price or 0), 4),
            "soft_boundary": price_thresholds["city_price_p50"],
            "hard_boundary": price_thresholds["city_price_p75"],
            "severity": severity,
            "unit": "currency",
        }
    ]

    signals = {
        "has_named_hotel": has_named_hotel,
        "prices_per_night": prices,
        "source_rule_ids": source_rule_ids,
        "preference_variants": sorted(variants),
    }

    return _level_result(
        rule_id,
        severity,
        "hotel cost is value-oriented",
        "hotel cost has a value-profile violation",
        signals,
        thresholds=thresholds,
        threshold_source=threshold_source,
        violations=violations,
    )


def _meal_costs(plan: Dict[str, Any]) -> List[float]:
    costs: List[float] = []
    for _day, act in _iter_activities(plan):
        if act.get("type") != "meal":
            continue
        cost = _safe_float((act.get("details") or {}).get("cost"))
        if cost is not None:
            costs.append(cost)
    return costs


def _meal_price_thresholds(
    database_dir: Optional[Path | str],
) -> tuple[Dict[str, float], str]:
    thresholds, source = _numeric_field_quantiles(
        database_dir,
        "restaurants/restaurants.csv",
        ("price_per_person", "price", "cost"),
        {
            "city_price_p25": 0.25,
            "city_price_p50": 0.50,
            "city_price_p75": 0.75,
            "city_price_p90": 0.90,
        },
    )
    if not thresholds:
        thresholds = {
            "city_price_p25": 60.0,
            "city_price_p50": 90.0,
            "city_price_p75": 120.0,
            "city_price_p90": 200.0,
        }
    return thresholds, source


def _score_budget(
    plan: Dict[str, Any],
    meta: Dict[str, Any],
    rule_id: str,
    database_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Score budget-sensitive choices from plan costs and hard budget limits."""
    costs = _meal_costs(plan)
    meal_thresholds, meal_threshold_source = _meal_price_thresholds(database_dir)
    expensive_threshold = meal_thresholds["city_price_p75"]
    very_expensive_threshold = meal_thresholds["city_price_p90"]
    expensive_meals = sum(1 for cost in costs if cost > expensive_threshold)
    very_expensive_meals = sum(1 for cost in costs if cost > very_expensive_threshold)
    total = compute_total_cost(plan, meta)
    max_budget = None
    hard_budget = (
        dict(iter_normalized_hard_constraints(meta)).get("budget_constraint") or {}
    )
    if isinstance(hard_budget, dict):
        max_budget = _safe_float(hard_budget.get("max_budget"))
    if (
        rule_id in {"budget_guarded", "budget_tight_cap"}
        and max_budget is not None
        and max_budget > 0
    ):
        return _not_applicable(
            rule_id,
            "Explicit hard budget cap is active; total-budget satisfaction is scored by budget_constraint.",
            {
                "computed_total_cost": total,
                "max_budget": max_budget,
                "meal_costs": costs,
            },
        )
    thresholds: Dict[str, Any] = {
        "expensive_meal": "city_p75",
        "very_expensive_meal": "city_p90",
        "city_price_quantile_p25": meal_thresholds.get("city_price_p25"),
        "city_price_quantile_p50": meal_thresholds.get("city_price_p50"),
        "city_price_quantile_p75": expensive_threshold,
        "city_price_quantile_p90": very_expensive_threshold,
    }
    threshold_source = meal_threshold_source

    if rule_id == "meal_avoid_expensive":
        if very_expensive_meals or expensive_meals >= 2:
            severity = "major"
        elif expensive_meals:
            severity = "minor"
        else:
            severity = "none"
        violations = [
            _count_violation(
                "expensive_meal_count", expensive_meals, minor_at=1, major_at=2
            ),
            _count_violation(
                "very_expensive_meal_count",
                very_expensive_meals,
                minor_at=1,
                major_at=1,
            ),
        ]
    else:
        if expensive_meals >= 2 or very_expensive_meals:
            severity = "major"
        elif expensive_meals:
            severity = "minor"
        else:
            severity = "none"
        violations = [
            _count_violation(
                "expensive_meal_count_without_budget_cap",
                expensive_meals,
                minor_at=1,
                major_at=2,
            )
        ]

    return _level_result(
        rule_id,
        severity,
        "budget-sensitive choices are respected",
        "cost choices have a budget-profile violation",
        {
            "meal_costs": costs,
            "expensive_meals": expensive_meals,
            "very_expensive_meals": very_expensive_meals,
            "computed_total_cost": total,
            "max_budget": max_budget,
            "city_price_quantile_p25": meal_thresholds.get("city_price_p25"),
            "city_price_quantile_p50": meal_thresholds.get("city_price_p50"),
            "city_price_quantile_p75": expensive_threshold,
            "city_price_quantile_p90": very_expensive_threshold,
        },
        thresholds=thresholds,
        threshold_source=threshold_source,
        violations=violations,
    )
