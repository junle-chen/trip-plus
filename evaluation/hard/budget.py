"""Budget hard-constraint check."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from ..costing import compute_plan_cost


def _eval_budget_constraint(
    constraint_data: Dict, plan: Dict, meta: Dict
) -> Tuple[bool, Optional[str]]:
    """
    Evaluate budget constraint

    Check if the calculated actual budget does not exceed the maximum budget constraint.
    Uses the same calculation logic as the feasibility budget check.

    Args:
        constraint_data: Budget constraint data containing:
            - max_budget: Maximum allowed budget
        plan: Travel plan containing daily_plans
        meta: Query metadata containing people_number and room_number

    Returns:
        (True, None) if actual budget is within limit
        (False, error_message) if budget exceeds limit or data is missing
    """
    max_budget = constraint_data.get("max_budget")

    if max_budget is None:
        return (False, "Budget constraint missing max_budget value")

    try:
        max_budget = float(max_budget)
    except (ValueError, TypeError):
        return (False, f"Invalid max_budget value: {max_budget}")

    # Get daily plans
    daily_plans = plan.get("daily_plans", [])
    if not daily_plans:
        return (False, "Plan missing daily_plans")

    costs = compute_plan_cost(plan, meta)
    calculated_total = costs["total"]

    # Check if budget is within limit
    if calculated_total <= max_budget:
        return (True, None)
    else:
        breakdown = (
            f"Actual budget exceeds limit: {calculated_total:.2f} > {max_budget:.2f} "
            f"(exceeded by {calculated_total - max_budget:.2f}). "
            f"Breakdown: Transportation={costs['transportation']:.2f}, "
            f"Accommodation={costs['accommodation']:.2f}, "
            f"Meals={costs['meals']:.2f}, "
            f"Attractions={costs['attractions']:.2f}"
        )
        return (False, breakdown)
