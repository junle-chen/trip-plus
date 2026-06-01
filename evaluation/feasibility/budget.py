"""Budget arithmetic check for itinerary feasibility."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..costing import compute_plan_cost


def check_budget_accuracy(plan: Dict[str, Any], daily_plans: List[Dict[str, Any]], meta: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Check if budget summary is accurate compared to calculated costs from daily plans.
    
    Rules:
    1. travel_intercity_public: Count one complete route price per same-day intercity chain, consider people_number
    2. accommodation: price_per_night * nights * room_number
    3. meal: price_per_person * people_number
    4. attraction: ticket_price * people_number
    5. travel_city: taxi/cab by vehicles, metro/bus by people, walking by returned cost
    
    Allow 10% margin of error.
    """
    # Get plan's budget summary
    budget_summary = plan.get("budget_summary", {})
    if not budget_summary:
        return False, "Missing budget_summary in plan"
    
    plan_total = budget_summary.get("total_estimated_budget")
    if plan_total is None:
        return False, "Missing total_estimated_budget in budget_summary"
    
    try:
        plan_total = float(plan_total)
    except:
        return False, f"Invalid total_estimated_budget: {plan_total}"

    subtotal_keys = [
        "transportation",
        "accommodation",
        "meals",
        "attractions_and_tickets",
        "other",
    ]
    parsed_subtotals: Dict[str, float] = {}
    for key in subtotal_keys:
        value = budget_summary.get(key)
        if value in (None, ""):
            continue
        try:
            parsed_subtotals[key] = float(value)
        except (TypeError, ValueError):
            return False, f"Invalid budget subtotal {key}: {value}"
    if parsed_subtotals:
        subtotal_sum = sum(parsed_subtotals.values())
        subtotal_error = abs(subtotal_sum - plan_total)
        subtotal_error_rate = subtotal_error / max(abs(plan_total), 1.0)
        if subtotal_error > 5.0 and subtotal_error_rate > 0.02:
            return (
                False,
                f"Budget subtotal sum mismatch: subtotals sum to {subtotal_sum:.2f}, "
                f"but total_estimated_budget is {plan_total:.2f} "
                f"(subtotals={parsed_subtotals})",
            )
    
    costs = compute_plan_cost(plan, meta)
    calculated_total = costs["total"]
    
    # Check if within 10% margin
    if plan_total == 0:
        if calculated_total == 0:
            return True, None
        else:
            return False, f"Plan shows 0 budget but calculated {calculated_total:.2f}"
    
    error_rate = abs(calculated_total - plan_total) / plan_total
    
    if error_rate <= 0.10:
        return True, None
    else:
        breakdown = (
            f"Budget accuracy failed: Plan total={plan_total:.2f}, "
            f"Calculated total={calculated_total:.2f} "
            f"(Transportation={costs['transportation']:.2f}, "
            f"Accommodation={costs['accommodation']:.2f}, "
            f"Meals={costs['meals']:.2f}, "
            f"Attractions={costs['attractions']:.2f}), "
            f"Error rate={error_rate:.2%} (exceeds 10% threshold)"
        )
        return False, breakdown


# ==============================================================================
# Plan Organization Support Signal: Activity Diversity
# These diversity checks remain organization diagnostics. Minimum meal and
# attraction coverage is part of feasibility's daily_content_coverage.
# ==============================================================================
