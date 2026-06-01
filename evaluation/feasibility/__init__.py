"""Deterministic itinerary-feasibility checks."""

from .config import ALL_FEASIBILITY_CHECKS, FEASIBILITY_CHECK_DIMENSIONS, FEASIBILITY_DIMENSIONS
from .runner import calculate_feasibility, eval_itinerary_feasibility

__all__ = [
    "ALL_FEASIBILITY_CHECKS",
    "FEASIBILITY_CHECK_DIMENSIONS",
    "FEASIBILITY_DIMENSIONS",
    "calculate_feasibility",
    "eval_itinerary_feasibility",
]
