"""Shared scoring constants for travel-plan evaluation.

Keep thresholds and weights here so papers, docs, and evaluator code do not
drift when a rule is adjusted.
"""

from __future__ import annotations

from typing import Dict


FLIGHT_MODE_ALIASES = {"flight", "airplane", "plane", "air"}
TRAIN_MODE_ALIASES = {
    "train",
    "railway",
    "rail",
    "high_speed_rail",
    "high-speed rail",
    "high speed rail",
    "gaotie",
}

INTERCITY_BUFFER_REQUIREMENTS = {
    "flight": {"before_minutes": 90, "after_minutes": 30},
    "train": {"before_minutes": 30, "after_minutes": 15},
    "default": {"before_minutes": 30, "after_minutes": 15},
}

PLAN_QUALITY_WEIGHTS = {
    "route_efficiency": 0.25,
    "generic_temporal_pacing": 0.25,
    "base_convenience": 0.25,
    "experience_diversity": 0.25,
}


def normalize_intercity_mode(mode: object) -> str:
    """Map transport mode labels to evaluator categories."""
    normalized = str(mode or "").strip().lower()
    if normalized in FLIGHT_MODE_ALIASES:
        return "flight"
    if normalized in TRAIN_MODE_ALIASES:
        return "train"
    return "default"


def get_intercity_buffer_requirements(mode: object) -> Dict[str, int]:
    """Return required pre/post buffer minutes for an intercity segment."""
    mode_key = normalize_intercity_mode(mode)
    return dict(INTERCITY_BUFFER_REQUIREMENTS.get(mode_key, INTERCITY_BUFFER_REQUIREMENTS["default"]))
