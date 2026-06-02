"""Configuration constants for English initial-query generation."""

from __future__ import annotations

from query_generation.common import BASE_DIR


DEFAULT_OUTPUT = BASE_DIR / "query" / "query_en" / "single" / "query.json"
DEFAULT_DB_ROOT = BASE_DIR / "database" / "sample"
DEFAULT_QUERY_ROOT = BASE_DIR / "query" / "query_en" / "single"
DEFAULT_INITIAL_RENDER_TEMPERATURE = 0.85
DEFAULT_INITIAL_RENDER_MAX_TOKENS = 320
INTERACTION_ARCHETYPES = {
    "user_state_evolution": {
        "short_id": "A",
        "label": "User-State Evolution",
        "description": "Tracks multi-turn user state, accumulated preferences, and preservation of earlier constraints.",
        "weight": 0.25,
    },
    "request_resolution": {
        "short_id": "B",
        "label": "Request Resolution",
        "description": "Tests ambiguity detection, profile-priority clarification, prior-constraint clarification, no-solution handling, and feasible alternatives.",
        "weight": 0.25,
    },
    "environment_driven_replanning": {
        "short_id": "C",
        "label": "Environment-Driven Replanning",
        "description": "Tests replanning under external changes such as weather, transport, attraction opening, seasonal risk, or price shifts.",
        "weight": 0.25,
    },
    "long_horizon_alignment": {
        "short_id": "D",
        "label": "Long-Horizon Alignment",
        "description": "Tests consistent preservation of user state, explicit constraints, environment changes, and request-resolution history over longer interactions.",
        "weight": 0.25,
    },
}
INTERACTION_ARCHETYPE_ALIASES = {
    str(config["short_id"]): key for key, config in INTERACTION_ARCHETYPES.items()
}
INTERACTION_ARCHETYPE_ALIASES.update(
    {
        "evolving_needs": "user_state_evolution",
        "feedback_reactive": "request_resolution",
        "revision_and_resolution": "request_resolution",
        "disruption_adaptive": "environment_driven_replanning",
        "mixed_interactive": "long_horizon_alignment",
    }
)
INTERACTION_ARCHETYPE_WEIGHTS = [
    (key, config["weight"]) for key, config in INTERACTION_ARCHETYPES.items()
]
CATEGORY_ORDER = ["transport", "hotel", "food", "attraction"]
BUDGET_TRIGGER_PROB = 0.08
BUDGET_SOFT_RULE_IDS = {"budget_guarded", "budget_tight_cap"}
PRACTICAL_ENV_SIGNAL_TYPES = {
    "cross_border",
    "local_payment",
    "document_permit",
    "security_check",
    "time_shift",
    "reservation",
    "ferry_transfer",
    "walk_transfer",
}
ROUTE_CATEGORY_BY_MODE = {"train": "trains", "flight": "flights"}
ROUTE_FILENAME_BY_MODE = {"train": "trains.csv", "flight": "flights.csv"}
DEFAULT_CURATED_DATE_WINDOWS = [
    ("2026-04-30", "2026-05-05"),
]
DEFAULT_MAIN_CURATED_WINDOW = ("2026-04-30", "2026-05-05")
DEFAULT_FALLBACK_CURATED_WINDOW = ("2025-11-01", "2025-11-30")
SEASONAL_COVERAGE_MANIFESTS = ("routes_january", "routes_july")
CURATED_MAIN_REMAINDER_SHARE = 0.85
DESTINATION_COVERAGE_REFERENCE = "database_destination_coverage"
DEFAULT_SEASONAL_ROUTE_MANIFESTS = [
    BASE_DIR / "query_generation" / "initial_query" / "constraints" / "route_manifests" / "routes_january.txt",
    BASE_DIR / "query_generation" / "initial_query" / "constraints" / "route_manifests" / "routes_july.txt",
]


def normalize_interaction_archetype(value: str) -> str:
    value = str(value or "").strip()
    return INTERACTION_ARCHETYPE_ALIASES.get(value, value)


def interaction_targets(count: int) -> dict[str, int]:
    raw = {label: count * weight for label, weight in INTERACTION_ARCHETYPE_WEIGHTS}
    targets = {label: int(value) for label, value in raw.items()}
    assigned = sum(targets.values())
    remainder = count - assigned
    if remainder > 0:
        order = sorted(
            INTERACTION_ARCHETYPE_WEIGHTS,
            key=lambda item: (raw[item[0]] - int(raw[item[0]]), item[1]),
            reverse=True,
        )
        for label, _ in order[:remainder]:
            targets[label] += 1
    return targets



def category_label(category: str) -> str:
    return {
        "transport": "intercity transport",
        "hotel": "lodging",
        "food": "dining",
        "attraction": "attraction",
    }[category]
