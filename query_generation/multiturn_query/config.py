"""Constants for English multi-turn query generation."""

from __future__ import annotations

from query_generation.common import BASE_DIR
from query_generation.initial_query.config import INTERACTION_ARCHETYPES


DEFAULT_INPUT = BASE_DIR / "query" / "query_en" / "single" / "query.json"
DEFAULT_OUTPUT = BASE_DIR / "query" / "query_en" / "multiturn" / "query.json"
DEFAULT_QUERY_ROOT = BASE_DIR / "query" / "query_en" / "multiturn"
DEFAULT_DB_ROOT = BASE_DIR / "database" / "sample" / "en"
DEFAULT_TURN_RENDER_MODEL = "qwen3.6-27b-vllm"
DEFAULT_TURN_RENDER_TEMPERATURE = 1.0
DEFAULT_TURN_RENDER_MAX_TOKENS = 260

INTERACTION_LABELS = {key: value["label"] for key, value in INTERACTION_ARCHETYPES.items()}
INTERACTION_DESCRIPTIONS = {key: value["description"] for key, value in INTERACTION_ARCHETYPES.items()}

CHECK_INITIAL = "satisfy_initial_hard_constraints"
CHECK_PRESERVE = "preserve_explicit_hard_constraints"
