"""Simulator model aliases and output naming."""

from __future__ import annotations

import re
from pathlib import Path


DEFAULT_SIMULATOR_MODEL = "qwen3.6-27b-vllm"

SIMULATOR_ALIASES: dict[str, str] = {
    "gpt": "gpt-5.4-nano",
    "gpt-nano": "gpt-5.4-nano",
    "gpt-5.4-nano": "gpt-5.4-nano",
    "gpt54nano": "gpt-5.4-nano",
    "gpt5.4nano": "gpt-5.4-nano",
    "claude": "claude-haiku-4-5-20251001",
    "haiku": "claude-haiku-4-5-20251001",
    "claude-haiku": "claude-haiku-4-5-20251001",
    "claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",
    "claudehaiku45": "claude-haiku-4-5-20251001",
    "claudehaiku4520251001": "claude-haiku-4-5-20251001",
    "gemini": "gemini-3.1-flash-lite",
    "gemini-flash-lite": "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
    "gemini-31-flash-lite": "gemini-3.1-flash-lite",
    "gemini31flashlite": "gemini-3.1-flash-lite",
    "gemini3.1flashlite": "gemini-3.1-flash-lite",
    "qwen": "qwen3.6-27b-vllm",
    "qwen36": "qwen3.6-27b-vllm",
    "qwen3.6": "qwen3.6-27b-vllm",
    "qwen-3.6": "qwen3.6-27b-vllm",
    "qwen3.6-27b": "qwen3.6-27b-vllm",
    "qwen3.6-27b-vllm": "qwen3.6-27b-vllm",
}


def slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return value or "run"


def normalize_simulator_model(value: object) -> str:
    text = str(value or "").strip()
    key = re.sub(r"[\s_]+", "-", text.lower())
    compact = re.sub(r"[^a-z0-9.]+", "", key)
    return SIMULATOR_ALIASES.get(key) or SIMULATOR_ALIASES.get(compact) or text


def default_output_root(simulator_model: str) -> Path:
    return Path("simulation") / simulator_model
