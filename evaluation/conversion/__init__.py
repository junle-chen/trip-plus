"""Report conversion helpers."""

__all__ = [
    "ConversionParseError",
    "convert_reports",
    "deterministic_convert_plan",
    "extract_balanced_json",
    "extract_json_from_response",
    "extract_last_plan_block",
    "process_single_report",
]


def __getattr__(name):
    if name in {"ConversionParseError", "convert_reports", "process_single_report"}:
        from . import runner

        return getattr(runner, name)
    if name == "deterministic_convert_plan":
        from .parser import deterministic_convert_plan

        return deterministic_convert_plan
    if name in {
        "extract_balanced_json",
        "extract_json_from_response",
        "extract_last_plan_block",
    }:
        from . import json_extract

        return getattr(json_extract, name)
    raise AttributeError(name)
