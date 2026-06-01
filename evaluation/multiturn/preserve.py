"""Preserve checks for active constraints carried across turns."""

from __future__ import annotations

from typing import Any, Dict, List


def build_preserve_checks(
    must_preserve: List[str],
    *,
    expected_status: str,
    expects_non_itinerary: bool,
    hard_constraints: Dict[str, Any],
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for key in must_preserve:
        if expected_status == "unsat" or expects_non_itinerary:
            checks.append(
                {
                    "name": key,
                    "passed": None,
                    "message": "Skipped for non-itinerary turn; preservation is checked after the request is clarified/resolved",
                }
            )
            continue

        detail = hard_constraints.get(key)
        passed = bool(detail and detail.get("passed"))
        checks.append(
            {
                "name": key,
                "passed": passed,
                "message": None
                if passed
                else (detail or {}).get(
                    "message", "preserved constraint not satisfied"
                ),
            }
        )
    return checks
