"""Extraction helpers for plan and JSON blocks in model outputs."""

from __future__ import annotations

import re
from typing import Optional

from .parser import _clean_plan_line, _DAY_RE, _TIME_ACTIVITY_RE

def extract_last_plan_block(text: str) -> str:
    """
    Prefer the last <plan>...</plan> block from the report.
    If the final answer is a clarification or unsat explanation, prefer the
    last <clarification>...</clarification> or <no_solution>...</no_solution>
    block instead.

    Some inference outputs contain draft plans followed by a final corrected
    plan. Sending only the last plan block to the conversion model reduces
    duplicate / conflicting JSON extraction.
    """
    if not text:
        return ""

    def _trim_repeated_day_one(body: str) -> str:
        day_one_matches = list(
            re.finditer(r"(?im)^\s*\**\s*day\s*1\s*:\s*\**\s*$", body)
        )
        if len(day_one_matches) > 1:
            body = body[day_one_matches[-1].start() :]
        return body.strip()

    matches = re.findall(r"<plan>\s*([\s\S]*?)\s*</plan>", text, flags=re.IGNORECASE)
    clarification_matches = re.findall(
        r"<clarification>\s*([\s\S]*?)\s*</clarification>", text, flags=re.IGNORECASE
    )
    unsat_matches = re.findall(
        r"<no_solution>\s*([\s\S]*?)\s*</no_solution>", text, flags=re.IGNORECASE
    )
    last_plan_end = max(text.lower().rfind("</plan>"), text.lower().rfind("</plan >"))
    last_clarification_end = max(
        text.lower().rfind("</clarification>"), text.lower().rfind("</clarification >")
    )
    last_unsat_end = max(
        text.lower().rfind("</no_solution>"), text.lower().rfind("</no_solution >")
    )
    if clarification_matches and last_clarification_end > max(
        last_plan_end, last_unsat_end
    ):
        last_clarification = clarification_matches[-1].strip()
        if last_clarification:
            return f"<clarification>\n{last_clarification}\n</clarification>"
    if unsat_matches and last_unsat_end > last_plan_end:
        last_unsat = unsat_matches[-1].strip()
        if last_unsat:
            return f"<no_solution>\n{last_unsat}\n</no_solution>"
    if matches:
        last = _trim_repeated_day_one(matches[-1])
        if last:
            return f"<plan>\n{last}\n</plan>"
    return _trim_repeated_day_one(text)


def extract_json_from_response(text: str) -> Optional[str]:
    """Extract JSON content from model output with tolerant fallbacks."""
    if not text:
        return None

    content = text.strip()

    # 1) Preferred format: <JSON>...</JSON>
    match = re.search(r"<JSON>\s*([\s\S]*?)\s*</JSON>", content, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 2) Markdown code block: ```json ... ``` or ``` ... ```
    code_match = re.search(
        r"```(?:json)?\s*([\s\S]*?)\s*```", content, flags=re.IGNORECASE
    )
    if code_match:
        return code_match.group(1).strip()

    # 3) Already raw JSON
    if content.startswith("{") or content.startswith("["):
        return content

    # 4) Free text with reasoning prefix/suffix, extract first balanced JSON block.
    balanced = extract_balanced_json(content)
    if balanced:
        return balanced

    return content


def extract_balanced_json(text: str) -> Optional[str]:
    """Extract the first balanced top-level JSON object/array from free text."""
    start_obj = text.find("{")
    start_arr = text.find("[")
    starts = [idx for idx in (start_obj, start_arr) if idx != -1]
    if not starts:
        return None

    start = min(starts)
    opening = text[start]
    closing = "}" if opening == "{" else "]"

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()

    return None


