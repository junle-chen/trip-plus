from __future__ import annotations

import re


def canonical_tag_block(text: str, tag: str) -> str:
    if not text:
        return ""
    open_matches = list(re.finditer(rf"<{tag}\b[^>]*>", text, flags=re.IGNORECASE))
    if not open_matches:
        return ""
    tail = text[open_matches[-1].end():]
    close_match = re.search(rf"</{tag}>", tail, flags=re.IGNORECASE)
    if close_match:
        tail = tail[:close_match.start()]
    content = tail.strip()
    if not content:
        return ""
    return f"<{tag}>\n{content}\n</{tag}>"


def looks_like_itinerary_text(text: str) -> bool:
    if not text:
        return False
    itinerary_patterns = (
        r"^\s*Day\s+\d+\s*:",
        r"^\s*\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\s*\|",
        r"\|\s*(?:buffer|travel_intercity_public|travel_city|hotel|meal|attraction)\s*\|",
        r"\bBudget\s+Summary\b",
        r"\bTotal\s+Estimated\s+Budget\b",
    )
    return any(
        re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        for pattern in itinerary_patterns
    )


def looks_like_implicit_clarification(text: str) -> bool:
    if not text:
        return False
    if looks_like_itinerary_text(text):
        return False

    clarification_patterns = (
        r"\bclarif(?:y|ication)\b",
        r"\bplease\s+confirm\b",
        r"\bcould\s+you\s+please\b",
        r"\bwould\s+you\s+(?:like|prefer)\b",
        r"\bdo\s+you\s+(?:want|prefer|have)\b",
        r"\bshould\s+i\b",
        r"\bbefore\s+finali[sz]ing\b",
        r"\bwhich\s+option\b",
        r"\blet\s+me\s+know\b",
        r"\bproceed\s+with\b",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in clarification_patterns)


def strip_model_output_artifacts(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<\|channel\>\s*thought\s*<channel\|>", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\|channel\>\s*thought\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("<channel|>", "")
    cleaned = re.sub(r"<\|tool_call\>.*?<tool_call\|>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<\|tool_call\>.*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"^\s*<plan\b[^>]*>\s*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def extract_plan_content(text: str) -> str:
    if not text:
        return ""

    think_end_matches = list(re.finditer(r"</think>", text, flags=re.IGNORECASE))
    if think_end_matches:
        text = text[think_end_matches[-1].end():]
    text = strip_model_output_artifacts(text)

    clarification_matches = re.findall(
        r"<clarification\b[^>]*>(.*?)</clarification>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if clarification_matches:
        clarification = canonical_tag_block(text, "clarification")
        if clarification:
            return clarification

    no_solution_matches = re.findall(
        r"<no_solution\b[^>]*>(.*?)</no_solution>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if no_solution_matches:
        no_solution = canonical_tag_block(text, "no_solution")
        if no_solution:
            return no_solution

    plan_open_matches = list(re.finditer(r"<plan\b[^>]*>", text, flags=re.IGNORECASE))
    if plan_open_matches:
        start = plan_open_matches[-1].end()
        tail = text[start:]
        close_match = re.search(r"</plan>", tail, flags=re.IGNORECASE)
        content = tail[:close_match.start()] if close_match else tail
        content = content.strip()
        if content:
            return content

    fallback = text.strip()
    if not fallback:
        return ""

    fence_matches = re.findall(r"```(?:[a-zA-Z0-9_-]+)?\n([\s\S]*?)```", fallback)
    if fence_matches:
        fence_cleaned = [match.strip() for match in fence_matches if match.strip()]
        if fence_cleaned:
            return fence_cleaned[-1]

    return fallback


def ensure_plan_wrapped(plan_content: str) -> str:
    if not plan_content:
        return ""

    normalized = plan_content.strip()
    no_solution_matches = re.findall(
        r"<no_solution\b[^>]*>(.*?)</no_solution>",
        normalized,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if no_solution_matches:
        no_solution = canonical_tag_block(normalized, "no_solution")
        if no_solution:
            return no_solution

    clarification_matches = re.findall(
        r"<clarification\b[^>]*>(.*?)</clarification>",
        normalized,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if clarification_matches:
        clarification = canonical_tag_block(normalized, "clarification")
        if clarification:
            return clarification

    if looks_like_implicit_clarification(normalized):
        return f"<clarification>\n{normalized}\n</clarification>"

    if re.match(r"^<plan\b", normalized, flags=re.IGNORECASE):
        return normalized
    return f"<plan>\n{normalized}\n</plan>"
