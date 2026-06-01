"""Deterministic parsing for canonical travel-plan reports."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from evaluation.scoring_config import normalize_intercity_mode

_DAY_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?\**\s*Day\s+(\d+)(?:\s*(?:[:：]|\().*)?\**\s*$",
    flags=re.IGNORECASE,
)
_TIME_ACTIVITY_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2})\s*[|｜]\s*\*{0,3}\s*([A-Za-z_ -]+)\s*\*{0,3}\s*[|｜]\s*(.*)$",
    flags=re.MULTILINE,
)
_ROUTE_SEPARATOR_RE = re.compile(r"\s*(?:->|→)\s*|\s+-\s+")


def _clean_plan_line(line: str) -> str:
    cleaned = str(line or "").strip()
    cleaned = re.sub(r"^\s*[-*]\s+", "", cleaned).strip()
    for pattern in (
        r"^\*{1,3}\s*([^*:：]+?)\s*[:：]\s*\*{1,3}\s*(.*)$",
        r"^\*{1,3}\s*([^*]+?)\s*\*{1,3}\s*[:：]\s*(.*)$",
    ):
        label_match = re.match(pattern, cleaned)
        if label_match:
            label = label_match.group(1).strip().rstrip(":：")
            return f"{label}: {label_match.group(2).strip()}".strip()
    return cleaned


def _normalize_activity_type(activity_type: str, detail: str) -> tuple[str, str]:
    raw_type = re.sub(r"\s+", " ", str(activity_type or "").strip().strip("*")).lower()
    raw_detail = str(detail or "").strip()
    normalized_mode = normalize_intercity_mode(raw_type)
    if normalized_mode in {"flight", "train"}:
        return "travel_intercity_public", f"{normalized_mode} {raw_detail}".strip()
    return raw_type, raw_detail


def _parse_number(text: Any) -> Optional[float]:
    if text is None:
        return None
    match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", str(text))
    if not match:
        return None
    value = float(match.group(0).replace(",", ""))
    return int(value) if value.is_integer() else value


def _parse_cost(text: str) -> Optional[float]:
    money_match = re.search(r"[￥¥]\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if money_match:
        value = float(money_match.group(1).replace(",", ""))
        return int(value) if value.is_integer() else value
    return None


def _split_detail(detail: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in str(detail or ""):
        if ch in "([（【":
            depth += 1
        elif ch in ")]）】" and depth > 0:
            depth -= 1
        if ch in "，," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)
    part = "".join(buf).strip()
    if part:
        parts.append(part)
    return parts


def _norm_name(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _known_name_set(known_names: Optional[set[str]] = None) -> set[str]:
    return {
        _norm_name(name) for name in (known_names or set()) if str(name or "").strip()
    }


def _exact_name_set(exact_names: Optional[set[str]] = None) -> set[str]:
    return {
        re.sub(r"\s+", " ", str(name or "").strip())
        for name in (exact_names or set())
        if str(name or "").strip()
    }


def _clean_entity_text(
    text: Any,
    known_names: Optional[set[str]] = None,
    exact_names: Optional[set[str]] = None,
) -> str:
    """Remove converter-introduced wrappers without changing real DB names."""
    value = re.sub(r"\s+", " ", str(text or "").strip())
    if not value:
        return ""

    # Models sometimes wrap entity names in square brackets to mark grounding.
    # Brackets are not part of DB names, and they break exact lookup.
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1].strip()
    else:
        value = value.lstrip("[").rstrip("]").strip()

    # Room count belongs to accommodation metadata, not to the hotel name.
    value = re.sub(
        r"\s*\(\s*\d+\s*(?:rooms?|room)\s*\)\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()
    value = re.sub(
        r"\s*\([^)]*(?:[￥¥]|\brmb\b|\byuan\b|\brooms?\b|\broom/night\b)[^)]*\)\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()
    value = re.sub(
        r"\s*[,，]\s*(?:free\s+entry|free\s+admission|no\s+ticket|required\s+ticket|ticket\s*[:：]?\s*[\d.]+|[¥￥]\s*[\d.]+(?:\s*/\s*person)?)\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()

    exact = _exact_name_set(exact_names)
    if exact and value in exact:
        return value

    prefix_stripped = re.sub(
        r"(?i)^\s*(?:"
        r"check[- ]?in at|check in at|stay at|rest at|overnight at|"
        r"accommodation[:：]?|hotel[:：]?|"
        r"breakfast at|lunch at|dinner at|meal at|eat at|visit|visit to|go to"
        r")\s+",
        "",
        value,
    ).strip()
    candidates = [
        re.sub(r"\s*\([^)]*\)\s*$", "", prefix_stripped).strip(),
        re.sub(r"\s*\([^)]*$", "", prefix_stripped).strip(),
        prefix_stripped,
        re.sub(r"(?i)^\s*(?:near|nearby|around|vicinity of)\s+", "", value).strip(),
        re.sub(r"(?i)\s+(?:area|vicinity|nearby)$", "", value).strip(),
        re.sub(r"\s*\([^)]*\)\s*$", "", value).strip(),
        re.sub(r"\s*\([^)]*$", "", value).strip(),
    ]
    if exact:
        for candidate in candidates:
            if candidate and candidate in exact:
                return candidate
        return value

    known = _known_name_set(known_names)
    if known:
        # Strip locative wording only when it recovers a known entity. Do not
        # blindly remove "Area", because many attraction names legitimately
        # contain that word.
        for candidate in candidates:
            if candidate and _norm_name(candidate) in known:
                return candidate
    return value


def _destination_city_from_route_label(text: Any) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    if not value:
        return ""
    patterns = (
        r"(?i)^\s*(?:from\s+)?(.+?)\s+(?:to|->|→)\s+(.+?)\s*$",
        r"^\s*(.+?)\s*(?:->|→)\s*(.+?)\s*$",
    )
    for pattern in patterns:
        match = re.match(pattern, value)
        if match:
            return _clean_entity_text(match.group(2))
    return ""


def _activity_city_from_current_city(current_city: Any) -> str:
    value = _clean_entity_text(current_city)
    return _destination_city_from_route_label(value) or value


def _find_known_entity_in_text(
    text: Any, known_names: Optional[set[str]] = None
) -> str:
    value = str(text or "")
    matches = [
        name
        for name in (known_names or set())
        if name and re.search(re.escape(name), value, flags=re.IGNORECASE)
    ]
    if not matches:
        return ""
    return max(matches, key=len)


def _find_known_entity_at_start(
    text: Any, known_names: Optional[set[str]] = None
) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    if not value:
        return ""
    matches = [
        name
        for name in (known_names or set())
        if name and value.lower().startswith(str(name).lower())
    ]
    return max(matches, key=len) if matches else ""


def _detail_before_cost(detail: str) -> str:
    match = re.search(r"\s*[,，]?\s*[￥¥]\s*[-+]?\d", str(detail or ""))
    if not match:
        return str(detail or "").strip()
    return str(detail or "")[: match.start()].strip(" ,，")


def _route_endpoints_from_known_names(
    detail: str,
    known_names: Optional[set[str]] = None,
    exact_names: Optional[set[str]] = None,
) -> tuple[str, str, str]:
    known = sorted(
        (name for name in (known_names or set()) if name), key=len, reverse=True
    )
    if not known:
        return "", "", ""
    best: tuple[int, int, int, str, str, str] | None = None
    for match in _ROUTE_SEPARATOR_RE.finditer(str(detail or "")):
        left_raw = detail[: match.start()].strip()
        right_raw = detail[match.end() :].strip()
        left_matches = [
            name for name in known if left_raw.lower().endswith(str(name).lower())
        ]
        right_matches = [
            name for name in known if right_raw.lower().startswith(str(name).lower())
        ]
        if not left_matches and not right_matches:
            continue
        if left_matches and not right_matches:
            left = max(left_matches, key=len)
            right_parts = _split_detail(right_raw)
            right = right_parts[0] if right_parts else right_raw
            trailing = right_raw[len(right) :].lstrip(" ,，")
        elif right_matches and not left_matches:
            right = max(right_matches, key=len)
            left = left_raw
            trailing = right_raw[len(right) :].lstrip(" ,，")
        else:
            left = max(left_matches, key=len)
            right = max(right_matches, key=len)
            trailing = right_raw[len(right) :].lstrip(" ,，")
        score = int(bool(left_matches)) + int(bool(right_matches))
        candidate = (
            score,
            len(left) + len(right),
            match.start(),
            left,
            right,
            trailing,
        )
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return "", "", ""
    return (
        _clean_entity_text(best[3], known_names, exact_names),
        _clean_entity_text(best[4], known_names, exact_names),
        best[5],
    )


def _split_route(
    text: str,
    known_names: Optional[set[str]] = None,
    exact_names: Optional[set[str]] = None,
) -> tuple[str, str]:
    text = str(text or "").strip()
    known = _known_name_set(known_names)
    matches = list(_ROUTE_SEPARATOR_RE.finditer(text))
    if not matches:
        return _clean_entity_text(text, known_names, exact_names), ""
    if not known:
        match = matches[0]
        return (
            _clean_entity_text(text[: match.start()].strip(), known_names, exact_names),
            _clean_entity_text(text[match.end() :].strip(), known_names, exact_names),
        )

    best: tuple[int, int, int, str, str] | None = None
    for match in matches:
        left = text[: match.start()].strip()
        right = text[match.end() :].strip()
        if not left or not right:
            continue
        left_known = _norm_name(left) in known
        right_known = _norm_name(right) in known
        score = int(left_known) + int(right_known)
        if score == 0:
            continue
        # Prefer splits that preserve longer known endpoint names. This avoids
        # breaking hotel names such as "Home Inn Neo - Dalian ...".
        candidate = (
            score,
            int(left_known) * len(left) + int(right_known) * len(right),
            -match.start(),
            left,
            right,
        )
        if best is None or candidate > best:
            best = candidate
    if best is not None:
        return (
            _clean_entity_text(best[3], known_names, exact_names),
            _clean_entity_text(best[4], known_names, exact_names),
        )

    match = matches[0]
    return (
        _clean_entity_text(text[: match.start()].strip(), known_names, exact_names),
        _clean_entity_text(text[match.end() :].strip(), known_names, exact_names),
    )


def _parse_intercity_details(
    detail: str,
    known_names: Optional[set[str]] = None,
    exact_names: Optional[set[str]] = None,
) -> Dict[str, Any]:
    parts = _split_detail(detail)
    head = parts[0] if parts else detail
    route = parts[1] if len(parts) > 1 else ""
    from_place, to_place = _split_route(route, known_names, exact_names)
    mode = ""
    number = ""
    head_match = re.match(
        r"(?i)\s*(flight|airplane|plane|air|train|railway|rail|high[-_ ]speed[-_ ]rail|high\s+speed\s+rail|gaotie)\s+([A-Z0-9]+)\s*$",
        head,
    )
    if head_match:
        mode = normalize_intercity_mode(head_match.group(1))
        number = head_match.group(2)
    else:
        number_match = re.search(
            r"\b([A-Z]{1,3}\d{2,5}|\d[A-Z]{1,2}\d{2,5}|\d{2,5})\b", head
        )
        number = number_match.group(1) if number_match else head
        normalized_head_mode = normalize_intercity_mode(head)
        if normalized_head_mode in {"flight", "train"}:
            mode = normalized_head_mode
        elif re.search(r"\b(flight|airplane|plane|air)\b", head, flags=re.IGNORECASE):
            mode = "flight"
        elif re.search(
            r"\b(train|railway|rail|high[-_ ]speed[-_ ]rail|high\s+speed\s+rail|gaotie)\b",
            head,
            flags=re.IGNORECASE,
        ) or re.match(r"^[GDCZKT]\d+", number):
            mode = "train"
    details: Dict[str, Any] = {
        "mode": mode,
        "number": number,
        "from": from_place,
        "to": to_place,
    }
    cost = _parse_cost(detail)
    if cost is not None:
        details["cost"] = cost
    for part in parts[2:]:
        if _parse_cost(part) is None:
            details["seat_class"] = part
            break
    return details


def _parse_city_transport_details(
    detail: str,
    known_names: Optional[set[str]] = None,
    exact_names: Optional[set[str]] = None,
) -> Dict[str, Any]:
    known_from, known_to, trailing = _route_endpoints_from_known_names(
        detail, known_names, exact_names
    )
    parts = _split_detail(trailing if known_from and known_to else detail)
    if known_from and known_to:
        from_place, to_place = known_from, known_to
    else:
        from_place, to_place = _split_route(
            parts[0] if parts else detail, known_names, exact_names
        )
    details: Dict[str, Any] = {"from": from_place, "to": to_place}
    for part in parts if known_from and known_to else parts[1:]:
        lower = part.lower()
        if _parse_cost(part) is not None:
            continue
        if "km" in lower or "meter" in lower or "metre" in lower:
            details["distance"] = part
        elif "min" in lower or "hour" in lower:
            details["duration"] = part
        elif not details.get("mode"):
            details["mode"] = part
    cost = _parse_cost(detail)
    if cost is not None:
        details["cost"] = cost
    return details


def _parse_activity_details(
    activity_type: str,
    detail: str,
    current_city: str,
    known_names: Optional[set[str]] = None,
    exact_names: Optional[set[str]] = None,
) -> Dict[str, Any]:
    parts = _split_detail(detail)
    if activity_type == "travel_intercity_public":
        return _parse_intercity_details(detail, known_names, exact_names)
    if activity_type == "travel_city":
        return _parse_city_transport_details(detail, known_names, exact_names)
    if activity_type == "attraction":
        name_source = _find_known_entity_at_start(
            _detail_before_cost(detail), known_names
        )
        parsed: Dict[str, Any] = {
            "name": _clean_entity_text(
                name_source or (parts[0] if parts else detail), known_names, exact_names
            ),
            "city": _activity_city_from_current_city(current_city),
        }
        cost = _parse_cost(detail)
        if cost is not None:
            parsed["cost"] = cost
        return parsed
    if activity_type == "meal":
        parsed = {
            "meal_type": parts[0] if parts else "",
            "name": _clean_entity_text(
                parts[1] if len(parts) > 1 else (parts[0] if parts else detail),
                known_names,
                exact_names,
            ),
        }
        cost = _parse_cost(detail)
        if cost is not None:
            parsed["cost"] = cost
        return parsed
    if activity_type == "hotel":
        if len(parts) > 1:
            return {
                "activity": ", ".join(parts[:-1]),
                "name": _clean_entity_text(parts[-1], known_names, exact_names),
            }
        inferred_name = _find_known_entity_in_text(detail, known_names)
        return {
            "activity": parts[0] if parts else detail,
            "name": _clean_entity_text(inferred_name, known_names, exact_names)
            if inferred_name
            else "",
        }
    if activity_type == "buffer":
        return {"description": detail}
    return {"description": detail}


def _parse_budget_summary(text: str) -> Dict[str, Any]:
    budget_starts = list(re.finditer(r"budget\s+summary", text, flags=re.IGNORECASE))
    if not budget_starts:
        return {}
    trip_total_matches = re.findall(
        r"Total\s+Trip\s+Estimated\s+Budget\s*:\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if len(budget_starts) > 1 and trip_total_matches:
        total = _parse_number(trip_total_matches[-1])
        if total is not None:
            return {
                "currency": "CNY",
                "total_estimated_budget": total,
                "other": 0,
            }
    budget_text = text[budget_starts[-1].start() :]

    def value(label_pattern: str) -> Optional[float]:
        matches = re.findall(
            rf"{label_pattern}\s*\*{{0,3}}\s*:\s*\*{{0,3}}\s*[￥¥]?\s*([-+]?\d+(?:,\d{{3}})*(?:\.\d+)?)",
            budget_text,
            flags=re.IGNORECASE,
        )
        return _parse_number(matches[-1]) if matches else None

    parsed: Dict[str, Any] = {
        "currency": "CNY",
    }
    field_patterns = {
        "transportation": r"Transportation",
        "accommodation": r"Accommodation",
        "meals": r"Meals",
        "attractions_and_tickets": r"Attractions\s*&\s*Tickets",
        "other": r"Other",
        "total_estimated_budget": r"Total\s+Estimated\s+Budget",
    }
    for key, pattern in field_patterns.items():
        amount = value(pattern)
        if amount is not None:
            parsed[key] = amount
    parsed.setdefault("other", 0)
    return parsed


def _strip_budget_sections_from_plan_body(text: str) -> str:
    """Remove inline budget blocks while preserving later day sections."""

    def is_budget_heading(line: str) -> bool:
        cleaned = _clean_plan_line(re.sub(r"^\s*#{1,6}\s*", "", line)).strip().lower()
        return bool(re.match(r"^budget\s+summary\s*:?\s*$", cleaned))

    def is_top_day_heading(line: str) -> bool:
        cleaned = _clean_plan_line(line).strip()
        if not _DAY_RE.match(cleaned):
            return False
        lower = cleaned.lower()
        budget_terms = (
            " meal",
            " lunch",
            " dinner",
            " breakfast",
            "taxi",
            "metro",
            "flight",
            "train",
            "hotel",
            "accommodation",
            "transport",
            "total",
            " rmb",
            "¥",
            "=",
            "yuan",
        )
        return not any(term in lower for term in budget_terms)

    kept: List[str] = []
    skipping = False
    for line in str(text or "").splitlines():
        if is_budget_heading(line):
            skipping = True
            continue
        if skipping and is_top_day_heading(line):
            skipping = False
        if not skipping:
            kept.append(line)
    return "\n".join(kept)


def _is_plan_block(text: str) -> bool:
    return bool(re.search(r"<\s*plan\s*>", text or "", flags=re.IGNORECASE))


def _has_canonical_plan_shape(text: str) -> bool:
    body = re.sub(r"</?plan>", "", text or "", flags=re.IGNORECASE).strip()
    has_day = any(_DAY_RE.match(_clean_plan_line(line)) for line in body.splitlines())
    return bool(has_day and _TIME_ACTIVITY_RE.search(body))


def _collect_known_entity_names(
    text: str, exact_names: Optional[set[str]] = None
) -> set[str]:
    names: set[str] = set()
    for raw_line in text.splitlines():
        line = _clean_plan_line(raw_line)
        if not line:
            continue
        if line.lower().startswith("accommodation:"):
            value = line.split(":", 1)[1].strip()
            if value and value != "-":
                parts = _split_detail(value)
                if parts:
                    names.add(_clean_entity_text(parts[0], exact_names=exact_names))
            continue
        activity_match = _TIME_ACTIVITY_RE.match(line)
        if not activity_match:
            continue
        _, activity_type, detail = activity_match.groups()
        activity_type, detail = _normalize_activity_type(activity_type, detail)
        parts = _split_detail(detail.strip())
        if activity_type == "attraction" and parts:
            names.add(
                _clean_entity_text(
                    _detail_before_cost(detail.strip()), exact_names=exact_names
                )
            )
        elif activity_type == "meal" and len(parts) > 1:
            names.add(_clean_entity_text(parts[1], exact_names=exact_names))
        elif activity_type == "hotel" and len(parts) > 1:
            names.add(_clean_entity_text(parts[-1], exact_names=exact_names))
        elif activity_type == "travel_intercity_public" and len(parts) > 1:
            left, right = _split_route(parts[1], exact_names=exact_names)
            if left:
                names.add(left)
            if right:
                names.add(right)
    return {name.strip() for name in names if name.strip()}


def _is_generic_hotel_anchor(value: Any) -> bool:
    text = re.sub(r"\s+", " ", str(value or "").strip()).lower()
    return text in {"hotel", "the hotel", "accommodation", "lodging"}


def _postprocess_daily_entities(
    plan: Dict[str, Any],
    known_names: set[str],
    exact_names: Optional[set[str]] = None,
) -> None:
    for day in plan.get("daily_plans", []) or []:
        accom = day.get("accommodation")
        accom_name = ""
        if isinstance(accom, dict):
            accom_name = _clean_entity_text(accom.get("name"), known_names, exact_names)
            accom["name"] = accom_name
        for act in day.get("activities", []) or []:
            details = act.get("details") or {}
            activity_type = act.get("type")
            if activity_type == "hotel":
                name = _clean_entity_text(details.get("name"), known_names, exact_names)
                if not name and accom_name:
                    name = accom_name
                details["name"] = name
            elif activity_type in {"meal", "attraction"}:
                details["name"] = _clean_entity_text(
                    details.get("name"), known_names, exact_names
                )
                if activity_type == "attraction" and details.get("city"):
                    details["city"] = _activity_city_from_current_city(
                        details.get("city")
                    )
            elif activity_type == "travel_city":
                for key in ("from", "to"):
                    value = details.get(key)
                    if _is_generic_hotel_anchor(value) and accom_name:
                        details[key] = accom_name
                    else:
                        details[key] = _clean_entity_text(
                            value, known_names, exact_names
                        )


def deterministic_convert_plan(
    text: str, exact_names: Optional[set[str]] = None
) -> Optional[Dict[str, Any]]:
    """Parse the canonical line-based plan format without another LLM call."""
    if not text:
        return None
    unsat_match = re.search(
        r"<no_solution>\s*([\s\S]*?)\s*</no_solution>", text, flags=re.IGNORECASE
    )
    if unsat_match:
        return {
            "status": "unsat",
            "unsat_explanation": unsat_match.group(1).strip(),
        }
    clarification_match = re.search(
        r"<clarification>\s*([\s\S]*?)\s*</clarification>", text, flags=re.IGNORECASE
    )
    if clarification_match:
        return {
            "status": "clarification",
            "clarification": clarification_match.group(1).strip(),
        }

    body = re.sub(r"</?plan>", "", text, flags=re.IGNORECASE).strip()
    if not _TIME_ACTIVITY_RE.search(body):
        return None

    budget_summary = _parse_budget_summary(body)
    plan_body = _strip_budget_sections_from_plan_body(body)
    known_entity_names = _collect_known_entity_names(plan_body, exact_names)
    parse_names = set(known_entity_names)
    parse_names.update(exact_names or set())
    lines = [line.strip() for line in plan_body.splitlines() if line.strip()]
    daily_plans: List[Dict[str, Any]] = []
    current_day: Optional[Dict[str, Any]] = None

    for raw_line in lines:
        line = _clean_plan_line(raw_line)
        day_match = _DAY_RE.match(line)
        if day_match:
            if current_day:
                daily_plans.append(current_day)
            current_day = {
                "day_number": int(day_match.group(1)),
                "current_city": "",
                "activities": [],
            }
            date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", line)
            if date_match:
                current_day["date"] = date_match.group(1)
            continue
        if current_day is None:
            continue

        clean_line = _clean_plan_line(line)
        if clean_line.lower().startswith("current city:"):
            current_day["current_city"] = clean_line.split(":", 1)[1].strip()
            continue
        if clean_line.lower().startswith("accommodation:"):
            value = clean_line.split(":", 1)[1].strip()
            if value and value != "-":
                name = _split_detail(value)[0] if _split_detail(value) else value
                name = _clean_entity_text(name, parse_names, exact_names)
                accommodation: Dict[str, Any] = {"name": name}
                price = _parse_cost(value)
                if price is not None:
                    accommodation["price_per_night"] = price
                current_day["accommodation"] = accommodation
            continue

        activity_match = _TIME_ACTIVITY_RE.match(clean_line)
        if activity_match:
            time_slot, activity_type, detail = activity_match.groups()
            activity_type, detail = _normalize_activity_type(activity_type, detail)
            current_day["activities"].append(
                {
                    "time_slot": re.sub(r"\s+", "", time_slot),
                    "type": activity_type,
                    "details": _parse_activity_details(
                        activity_type,
                        detail.strip(),
                        str(current_day.get("current_city") or ""),
                        parse_names,
                        exact_names,
                    ),
                }
            )

    if current_day:
        daily_plans.append(current_day)
    if not daily_plans:
        return None
    parsed: Dict[str, Any] = {"daily_plans": daily_plans}
    if budget_summary:
        parsed["budget_summary"] = budget_summary
    _postprocess_daily_entities(parsed, parse_names, exact_names)
    return parsed


def _postprocess_converted_plan(
    plan: Any,
    source_text: str,
    exact_names: Optional[set[str]] = None,
) -> Any:
    if not isinstance(plan, dict) or not isinstance(plan.get("daily_plans"), list):
        return plan
    known_entity_names = _collect_known_entity_names(source_text, exact_names)
    parse_names = set(known_entity_names)
    parse_names.update(exact_names or set())
    _postprocess_daily_entities(plan, parse_names, exact_names)
    return plan

