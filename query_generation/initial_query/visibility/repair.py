"""Repair helpers that make rendered initial queries evaluable without changing intent."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from query_generation.common import date_text, safe_int
from query_generation.initial_query.visibility.checks import (
    _constraint_visible_enough,
    _exactish_reference_visible,
    _loose_reference_visible,
    _name_visible_in_query,
    _query_has_days,
    _query_has_people_number,
    _query_has_room_number,
)


def normalize_query_punctuation(query: str) -> str:
    query = re.sub(r"\s+", " ", query).strip()
    query = query.replace("?.", "?").replace("!.", "!")
    query = query.replace(".;", ".").replace(";.", ".")
    query = re.sub(r"\.{2,}", ".", query)
    query = re.sub(r"[;]{2,}", ";", query)
    query = re.sub(r";\s+(?=[A-Z])", ". ", query)
    sentences = re.split(r"(?<=[.!?])\s+", query)
    deduped_sentences = []
    seen_en = set()
    for sentence in sentences:
        cleaned = sentence.strip()
        key = cleaned.rstrip(".!?").strip().lower()
        if not key or key in seen_en:
            continue
        seen_en.add(key)
        deduped_sentences.append(cleaned)
    if deduped_sentences:
        query = " ".join(deduped_sentences)
    return query


def _append_query_clauses(query: str, clauses: list[str]) -> str:
    cleaned = []
    seen = set()
    for clause in clauses:
        text = str(clause or "").strip().rstrip(".;,) ")
        if not text or text in query or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    if not cleaned:
        return query
    return normalize_query_punctuation(query.rstrip(".;,) ") + ". " + " ".join(clause.rstrip(".") + "." for clause in cleaned))


def _ensure_core_trip_fields_visible(query: str, meta: dict[str, Any]) -> str:
    clauses: list[str] = []
    for dest in meta.get("dest") or []:
        dest_text = str(dest or "").strip()
        if dest_text and dest_text not in query:
            clauses.append(f"The destination is {dest_text}")

    days = safe_int(meta.get("days"), 0)
    if days > 0 and not _query_has_days(query, days):
        clauses.append(f"Please plan this as a {days}-day trip")

    people_number = safe_int(meta.get("people_number"), 0)
    if people_number > 0 and not _query_has_people_number(query, people_number):
        clauses.append(f"There will be {people_number} traveler(s)")

    room_number = safe_int(meta.get("room_number"), 0)
    if room_number > 0 and not _query_has_room_number(query, room_number):
        clauses.append(f"Please plan for {room_number} room(s)")

    return _append_query_clauses(query, clauses)


def _budget_phrase(budget: dict[str, Any]) -> str:
    if "min_budget" in budget and "max_budget" in budget:
        return f"I would like the total budget to stay around {budget['min_budget']}-{budget['max_budget']} RMB."
    if "max_budget" in budget:
        return f"Please keep the total budget under {budget['max_budget']} RMB if possible."
    return ""


def _normalize_budget_mentions(query: str, budget: dict[str, Any]) -> str:
    phrase = _budget_phrase(budget)
    if not phrase:
        return query
    required_numbers = [
        str(budget[key])
        for key in ("min_budget", "max_budget")
        if key in budget
    ]
    if not required_numbers:
        return query

    budget_clause_pattern = re.compile(
        r"(?:[,]\s*)?(?:(?:I|we)\s+(?:would\s+like|want|need|prefer)[^.;\n]{0,80}?\s+)?"
        r"(?:the\s+)?(?:total\s+)?budget[^.;\n]*\d+[^.;\n]*(?:;|\.|\n|$)?",
        flags=re.IGNORECASE,
    )
    kept_required = False

    def replace_clause(match: re.Match[str]) -> str:
        nonlocal kept_required
        clause = match.group(0)
        has_required = all(number in clause for number in required_numbers)
        if has_required and not kept_required:
            kept_required = True
            needs_boundary = (
                match.start() > 0
                and query[match.start() - 1] not in ";,.\n "
            )
            suffix = "\n" if clause.endswith("\n") else ""
            return (". " if needs_boundary else "") + phrase + suffix
        return ""

    normalized = budget_clause_pattern.sub(replace_clause, query)
    normalized = normalize_query_punctuation(normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    if all(number in normalized for number in required_numbers):
        kept_required = True
    if not kept_required:
        normalized = normalized.rstrip(". ") + ". " + phrase
    return normalized


def _ensure_selected_constraints_visible(query: str, meta: dict[str, Any]) -> str:
    missing_hints = []
    for payload in meta.get("t0_structure", {}).get("visible_constraint_payloads", []) or []:
        key = str(payload.get("constraint_key", "")).strip()
        check_payload = dict(payload)
        if key and not check_payload.get("constraint"):
            check_payload["constraint"] = (meta.get("hard_constraints") or {}).get(key, {})
        visible_hint = str(payload.get("visible_hint", "")).strip()
        repair_hint = visible_hint
        if key.startswith("train_") and repair_hint and not any(word in repair_hint.lower() for word in ("train", "rail")):
            repair_hint = "For train travel, " + repair_hint
        if repair_hint and not _constraint_visible_enough(query, check_payload):
            missing_hints.append(repair_hint.rstrip(".; "))
    if not missing_hints:
        return query
    return _append_query_clauses(query, missing_hints)


def _remove_redundant_appended_hints(query: str, meta: dict[str, Any]) -> str:
    for payload in meta.get("t0_structure", {}).get("visible_constraint_payloads", []) or []:
        key = str(payload.get("constraint_key", "")).strip()
        check_payload = dict(payload)
        if key and not check_payload.get("constraint"):
            check_payload["constraint"] = (meta.get("hard_constraints") or {}).get(key, {})
        visible_hint = str(payload.get("visible_hint", "")).strip().rstrip(".;, ")
        if not visible_hint:
            continue
        for pattern in (
            f". Also, {visible_hint}.",
            f". Also, {visible_hint}",
            f"? Also, {visible_hint}.",
            f"? Also, {visible_hint}",
            f"! Also, {visible_hint}.",
            f"! Also, {visible_hint}",
            f". {visible_hint}.",
        ):
            if pattern not in query:
                continue
            candidate = query.replace(pattern, ".", 1)
            if _constraint_visible_enough(candidate, check_payload):
                query = candidate
                break
    return normalize_query_punctuation(query)


def _remove_redundant_lingering_additions(query: str, meta: dict[str, Any]) -> str:
    """Drop appended English repair clauses if the natural wording suffices."""
    hard_constraints = meta.get("hard_constraints") or {}
    payloads = meta.get("t0_structure", {}).get("visible_constraint_payloads", []) or []
    for payload in payloads:
        key = str(payload.get("constraint_key") or "")
        constraint = hard_constraints.get(key, {})
        if not isinstance(constraint, dict):
            continue
        check_payload = dict(payload)
        check_payload["constraint"] = constraint
        for match in reversed(list(re.finditer(r"Also,\s+[^.!?]*[.!?]?", query, flags=re.IGNORECASE))):
            if match.start() > 0 and query[match.start() - 1] not in ".!?":
                continue
            sentence = match.group(0)
            sentence_lower = sentence.lower()
            should_consider = False
            if key.startswith("restaurant_"):
                anchor = str(constraint.get("attraction_name") or "").strip()
                should_consider = bool(anchor and anchor in sentence and any(word in sentence_lower for word in ("restaurant", "meal", "food", "dining")))
            elif key == "attraction_avoid_high_queue":
                should_consider = any(word in sentence_lower for word in ("queue", "line", "crowd"))
            elif key.startswith("attraction_"):
                should_consider = any(
                    str(name).strip() and _name_visible_in_query(sentence, str(name))
                    for name in constraint.get("attraction_names", []) or []
                )
            elif key.startswith("hotel_") or key == "hotel_price_range":
                should_consider = (
                    ("hotel" in sentence_lower or "room" in sentence_lower or "stay" in sentence_lower)
                    and "destination" not in sentence_lower
                )
            if not should_consider:
                continue
            start = match.start() - 1 if match.start() > 0 and query[match.start() - 1] in ".!?" else match.start()
            candidate = query[:start] + "." + query[match.end() :]
            candidate = normalize_query_punctuation(candidate)
            if key.startswith("restaurant_") and not _loose_reference_visible(candidate, constraint.get("attraction_name")):
                continue
            if _constraint_visible_enough(candidate, check_payload):
                query = candidate
                break

    people_number = safe_int(meta.get("people_number"), 0)
    if people_number > 0:
        for match in list(re.finditer(r"[.!?]\s*Also,\s+there will be\s+\d+\s+(?:traveler|travelers|people)[.!?]", query, flags=re.IGNORECASE)):
            candidate = normalize_query_punctuation(query[: match.start()] + "." + query[match.end() :])
            if _query_has_people_number(candidate, people_number):
                query = candidate
                break
    return normalize_query_punctuation(query)


def _remove_redundant_people_tail(query: str, meta: dict[str, Any]) -> str:
    people_number = safe_int(meta.get("people_number"), 0)
    if people_number <= 0:
        return query
    pattern = re.compile(
        rf"(?:[.!?;]\s*)?Also,\s+there will be\s+{people_number}\s+(?:traveler|travelers|people)[.!?;]?",
        flags=re.IGNORECASE,
    )
    for match in reversed(list(pattern.finditer(query))):
        candidate = normalize_query_punctuation((query[: match.start()] + "." + query[match.end() :]).strip("."))
        if _query_has_people_number(candidate, people_number):
            query = candidate
    return normalize_query_punctuation(query)


def _remove_duplicate_transport_hard_phrase(query: str, meta: dict[str, Any]) -> str:
    hard_constraints = meta.get("hard_constraints") or {}
    phrase_specs = [
        ("flight_shortest_duration_direct", "For the outbound trip, choose the shortest-duration direct flight."),
        ("flight_cheapest_direct", "For the outbound trip, choose the cheapest direct flight."),
        ("train_shortest_duration_direct", "For the outbound trip, choose the shortest-duration direct train."),
        ("train_cheapest_direct", "For the outbound trip, choose the cheapest direct train."),
    ]

    def is_transport_sentence(sentence: str, key: str) -> bool:
        lower = sentence.lower()
        if not any(marker in lower for marker in ("outbound", "departure", "departing", "going there", "to the destination")):
            return False
        if key.startswith("flight_"):
            has_mode = "flight" in lower and any(marker in lower for marker in ("direct", "nonstop", "non-stop"))
        else:
            has_mode = any(marker in lower for marker in ("train", "rail")) and "direct" in lower
        if not has_mode:
            return False
        if "cheapest" in key:
            return any(marker in lower for marker in ("cheapest", "lowest price", "least expensive", "lowest fare"))
        return any(marker in lower for marker in ("shortest", "fastest", "quickest", "shortest-duration", "least travel time"))

    for key, hard_phrase in phrase_specs:
        if key not in hard_constraints:
            continue
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", query) if part.strip()]
        phrase_sentence_indexes = [
            idx for idx, sentence in enumerate(sentences)
            if sentence.rstrip(".!?").strip().lower() == hard_phrase.rstrip(".!?").lower()
        ]
        has_semantic_duplicate = any(
            idx not in phrase_sentence_indexes
            and is_transport_sentence(sentence, key)
            for idx, sentence in enumerate(sentences)
        )
        if not phrase_sentence_indexes or not has_semantic_duplicate:
            continue
        remove_indexes = set(phrase_sentence_indexes)
        query = " ".join(sentence for idx, sentence in enumerate(sentences) if idx not in remove_indexes)
    return normalize_query_punctuation(query)


def _fix_visible_day_night_mismatch(query: str, meta: dict[str, Any]) -> str:
    days = safe_int(meta.get("days"), 0)
    if days <= 0:
        return query
    expected_nights = max(0, days - 1)

    query = re.sub(
        rf"\b{days}\s*days?\s+\d+\s*nights?\b",
        f"{days} days {expected_nights} nights",
        query,
        flags=re.IGNORECASE,
    )
    return normalize_query_punctuation(query)


def _finalize_visible_query_quality(query: str, meta: dict[str, Any]) -> str:
    """Clean generation artifacts while preserving evaluable visible constraints."""
    query = normalize_query_punctuation(query)
    query = _fix_visible_day_night_mismatch(query, meta)
    query = _remove_duplicate_transport_hard_phrase(query, meta)
    query = _remove_redundant_people_tail(query, meta)
    return normalize_query_punctuation(query)


def _harden_visible_hard_constraint_wording(query: str, meta: dict[str, Any]) -> str:
    """Keep planner-visible query wording aligned with hidden hard oracles."""
    hard_constraints = meta.get("hard_constraints") or {}
    if not isinstance(hard_constraints, dict) or not hard_constraints:
        return query

    def hard_transport_sentences(key: str) -> list[str]:
        hard_markers = (
            "must",
            "please",
            "need",
            "required",
            "require",
            "choose",
            "select",
            "arrange",
            "book",
            "cannot change",
            "fixed",
            "locked",
        )
        sentences = [item for item in re.split(r"[.!?;]", query) if item.strip()]
        matched: list[str] = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(marker in sentence_lower for marker in ("if possible", "prefer", "would like", "try to")):
                continue
            has_marker = any(marker in sentence_lower for marker in hard_markers)
            has_direct_flight = "flight" in sentence_lower and any(marker in sentence_lower for marker in ("direct", "nonstop", "non-stop"))
            has_direct_train = any(marker in sentence_lower for marker in ("train", "rail")) and "direct" in sentence_lower
            has_cheapest = any(marker in sentence_lower for marker in ("cheapest", "lowest price", "least expensive", "lowest fare"))
            has_shortest = any(marker in sentence_lower for marker in ("shortest", "fastest", "quickest", "shortest-duration", "least travel time"))
            if key == "flight_cheapest_direct" and has_marker and has_direct_flight and has_cheapest:
                matched.append(sentence)
            if key == "train_cheapest_direct" and has_marker and has_direct_train and has_cheapest:
                matched.append(sentence)
            if key == "flight_shortest_duration_direct" and has_marker and has_direct_flight and has_shortest:
                matched.append(sentence)
            if key == "train_shortest_duration_direct" and has_marker and has_direct_train and has_shortest:
                matched.append(sentence)
        return matched

    replacements = [
        ("flight_cheapest_direct", "For the outbound trip, choose the cheapest direct flight."),
        ("train_cheapest_direct", "For the outbound trip, choose the cheapest direct train."),
        ("flight_shortest_duration_direct", "For the outbound trip, choose the shortest-duration direct flight."),
        ("train_shortest_duration_direct", "For the outbound trip, choose the shortest-duration direct train."),
    ]
    for key, hard_phrase in replacements:
        if key not in hard_constraints:
            continue
        hard_sentences = hard_transport_sentences(key)
        if hard_sentences:
            if len(hard_sentences) > 1:
                query = re.sub(rf"(?:\.|;)?\s*(?:Also,\s*)?{re.escape(hard_phrase)}\.?", ".", query, count=1)
            continue
        patterns = [
            r"(?:outbound|departure|departing|going there)[^.!?;]*?(?:cheapest|lowest price|least expensive|lowest fare)[^.!?;]*?(?:direct|nonstop|non-stop)[^.!?;]*?flight",
            r"(?:outbound|departure|departing|going there)[^.!?;]*?(?:direct|nonstop|non-stop)[^.!?;]*?flight[^.!?;]*?(?:cheapest|lowest price|least expensive|lowest fare)",
            r"(?:outbound|departure|departing|going there)[^.!?;]*?(?:shortest|fastest|quickest|least travel time)[^.!?;]*?(?:direct|nonstop|non-stop)[^.!?;]*?flight",
            r"(?:outbound|departure|departing|going there)[^.!?;]*?(?:direct|nonstop|non-stop)[^.!?;]*?flight[^.!?;]*?(?:shortest|fastest|quickest|least travel time)",
            r"(?:outbound|departure|departing|going there)[^.!?;]*?(?:cheapest|lowest price|least expensive|lowest fare)[^.!?;]*?direct[^.!?;]*?(?:train|rail)",
            r"(?:outbound|departure|departing|going there)[^.!?;]*?direct[^.!?;]*?(?:train|rail)[^.!?;]*?(?:cheapest|lowest price|least expensive|lowest fare)",
            r"(?:outbound|departure|departing|going there)[^.!?;]*?(?:shortest|fastest|quickest|least travel time)[^.!?;]*?direct[^.!?;]*?(?:train|rail)",
            r"(?:outbound|departure|departing|going there)[^.!?;]*?direct[^.!?;]*?(?:train|rail)[^.!?;]*?(?:shortest|fastest|quickest|least travel time)",
        ]
        changed = False
        for pattern in patterns:
            if re.search(pattern, query, flags=re.IGNORECASE):
                query = re.sub(pattern, hard_phrase.rstrip("."), query, count=1, flags=re.IGNORECASE)
                changed = True
                break
        if not changed and hard_phrase not in query:
            query = _append_query_clauses(query, [hard_phrase])

    if "restaurant_cheapest_nearby_attraction" in hard_constraints:
        query = re.sub(r"prefer(?:ably)?\s+([^.!?;]*?)(?:cheap|affordable|not too expensive)", r"please choose \1cheapest", query, flags=re.IGNORECASE)
    if "restaurant_highest_rated" in hard_constraints:
        query = re.sub(r"prefer(?:ably)?\s+([^.!?;]*?(?:highest-rated|highest rated|top-rated|top rated))", r"please choose \1", query, flags=re.IGNORECASE)
    if "hotel_highest_rated" in hard_constraints:
        query = re.sub(r"prefer(?:ably)?\s+([^.!?;]*?(?:highest-rated|highest rated|top-rated|top rated))", r"please choose \1", query, flags=re.IGNORECASE)

    return normalize_query_punctuation(query)


def _canonicalize_loose_reference(query: str, value: object) -> str:
    canonical = str(value or "").strip()
    if not canonical or canonical in query:
        return query
    stripped = re.sub(r"\([^)]*\)", "", canonical).strip()
    candidates = [stripped]
    for candidate in candidates:
        if candidate and candidate in query:
            query = query.replace(candidate, canonical, 1)
            break
    return query


def _ensure_exact_db_terms_visible(query: str, meta: dict[str, Any]) -> str:
    repairs: list[str] = []
    hard_constraints = meta.get("hard_constraints") or {}
    for key, constraint in hard_constraints.items():
        if isinstance(constraint, dict) and key.startswith("restaurant_"):
            query = _canonicalize_loose_reference(query, constraint.get("attraction_name"))
        if key.startswith("restaurant_"):
            anchor = str(constraint.get("attraction_name") or "").strip()
            if anchor and not _loose_reference_visible(query, anchor):
                repairs.append(f"Please make the restaurant choice explicitly near {anchor}")
        if key == "attraction_must_visit_named":
            missing = [
                str(name).strip()
                for name in constraint.get("attraction_names", []) or []
                if str(name).strip() and not _loose_reference_visible(query, name)
            ]
            if missing:
                repairs.append("Please explicitly include these attractions: " + ", ".join(missing))
        if key == "hotel_star_service_required":
            required = str(constraint.get("required_service_label") or "").strip()
            if required and not _exactish_reference_visible(query, required):
                repairs.append(f"The hotel must provide {required}")
        if key.startswith(("flight_", "train_")):
            query_lower = query.lower()
            if key.startswith("train_") and not any(word in query_lower for word in ("train", "rail", "railway")):
                repairs.append("Please explicitly use train or rail for intercity transport")
            seat_class = str(constraint.get("seat_class") or "").strip()
            if seat_class and not _exactish_reference_visible(query, seat_class):
                repairs.append(f"Please use {seat_class} for intercity transport")
    if not repairs:
        return query
    return _append_query_clauses(query, repairs)


def _full_date_with_weekday(date_str: str, weekday_value: int | None = None) -> str:
    if weekday_value is None:
        weekday_value = datetime.strptime(date_str, "%Y-%m-%d").date().isoweekday()
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = weekday_names[weekday_value - 1] if 1 <= weekday_value <= 7 else ""
    return f"{date_str} ({weekday})" if weekday else date_str


def _ensure_depart_date_visible(query: str, meta: dict[str, Any]) -> str:
    depart_date = str(meta.get("depart_date") or "")
    if not depart_date:
        return query
    try:
        year, _, day = depart_date.split("-")
        day_i = int(day)
    except ValueError:
        return query

    full_date = date_text(depart_date)
    full_with_weekday = _full_date_with_weekday(depart_date, safe_int(meta.get("depart_weekday")))
    if depart_date in query:
        return query
    if full_date in query:
        return query
    month_name = datetime.strptime(depart_date, "%Y-%m-%d").strftime("%B")
    english_short = re.compile(rf"\b{month_name}\s+{day_i}(?:,\s*{year})?\b", flags=re.IGNORECASE)
    if english_short.search(query):
        return normalize_query_punctuation(english_short.sub(full_with_weekday, query, count=1))
    return query.rstrip(".;,) ") + f". Departure date: {full_with_weekday}."


def _ensure_return_date_visible(query: str, meta: dict[str, Any]) -> str:
    return_date = str(meta.get("return_date") or "")
    if not return_date:
        return query
    try:
        year, _, day = return_date.split("-")
        day_i = int(day)
    except ValueError:
        return query

    full_date = date_text(return_date)
    full_with_weekday = _full_date_with_weekday(return_date)
    if return_date in query:
        return query
    if full_date in query:
        return query
    month_name = datetime.strptime(return_date, "%Y-%m-%d").strftime("%B")
    english_short = re.compile(rf"\b{month_name}\s+{day_i}(?:,\s*{year})?\b", flags=re.IGNORECASE)
    if english_short.search(query):
        return normalize_query_punctuation(english_short.sub(full_with_weekday, query, count=1))
    return query.rstrip(".;,) ") + f". Return date: {full_with_weekday}."
