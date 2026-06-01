"""Text checks that decide whether rendered queries preserve visible constraints."""

from __future__ import annotations

import json
import re
from typing import Any

from query_generation.common import safe_int


LOCAL_TRANSPORT_SUPPORT_RULE_IDS = {
    "mobility_accessibility",
    "transport_avoid_transfer",
}
LOCAL_TRANSPORT_SUPPORT_CITY_TAGS = {
    "mega_city_metro_transfer",
    "mountain_transfer_city",
    "steep_walk_city",
    "old_town_walk_city",
    "river_city_transfer",
    "lake_city_transfer",
    "three_towns_river_crossing",
}
LOCAL_TRANSPORT_TEXT_PATTERN = re.compile(
    r"\b(metro|subway|taxi|ride-hailing|rideshare|transfer|walking|walk|detour|backtrack|local transport)\b|"
    r"fewer transfers|less walking|shorter transfer|smooth route|avoid backtracking|avoid detours",
    flags=re.IGNORECASE,
)
LOCAL_TRANSPORT_SUPPORT_TEXT_PATTERN = re.compile(
    r"\b(transfer|walking|walk|transport|metro|subway|route|backtracking|detour)\b|"
    r"fewer transfers|less walking|smooth route",
    flags=re.IGNORECASE,
)


def _query_local_transport_signal_text(query: str) -> str:
    text = str(query or "")
    text = re.sub(r'"[^"]*"', "", text)
    text = re.sub(r"'[^']*'", "", text)
    return text


def _hard_constraints_support_nearby_transfer_reason(meta: dict[str, Any]) -> bool:
    hard_constraints = meta.get("hard_constraints") or {}
    if not isinstance(hard_constraints, dict):
        return False
    for key, constraint in hard_constraints.items():
        if any(token in str(key) for token in ("nearby", "closest")):
            return True
        if isinstance(constraint, dict) and any(
            token in str(constraint.get("constraint_type") or "") for token in ("nearby", "closest")
        ):
            return True
    return False


def _query_local_transport_alignment_issue(query: str, meta: dict[str, Any]) -> str | None:
    signal_text = _query_local_transport_signal_text(query)
    if not LOCAL_TRANSPORT_TEXT_PATTERN.search(signal_text):
        return None

    rule_ids = set((meta.get("user_profile") or {}).get("rule_ids") or [])
    if rule_ids & LOCAL_TRANSPORT_SUPPORT_RULE_IDS:
        return None

    soft_constraints_text = json.dumps(meta.get("soft_constraints") or {}, ensure_ascii=False)
    if LOCAL_TRANSPORT_SUPPORT_TEXT_PATTERN.search(soft_constraints_text):
        return None

    city_context = meta.get("city_context") or {}
    city_tags = set(city_context.get("city_tags") or [])
    if city_tags & LOCAL_TRANSPORT_SUPPORT_CITY_TAGS:
        return None

    planning_checks_text = json.dumps(city_context.get("evaluable_planning_checks") or [], ensure_ascii=False)
    if LOCAL_TRANSPORT_SUPPORT_TEXT_PATTERN.search(planning_checks_text):
        return None

    if re.search(r"avoid (?:detours|backtracking)|save travel time|nearby", signal_text, flags=re.IGNORECASE) and _hard_constraints_support_nearby_transfer_reason(meta):
        return None

    return "local transport preference appears in query without supporting profile, city context, or nearby/closest constraint"


def _query_has_people_number(query: str, people_number: int) -> bool:
    lower = query.lower()
    if re.search(rf"\b{people_number}\s+(traveler|travelers|people|adults?|friends?|person|persons?)\b", lower):
        return True
    if people_number == 1 and re.search(r"\b(solo|alone|by myself|myself|one traveler|1 traveler)\b", lower):
        return True
    if people_number == 2 and re.search(r"\b(two of us|2 of us|couple|two travelers|2 travelers)\b", lower):
        return True
    return False


def _query_has_room_number(query: str, room_number: int) -> bool:
    if re.search(rf"\b{room_number}\s+(room|rooms)\b", query, flags=re.IGNORECASE):
        return True
    if room_number == 1 and re.search(r"\bone room\b|\ba single room\b", query, flags=re.IGNORECASE):
        return True
    return False


def _query_has_days(query: str, days: int) -> bool:
    if re.search(rf"\b{days}\s+days?\b", query, flags=re.IGNORECASE):
        return True
    return False


def query_passes_render_sanity(query: str, meta: dict[str, Any]) -> bool:
    if _query_local_transport_alignment_issue(query, meta):
        return False

    expected_days = int(meta.get("days") or 0)
    if expected_days <= 0:
        return True
    expected_nights = max(0, expected_days - 1)

    for match in re.finditer(r"\b([0-9]+)\s*days?\b", query, flags=re.IGNORECASE):
        if int(match.group(1)) != expected_days:
            return False

    for match in re.finditer(r"\b([0-9]+)\s*nights?\b", query, flags=re.IGNORECASE):
        if int(match.group(1)) != expected_nights:
            return False

    return True


def _number_visible_in_query(query: str, value: object) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return True
    forms = {raw}
    try:
        number = float(raw)
    except ValueError:
        number = None
    if number is not None:
        if number.is_integer():
            forms.add(str(int(number)))
        forms.add(f"{number:g}")
    return any(form and form in query for form in forms)


def _hour_surface_forms(hour: int) -> list[str]:
    if hour < 0:
        return []
    forms = []
    if hour == 0:
        forms.extend(["midnight", "12 am", "12:00 am", "00:00"])
    elif hour < 6:
        forms.extend([f"{hour} am", f"{hour}:00 am", f"{hour:02d}:00"])
    elif hour < 12:
        forms.extend([f"{hour} am", f"{hour}:00 am", f"{hour:02d}:00"])
    elif hour == 12:
        forms.extend(["noon", "12 pm", "12:00 pm"])
    elif hour < 18:
        forms.extend([f"{hour - 12} pm", f"{hour - 12}:00 pm", f"{hour:02d}:00"])
    else:
        forms.extend([f"{hour - 12} pm", f"{hour - 12}:00 pm", f"{hour:02d}:00"])
    return list(dict.fromkeys(forms))


def _hour_surface_forms_en(hour: int) -> list[str]:
    if hour < 0:
        return []
    forms = [f"{hour}:00", f"{hour}"]
    suffix_hour = hour if hour <= 12 else hour - 12
    if hour == 0:
        forms.extend(["midnight", "12 am", "12:00 am"])
    elif hour < 12:
        forms.extend([f"{hour} am", f"{hour}:00 am"])
    elif hour == 12:
        forms.extend(["noon", "12 pm", "12:00 pm"])
    else:
        forms.extend([f"{suffix_hour} pm", f"{suffix_hour}:00 pm"])
    return list(dict.fromkeys(forms))


def _time_range_visible(query: str, constraint: dict[str, Any]) -> bool:
    start_hour = safe_int(constraint.get("start_hour"), -1)
    end_hour = safe_int(constraint.get("end_hour"), -1)
    if start_hour < 0 or end_hour < 0:
        return False
    query_lower = query.lower()
    start_forms = set(_hour_surface_forms(start_hour))
    end_forms = set(_hour_surface_forms(end_hour))
    start_forms.update(_hour_surface_forms_en(start_hour))
    end_forms.update(_hour_surface_forms_en(end_hour))
    return (
        any(form and form.lower() in query_lower for form in start_forms)
        and any(form and form.lower() in query_lower for form in end_forms)
    )


def _seat_class_visible(query_lower: str, seat_class: str) -> bool:
    normalized = seat_class.strip().lower()
    if not normalized:
        return True
    aliases = {
        "business class": ("business class", "business cabin", "business seat"),
        "economy": ("economy", "economy class"),
        "first class": ("first class",),
        "first class seat": ("first class", "first-class", "first-class seat", "first-class seats"),
        "second class seat": ("second class", "second-class", "second-class seat", "second-class seats"),
        "special class seat": ("special class", "special-class", "special class seat", "special class seats"),
    }
    forms = aliases.get(normalized, (normalized,))
    return any(form and form.lower() in query_lower for form in forms)


def _attraction_type_visible(query_lower: str, label: str, raw_type: str = "") -> bool:
    text = " ".join([label, raw_type]).lower()
    forms = {label.lower(), raw_type.lower()}
    if "natural" in text or "scenic" in text:
        forms.update({"natural scenery", "nature", "scenic spot", "scenic spots"})
    if "art" in text or "gallery" in text:
        forms.update({"art", "art gallery", "art galleries", "art museum"})
    if "theme" in text or "amusement" in text:
        forms.update({"theme park", "theme parks", "amusement park"})
    if "memorial" in text:
        forms.update({"memorial", "memorials"})
    if "leisure" in text:
        forms.update({"leisure", "leisurely", "leisure experience", "leisure experiences", "leisurely attractions"})
    if "park" in text:
        forms.update({"park", "parks"})
    return any(form and form in query_lower for form in forms)


def _constraint_visible_enough(query: str, payload: dict[str, Any]) -> bool:
    key = str(payload.get("constraint_key", ""))
    visible_hint = str(payload.get("visible_hint", ""))
    visible_hint_clean = visible_hint.strip().rstrip(".;, ")
    query_lower = query.lower()
    constraint = payload.get("constraint", {})
    if visible_hint_clean and visible_hint_clean.lower() in query_lower:
        return True
    if key.startswith("flight_"):
        seat_class = str(constraint.get("seat_class") or "").strip()
        if seat_class and not _seat_class_visible(query_lower, seat_class):
            return False
        if "cheapest" in key:
            return ("direct" in query_lower or "nonstop" in query_lower) and any(word in query_lower for word in ("cheapest", "lowest", "least expensive", "lowest-priced"))
        if "shortest_duration" in key:
            return ("direct" in query_lower or "nonstop" in query_lower) and any(word in query_lower for word in ("shortest", "duration", "quickest", "fastest"))
        if "arrival_time_range" in key:
            return any(word in query_lower for word in ("return", "arrive", "arrival", "land")) and _time_range_visible(query, constraint)
        if "departure_time_range" in key:
            return any(word in query_lower for word in ("outbound", "depart", "departure", "take off", "flight")) and _time_range_visible(query, constraint)
        return any(word in query_lower for word in ("flight", "fly", "plane", "air"))
    if key.startswith("train_"):
        has_train_mode = any(word in query_lower for word in ("train", "rail", "high-speed rail", "railway"))
        if not has_train_mode:
            return False
        seat_class = str(constraint.get("seat_class") or "").strip()
        if seat_class and not _seat_class_visible(query_lower, seat_class):
            return False
        if "cheapest" in key:
            return any(word in query_lower for word in ("cheapest", "lowest", "least expensive", "lowest-priced"))
        if "shortest_duration" in key:
            return any(word in query_lower for word in ("shortest", "quickest", "fastest", "duration", "travel time"))
        if "departure_time_range" in key:
            return "depart" in query_lower or "departure" in query_lower
        if "latest_arrival" in key:
            return "late" in query_lower or "arrival" in query_lower or "more time" in query_lower
        return True
    if "type_highest_rated" in key:
        label = str(constraint.get("attraction_type_label") or constraint.get("attraction_type") or "")
        raw_type = str(constraint.get("attraction_type") or "")
        return bool(
            (label or raw_type)
            and _attraction_type_visible(query_lower, label, raw_type)
            and any(
                word in query_lower
                for word in (
                    "highest-rated",
                    "highest rated",
                    "top-rated",
                    "top rated",
                    "best-rated",
                    "best reviewed",
                    "best reputation",
                    "top reputation",
                    "most highly rated",
                )
            )
        )
    if "top_rated_must_visit" in key:
        return bool(re.search(r"(top|highest[- ]rated).{0,20}(three|3)", query_lower))
    if "cheapest" in key:
        return (
            "cheapest" in query_lower
            or "cheap" in query_lower
            or "low-cost" in query_lower
            or "affordable" in query_lower
            or "lowest" in query_lower
            or "least expensive" in query_lower
            or "not too expensive" in query_lower
            or "budget-friendly" in query_lower
            or "cost-effective" in query_lower
        )
    if "highest_rated" in key or "top_rated" in key:
        return any(
            marker in query_lower
            for marker in (
                "highest-rated",
                "highest rated",
                "top-rated",
                    "top rated",
                    "top ratings",
                    "highly-rated",
                    "highly rated",
                "best-rated",
                "best reviewed",
                "best reputation",
                "top reputation",
                "highest local rating",
                "most highly rated",
            )
        )
    if "closest" in key:
        return any(
            marker in query_lower
            for marker in ("closest", "nearest", "near ", "nearby", "close to", "right next to", "save on travel", "travel short")
        )
    if "all_" in key:
        label = str(constraint.get("attraction_type_label") or constraint.get("attraction_type") or "")
        raw_type = str(constraint.get("attraction_type") or "")
        return (
            _attraction_type_visible(query_lower, label, raw_type)
            and any(marker in query_lower for marker in ("all", "every", "as many as possible", "several", "include them all"))
        )
    if "price_range" in key:
        return _number_visible_in_query(query, constraint.get("min_price")) and _number_visible_in_query(query, constraint.get("max_price"))
    if "service_required" in key or "specific_tag" in key:
        required = str(
            constraint.get("required_service_label")
            or constraint.get("required_tag_label")
            or ""
        )
        if required and required in query:
            return True
        required_lower = required.lower()
        if "robot" in required_lower and "robot" in query_lower:
            return True
        if "star" in required_lower and "star" in query_lower:
            return True
        return False
    if "specific_cuisine" in key:
        cuisine = str(constraint.get("cuisine_type") or "").strip()
        cuisine_forms = {cuisine}
        cuisine_forms.add(re.sub(r"\(.*?\)", "", cuisine).strip())
        cuisine_forms.update(part.strip() for part in re.split(r"[()/;]", cuisine) if part.strip())
        cuisine_lower = cuisine.lower()
        for suffix in (" cuisine", " restaurant"):
            if cuisine_lower.endswith(suffix):
                cuisine_forms.add(cuisine[: -len(suffix)].strip())
        return any(form and form.lower() in query_lower for form in cuisine_forms)
    if "must_eat_named" in key:
        return str(constraint.get("restaurant_name") or "").lower() in query_lower
    if "must_visit_named" in key:
        return all(_name_visible_in_query(query, str(name)) for name in constraint.get("attraction_names", []) or [])
    if "require_popular_hotspot" in key:
        return any(word in query_lower for word in ("popular", "representative", "landmark", "check-in", "hotspot", "iconic", "photo", "major", "well-known"))
    if "avoid_high_queue" in key:
        banned_names = [str(name) for name in constraint.get("banned_attraction_names", []) or []]
        return (
            any(word in query_lower for word in ("queue", "crowd", "avoid", "skip", "line", "long lines"))
            and (not banned_names or any(_name_visible_in_query(query, name) for name in banned_names[:3]))
        )
    if "newest_decoration" in key:
        return str(constraint.get("year_threshold") or constraint.get("decoration_time") or "") in query
    return True


def _exactish_reference_visible(query: str, value: object) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return True
    return raw in query


def _loose_reference_visible(query: str, value: object) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return True
    if _exactish_reference_visible(query, raw):
        return True
    if any(form in query for form in _name_surface_forms(raw)):
        return True
    normalized_query = re.sub(r"[\s\"'()\-_:./]", "", query)
    normalized_raw = re.sub(r"[\s\"'()\-_:./]", "", raw)
    if normalized_raw and normalized_raw in normalized_query:
        return True
    parts = [part for part in re.split(r"[\-_:./()]", raw) if len(part.strip()) >= 2]
    return bool(parts) and all(part.strip() in query for part in parts)


def _name_surface_forms(name: str) -> list[str]:
    cleaned = str(name or "").strip()
    forms = [cleaned] if cleaned else []
    without_parenthetical = re.sub(r"\([^)]*\)", "", cleaned).strip()
    if without_parenthetical and without_parenthetical not in forms:
        forms.append(without_parenthetical)
    tail = re.sub(r"^.*\)", "", cleaned).strip()
    if tail and tail not in forms:
        forms.append(tail)
    for suffix in (" scenic area", " museum", " church", " park"):
        if cleaned.lower().endswith(suffix) and len(cleaned) > len(suffix) + 1:
            short = cleaned[: -len(suffix)].strip()
            if short and short not in forms:
                forms.append(short)
    return [form for form in forms if len(form) >= 2]


def _name_visible_in_query(query: str, name: str) -> bool:
    return any(form in query for form in _name_surface_forms(name))
