from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.sample_db_resolver import resolve_sample_database_path_with_query


TOOL_DATABASE_FILES = {
    "query_train_info": "trains/trains.csv",
    "query_flight_info": "flights/flights.csv",
    "query_hotel_info": "hotels/hotels.csv",
    "query_attraction_details": "attractions/attractions.csv",
    "recommend_attractions": "attractions/attractions.csv",
    "search_location": "locations/locations_coords.csv",
    "query_road_route_info": "transportation/distance_matrix.csv",
    "recommend_restaurants": "restaurants/restaurants.csv",
    "query_restaurant_details": "restaurants/restaurants.csv",
}


TOOL_ARGUMENT_FIELDS: Dict[str, List[Tuple[str, Tuple[str, ...]]]] = {
    "query_city_transport_plan": [
        ("city", ("city",)),
        ("origin_place", ("origin_place",)),
        ("destination_place", ("destination_place",)),
        ("traveler_preference", ("traveler_preference",)),
    ],
    "query_road_route_info": [("origin", ("origin",)), ("destination", ("destination",))],
    "query_attraction_details": [("attraction_name", ("attraction_name", "name"))],
    "recommend_attractions": [
        ("city", ("city",)),
        ("attraction_type", ("attraction_type",)),
    ],
    "recommend_restaurants": [
        ("nearby_attraction_name", ("nearby_attraction_name", "attraction_name", "place_name")),
        ("sort_by", ("sort_by",)),
    ],
    "query_restaurant_details": [("restaurant_name", ("restaurant_name", "name"))],
    "search_location": [("place_name", ("place_name", "query", "name", "location"))],
    "query_hotel_info": [
        ("destination", ("destination",)),
        ("hotelStar", ("hotelStar",)),
        ("hotelName", ("hotelName", "hotel_name", "name")),
        ("hotelBrands", ("hotelBrands",)),
        ("minPrice", ("minPrice", "priceMin", "min_price")),
        ("maxPrice", ("maxPrice", "priceMax", "max_price")),
        ("requiredService", ("requiredService", "required_service", "service")),
        ("sortBy", ("sortBy", "sort_by")),
    ],
    "query_flight_info": [
        ("origin", ("origin",)),
        ("destination", ("destination",)),
        ("depDate", ("depDate",)),
        ("seatClassName", ("seatClassName",)),
        ("directOnly", ("directOnly",)),
        ("sortBy", ("sortBy",)),
        ("depTimeStart", ("depTimeStart", "departureTimeStart")),
        ("depTimeEnd", ("depTimeEnd", "departureTimeEnd")),
        ("arrTimeStart", ("arrTimeStart", "arrivalTimeStart")),
        ("arrTimeEnd", ("arrTimeEnd", "arrivalTimeEnd")),
    ],
    "query_train_info": [
        ("origin", ("origin",)),
        ("destination", ("destination",)),
        ("depDate", ("depDate",)),
        ("seatClassName", ("seatClassName",)),
        ("trainType", ("trainType",)),
        ("directOnly", ("directOnly",)),
        ("sortBy", ("sortBy",)),
        ("depTimeStart", ("depTimeStart", "departureTimeStart")),
        ("depTimeEnd", ("depTimeEnd", "departureTimeEnd")),
        ("arrTimeStart", ("arrTimeStart", "arrivalTimeStart")),
        ("arrTimeEnd", ("arrTimeEnd", "arrivalTimeEnd")),
    ],
    "query_city_weather": [("city", ("city",)), ("date", ("date",))],
}

TOOL_RESULT_FIELDS: Dict[str, List[str]] = {
    "query_city_transport_plan": [
        "city", "origin_place", "destination_place", "recommended_mode",
        "direct_distance_meters", "estimated_duration_minutes", "estimated_cost",
        "currency", "traveler_preference", "note",
    ],
    "query_road_route_info": ["origin", "destination", "distance_in_meters", "duration_in_minutes", "cost", "source"],
    "query_hotel_info": [
        "name", "address", "hotelStar", "price", "score", "brand",
        "decorationTime", "services", "price_missing",
    ],
    "recommend_restaurants": [
        "name", "price_per_person", "cuisine", "opening_time", "closing_time",
        "nearby_attraction_name", "rating", "distance_meters", "matched_nearby_attraction_name",
        "nearby_attraction_match_type", "tags", "price_missing",
    ],
    "query_restaurant_details": [
        "name", "restaurant_name", "price_per_person", "cuisine", "opening_time",
        "closing_time", "nearby_attraction_name", "rating", "tags", "price_missing", "message",
    ],
    "search_location": ["place_name", "latitude", "longitude", "matched_place_name", "matched_city"],
    "query_city_weather": [
        "city", "date", "requested_date", "date_fallback", "fallback_reason",
        "weather_code", "condition", "weather_condition", "temperature_min_c",
        "temperature_max_c", "precipitation_mm", "precipitation_hours",
        "travel_advisory",
    ],
}

ROUTE_DETAIL_FIELDS = [
    "distance_meters", "estimated_mode", "start_station", "end_station",
    "station_path", "line_path", "edge_line_path", "station_hops",
    "walk_start_meters", "walk_end_meters", "metro_distance_meters",
    "path_distance_meters", "pricing_rule", "reason", "fallback_reason",
]
TRANSPORT_SEGMENT_FIELDS = [
    "depCityName", "depStationName", "depDateTime", "arrCityName", "arrStationName",
    "arrDateTime", "duration", "marketingTransportName", "marketingTransportNo",
    "seatClassName", "price",
]
ATTRACTION_RESULT_PREFIXES = (
    "Attraction Name", "City", "Rating", "Opening Hours", "Open Status on Visit Day",
    "Minimum Visit Duration", "Maximum Visit Duration", "Ticket Price", "Attraction Type",
    "Popularity Tags", "Crowd Risk", "Queue Risk", "Peak Crowd Windows",
)


class TravelToolExecutor:
    """Loads travel tool classes lazily and executes tool calls."""

    def __init__(
        self,
        *,
        sample_id: object | None,
        database_base_path: Path,
        test_data_path: Path | None,
        language: str,
        sample_meta: Dict[str, Any],
    ) -> None:
        self.sample_id = sample_id
        self.database_base_path = database_base_path
        self.test_data_path = test_data_path
        self.language = language
        self.sample_meta = sample_meta
        self.tool_classes = self._load_tool_classes()
        self.tool_instances: Dict[str, Any] = {}

    def _load_tool_classes(self) -> Dict[str, Any]:
        classes: Dict[str, Any] = {}
        tools_dir = Path(__file__).resolve().parent.parent / "tools"
        sys.path.insert(0, str(tools_dir.parent))
        sys.path.insert(0, str(tools_dir))

        try:
            import tools  # noqa: F401
        except Exception as exc:
            raise RuntimeError(f"Cannot import tools package: {exc}") from exc

        try:
            tools_mod = importlib.import_module("tools.base_travel_tool")
            base_tool_cls = getattr(tools_mod, "BaseTravelTool", None)
        except Exception as exc:
            raise RuntimeError(f"Cannot import base_travel_tool: {exc}") from exc

        if base_tool_cls is None:
            raise RuntimeError("BaseTravelTool class not found in tools.base_travel_tool")

        for cls in base_tool_cls.__subclasses__():
            name = getattr(cls, "name", None)
            if name:
                classes[name] = cls
        if not classes:
            raise RuntimeError("No tool classes registered. Check dependencies and tool imports.")
        return classes

    def _build_config(self, tool_cls) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {"language": self.language}
        project_root = Path(__file__).resolve().parent.parent
        for city_db_root in (
            project_root / "database" / self.language,
            project_root / "database" / "database_by_city" / self.language,
        ):
            if (city_db_root / "city_index.json").exists():
                cfg["city_db_root"] = str(city_db_root)
                break

        candidate_cities: list[str] = []
        origin = str(self.sample_meta.get("org", "")).strip()
        if origin:
            candidate_cities.append(origin)
        for city in self.sample_meta.get("dest", []) or []:
            city_text = str(city).strip()
            if city_text and city_text not in candidate_cities:
                candidate_cities.append(city_text)
        if candidate_cities:
            cfg["candidate_cities"] = candidate_cities

        if self.sample_id is None:
            return cfg

        sample_db_path = resolve_sample_database_path_with_query(
            sample_id=str(self.sample_id),
            database_root=self.database_base_path,
            language=self.language,
            query_file=self.test_data_path,
        )
        cfg["sample_db_path"] = str(sample_db_path)

        relative_db_path = TOOL_DATABASE_FILES.get(getattr(tool_cls, "name", ""))
        if relative_db_path:
            db_path = sample_db_path / relative_db_path
            if db_path.exists():
                cfg["database_path"] = str(db_path)
        return cfg

    def _get_instance(self, name: str):
        inst = self.tool_instances.get(name)
        if inst is not None:
            return inst

        tool_cls = self.tool_classes.get(name)
        if tool_cls is None:
            return None

        inst = tool_cls(cfg=self._build_config(tool_cls))
        inst_name = getattr(inst, "name", None) or getattr(tool_cls, "name", None) or name
        self.tool_instances[inst_name] = inst
        return inst

    def call(self, name: str, arguments_json: str) -> str:
        try:
            inst = self._get_instance(name)
        except Exception as exc:
            return compact_json({"error": "tool_init_failed", "tool": name, "details": str(exc)})
        if not inst:
            return compact_json({"error": "tool_not_found", "tool": name, "details": f"tool '{name}' not found"})

        try:
            args = json.loads(arguments_json) if arguments_json else {}
        except Exception as exc:
            return compact_json({
                "error": "invalid_tool_arguments_json",
                "tool": name,
                "details": str(exc),
                "arguments": arguments_json,
            })
        if not isinstance(args, dict):
            return compact_json({
                "error": "invalid_tool_arguments_type",
                "tool": name,
                "details": f"tool arguments must be a JSON object, got {type(args).__name__}",
                "arguments": args,
            })

        try:
            result = inst.call(args)
        except Exception as exc:
            return compact_json({"error": "tool_call_failed", "tool": name, "details": str(exc)})
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


def _tool_call_extra_content(tool_call: Any) -> Optional[Any]:
    if isinstance(tool_call, dict):
        return tool_call.get("extra_content")
    extra_content = getattr(tool_call, "extra_content", None)
    if extra_content is not None:
        return extra_content
    model_extra = getattr(tool_call, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get("extra_content")
    return None


def detect_tool_calls(assistant_message: Any, openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tool_calls = getattr(assistant_message, "tool_calls", None)
    calls: List[Dict[str, Any]] = []
    if tool_calls:
        for tool_call in tool_calls:
            try:
                tool_call_id = tool_call.id or f"call_{uuid.uuid4().hex[:24]}"
                call = {
                    "id": tool_call_id,
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                }
                extra_content = _tool_call_extra_content(tool_call)
                if extra_content is not None:
                    call["extra_content"] = extra_content
                calls.append(call)
            except Exception:
                continue

    if calls:
        return calls

    content = getattr(assistant_message, "content", "") or ""
    return detect_text_tool_calls(content, openai_tools)


def detect_text_tool_calls(content: str, openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not content:
        return []

    calls: List[Dict[str, Any]] = []
    if "<|tool_call>" in content:
        pattern = re.compile(r"<\|tool_call\>\s*call:([A-Za-z_][\w]*)\s*\{(.*?)\}\s*<tool_call\|>", re.DOTALL)
        for match in pattern.finditer(content):
            arguments = parse_gemma_text_tool_args(match.group(2))
            if arguments is None:
                continue
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "name": match.group(1),
                "arguments": arguments,
            })
        if calls:
            return calls

    return detect_plain_json_tool_calls(content, known_tool_names(openai_tools))


def known_tool_names(openai_tools: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for tool in openai_tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return sorted(set(names), key=len, reverse=True)


def detect_plain_json_tool_calls(content: str, tool_names: List[str]) -> List[Dict[str, Any]]:
    if not content or not tool_names:
        return []

    name_pattern = "|".join(re.escape(name) for name in tool_names)
    pattern = re.compile(rf"(?<![A-Za-z0-9_])({name_pattern})(?![A-Za-z0-9_])")
    calls: List[Dict[str, Any]] = []
    search_pos = 0
    while search_pos < len(content):
        match = pattern.search(content, search_pos)
        if not match:
            break

        name = match.group(1)
        brace_pos = content.find("{", match.end())
        if brace_pos < 0:
            break

        between = content[match.end():brace_pos]
        if len(between) > 80 or re.search(r"[^\s:=(]", between):
            search_pos = match.end()
            continue

        raw_args, end_pos = extract_balanced_json_object(content, brace_pos)
        if raw_args is None:
            search_pos = match.end()
            continue
        try:
            parsed_args = json.loads(raw_args)
        except json.JSONDecodeError:
            search_pos = end_pos
            continue
        if not isinstance(parsed_args, dict):
            search_pos = end_pos
            continue

        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "name": name,
            "arguments": json.dumps(parsed_args, ensure_ascii=False),
        })
        search_pos = end_pos
    return calls


def extract_balanced_json_object(text: str, start_pos: int) -> Tuple[Optional[str], int]:
    if start_pos >= len(text) or text[start_pos] != "{":
        return None, start_pos

    depth = 0
    in_string = False
    escape = False
    for pos in range(start_pos, len(text)):
        char = text[pos]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_pos:pos + 1], pos + 1
    return None, len(text)


def parse_gemma_text_tool_args(raw_args: str) -> Optional[str]:
    args: Dict[str, Any] = {}
    for part in split_gemma_arg_parts(raw_args):
        if ":" not in part:
            return None
        key, value = part.split(":", 1)
        key = key.strip().strip('"').strip("'")
        if not key:
            return None
        args[key] = parse_gemma_text_value(value.strip())
    return json.dumps(args, ensure_ascii=False)


def split_gemma_arg_parts(raw_args: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    brace_depth = 0
    bracket_depth = 0
    quote_open = False
    i = 0
    while i < len(raw_args):
        if raw_args.startswith('<|"|>', i):
            quote_open = not quote_open
            current.append('<|"|>')
            i += len('<|"|>')
            continue
        char = raw_args[i]
        if not quote_open:
            if char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth = max(0, brace_depth - 1)
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth = max(0, bracket_depth - 1)
            elif char == "," and brace_depth == 0 and bracket_depth == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                i += 1
                continue
        current.append(char)
        i += 1
    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def parse_gemma_text_value(value: str) -> Any:
    if value.startswith('<|"|>') and value.endswith('<|"|>'):
        return value[len('<|"|>'):-len('<|"|>')]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value.strip('"').strip("'")


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def pick_fields(payload: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    return {
        key: payload[key]
        for key in fields
        if key in payload and payload[key] not in (None, "", [], {})
    }


def normalize_tool_arguments(name: str, arguments_json: str) -> str:
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except Exception:
        return arguments_json or "{}"
    if not isinstance(args, dict) or name not in TOOL_ARGUMENT_FIELDS:
        return compact_json(args)

    normalized: Dict[str, Any] = {}
    for out_key, aliases in TOOL_ARGUMENT_FIELDS[name]:
        for key in aliases:
            value = args.get(key)
            if value not in (None, "", [], {}):
                normalized[out_key] = value
                break
    return compact_json(normalized)


def normalize_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(call)
    normalized["arguments"] = normalize_tool_arguments(
        normalized["name"],
        normalized.get("arguments") or "{}",
    )
    return normalized


def tool_call_key(name: str, arguments_json: str) -> str:
    raw = f"{name}\n{normalize_tool_arguments(name, arguments_json)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def duplicate_tool_notice(name: str) -> str:
    return compact_json({"duplicate": True, "tool": name})


def _compact_transport_candidates(payload: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(payload, list):
        return None
    compact_routes = []
    for raw_item in payload[:6]:
        item = raw_item[0] if isinstance(raw_item, list) and raw_item else raw_item
        if not isinstance(item, dict):
            continue
        route = pick_fields(item, ["price", "segmentCount", "isDirect"])
        segments = []
        raw_segments = item.get("segments")
        if isinstance(raw_segments, list):
            segment_items = raw_segments
        else:
            segment_items = [
                item.get(f"Segment {idx}")
                for idx in range(1, int(item.get("segmentCount") or 1) + 1)
            ]
        for segment in segment_items:
            if isinstance(segment, dict):
                segments.append(pick_fields(segment, TRANSPORT_SEGMENT_FIELDS))
        if segments:
            route["segments"] = segments
        compact_routes.append(route)
    return compact_routes


def minimize_tool_result_for_context(tool_name: str, tool_result: str) -> str:
    if not isinstance(tool_result, str):
        return tool_result
    if tool_name == "query_attraction_details":
        lines = [
            line for line in tool_result.splitlines()
            if any(line.strip().startswith(prefix) for prefix in ATTRACTION_RESULT_PREFIXES)
        ]
        return "\n".join(lines) if lines else tool_result

    try:
        payload = json.loads(tool_result)
    except Exception:
        return tool_result
    if isinstance(payload, dict) and (payload.get("duplicate") is True or "error" in payload):
        return compact_json(payload)

    if tool_name in {"query_train_info", "query_flight_info"}:
        compact = _compact_transport_candidates(payload)
    else:
        fields = TOOL_RESULT_FIELDS.get(tool_name)
        items = payload if isinstance(payload, list) else [payload] if isinstance(payload, dict) else None
        compact = None
        if fields and items is not None:
            compact = [pick_fields(item, fields) for item in items[:6] if isinstance(item, dict)]
            if tool_name == "query_city_transport_plan" and isinstance(payload, dict):
                details = payload.get("details")
                if isinstance(details, dict) and compact:
                    detail = pick_fields(details, ROUTE_DETAIL_FIELDS)
                    if detail:
                        compact[0]["details"] = detail
            if isinstance(payload, dict):
                compact = compact[0] if compact else {}
    return compact_json(compact if compact is not None else payload)


def compact_tool_result_for_context(tool_name: str, tool_result: str, max_chars: int, language: str) -> str:
    if max_chars <= 0 or len(tool_result) <= max_chars:
        return tool_result
    notice = (
        f"\n\n[NOTICE] {tool_name} result was too long and was truncated from "
        f"{len(tool_result)} to {max_chars} characters for context safety. Plan from the visible "
        "tool evidence; if evidence is insufficient, use a more specific query instead of broad scanning.\n"
    )
    keep = max(0, max_chars - len(notice))
    return tool_result[:keep] + notice
