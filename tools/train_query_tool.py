"""
Train Query Tool - Query train ticket information (English-only)
"""
import os
import re
from datetime import datetime
from datetime import time as dt_time
from typing import Dict, Optional, Union

from .base_travel_tool import BaseTravelTool, register_tool


@register_tool('query_train_info')
class TrainQueryTool(BaseTravelTool):
    """Tool for querying train ticket information (English-only)"""
    
    # English field mappings
    LANG_FIELDS = {
        'en': {
            'segment': lambda idx: f"Segment {idx}",
            'sufficient': "Available",
            'no_info': "No information found",
            'not_found': lambda o, d, date: f"No train information found from {o} to {d} on {date}",
            'station_sep': ' Station',
        }
    }
    
    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.database_path = cfg.get('database_path') if cfg else None
        self.city_alias_to_city = {}
        
        # Get English fields
        self.fields = self.LANG_FIELDS.get(self.language, self.LANG_FIELDS['en'])
        
        if self.database_path and os.path.exists(self.database_path):
            self.data = self.load_csv_database(self.database_path)
            self.city_alias_to_city = self._build_city_alias_to_city()
        else:
            self.data = None

    def _slugify_token(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

    def _compact_token(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    def _normalize_seat_class(self, value: object) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        compact = self._compact_token(text)
        if "business" in compact:
            return "Business Seat"
        if "firstclass" in compact or compact in {"first", "firstseat"}:
            return "First Class Seat"
        if "secondclass" in compact or compact in {"second", "secondseat"}:
            return "Second Class Seat"
        if "hardseat" in compact:
            return "Hard Seat"
        if "softseat" in compact:
            return "Soft Seat"
        if "hardsleeper" in compact:
            return "Hard Sleeper"
        if "softsleeper" in compact:
            return "Soft Sleeper"
        if "noseat" in compact:
            return "No Seat"
        return text

    def _alias_variants(self, text: object) -> set[str]:
        alias_text = str(text or "").strip()
        if not alias_text:
            return set()

        variants = {alias_text}
        queue = [alias_text]
        replacement_pairs = (
            (" Railway Station", " Station"),
            (" railway station", " station"),
            (" Train Station", " Station"),
            (" train station", " station"),
        )
        removable_suffixes = (
            " City",
            " city",
            " Station",
            " station",
            " Railway Station",
            " railway station",
            " Train Station",
            " train station",
        )

        while queue:
            current = queue.pop()
            normalized_current = current.strip()
            if not normalized_current:
                continue

            for old, new in replacement_pairs:
                if normalized_current.endswith(old):
                    replaced = f"{normalized_current[:-len(old)]}{new}".strip()
                    if replaced and replaced not in variants:
                        variants.add(replaced)
                        queue.append(replaced)

            for suffix in removable_suffixes:
                if normalized_current.endswith(suffix):
                    stripped = normalized_current[:-len(suffix)].strip()
                    if stripped and stripped not in variants:
                        variants.add(stripped)
                        queue.append(stripped)

        enriched: set[str] = set()
        for variant in variants:
            candidate = variant.strip()
            if not candidate:
                continue
            enriched.add(candidate)
            slug = self._slugify_token(candidate)
            compact = self._compact_token(candidate)
            if slug:
                enriched.add(slug)
            if compact:
                enriched.add(compact)
        return enriched

    def _register_city_alias(self, alias_map: dict[str, str], alias: object, city: object) -> None:
        alias_text = str(alias or "").strip()
        city_text = str(city or "").strip()
        if not alias_text or not city_text:
            return
        for candidate in self._alias_variants(alias_text):
            candidate = str(candidate).strip().lower()
            if candidate:
                alias_map.setdefault(candidate, city_text)

    def _build_city_alias_to_city(self) -> dict[str, str]:
        alias_map: dict[str, str] = {}
        if self.data is None:
            return alias_map

        for row in self.data.itertuples():
            self._register_city_alias(alias_map, row.origin_city, row.origin_city)
            self._register_city_alias(alias_map, row.destination_city, row.destination_city)
            self._register_city_alias(alias_map, row.dep_station_name, row.origin_city)
            self._register_city_alias(alias_map, row.arr_station_name, row.destination_city)
        return alias_map

    def _normalize_city_query(self, value: object) -> str:
        text = str(value or "").strip()
        if not text:
            return text

        candidates = sorted(self._alias_variants(text), key=len, reverse=True)
        for candidate in candidates:
            normalized = self.city_alias_to_city.get(str(candidate).strip().lower())
            if normalized:
                return normalized
        return text

    def _to_bool(self, value: object) -> bool:
        if isinstance(value, bool):
            return value
        return str(value or "").strip().lower() in {"1", "true", "yes", "y", "direct"}

    def _parse_clock_time(self, value: object, *, is_end: bool = False) -> Optional[dt_time]:
        text = str(value or "").strip()
        if not text:
            return None
        hour: Optional[int] = None
        minute = 0
        match = re.search(r"(\d{1,2})(?::(\d{1,2}))?", text)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2) or 0)
        if hour is None or hour < 0 or hour > 24 or minute < 0 or minute > 59:
            return None
        if hour == 24:
            return dt_time(23, 59, 59) if is_end else dt_time(0, 0)
        return dt_time(hour, minute)

    def _time_in_window(self, timestamp: object, start: Optional[dt_time], end: Optional[dt_time]) -> bool:
        if start is None and end is None:
            return True
        try:
            current = datetime.fromisoformat(str(timestamp)).time()
        except Exception:
            return False
        if start is not None and end is not None and start > end:
            return current >= start or current <= end
        if start is not None and current < start:
            return False
        if end is not None and current > end:
            return False
        return True

    def _terminal_station_sets(self, origin: str, destination: str) -> tuple[set[str], set[str]]:
        if self.data is None:
            return set(), set()
        rows = self.data[
            (self.data['origin_city'] == origin) &
            (self.data['destination_city'] == destination)
        ]
        terminal_codes: set[str] = set()
        terminal_names: set[str] = set()
        for _route_idx, route_segments in rows.groupby('route_index'):
            max_segment_index = 0
            for row in route_segments.itertuples():
                max_segment_index = max(max_segment_index, int(row.segment_index))
            if max_segment_index <= 1:
                continue
            for row in route_segments.itertuples():
                if int(row.segment_index) != max_segment_index:
                    continue
                code = str(row.arr_station_code or "").strip()
                name = str(row.arr_station_name or "").strip()
                if code:
                    terminal_codes.add(code)
                if name:
                    terminal_names.add(name)
        return terminal_codes, terminal_names

    def _reaches_terminal_station(self, row, terminal_codes: set[str], terminal_names: set[str]) -> bool:
        if not terminal_codes and not terminal_names:
            return True
        code = str(row.arr_station_code or "").strip()
        name = str(row.arr_station_name or "").strip()
        return code in terminal_codes or name in terminal_names

    def _segment_identity(self, row) -> tuple:
        return (
            int(row.segment_index),
            row.dep_station_code,
            row.dep_station_name,
            row.arr_station_code,
            row.arr_station_name,
            row.dep_datetime,
            row.arr_datetime,
            row.train_no,
            row.train_type,
            row.seat_class,
            row.price,
        )

    def _build_route_candidates(self, route_segments):
        """Rebuild complete route chains from possibly duplicated rows."""
        deduped_rows = {}
        for row in route_segments.itertuples():
            key = self._segment_identity(row)
            if key not in deduped_rows:
                deduped_rows[key] = row

        rows = list(deduped_rows.values())
        if not rows:
            return []

        rows.sort(key=lambda row: (
            row.seat_class,
            int(row.segment_index),
            row.dep_datetime,
            row.arr_datetime,
            row.dep_station_name,
            row.arr_station_name,
            row.train_no,
        ))

        rows_by_segment = {}
        max_segment_index = 0
        for row in rows:
            segment_index = int(row.segment_index)
            max_segment_index = max(max_segment_index, segment_index)
            rows_by_segment.setdefault((row.seat_class, segment_index), []).append(row)

        route_candidates = []

        def same_station(prev_row, next_row) -> bool:
            prev_code = str(prev_row.arr_station_code or "").strip()
            next_code = str(next_row.dep_station_code or "").strip()
            if prev_code and next_code:
                return prev_code == next_code
            return str(prev_row.arr_station_name).strip() == str(next_row.dep_station_name).strip()

        def is_chronological(prev_row, next_row) -> bool:
            prev_arr = datetime.fromisoformat(str(prev_row.arr_datetime))
            next_dep = datetime.fromisoformat(str(next_row.dep_datetime))
            return next_dep >= prev_arr

        def build_from_chain(chain, seat_class, next_segment_index):
            if next_segment_index > max_segment_index:
                route_candidates.append(chain)
                return

            next_rows = rows_by_segment.get((seat_class, next_segment_index), [])
            matched = False
            for next_row in next_rows:
                if same_station(chain[-1], next_row) and is_chronological(chain[-1], next_row):
                    matched = True
                    build_from_chain(chain + [next_row], seat_class, next_segment_index + 1)

            if not matched and len(chain) == next_segment_index - 1:
                # No continuation: only keep if this route is naturally complete.
                if next_segment_index == 2 and max_segment_index == 1:
                    route_candidates.append(chain)

        for seat_class in sorted({row.seat_class for row in rows}):
            first_segments = rows_by_segment.get((seat_class, 1), [])
            if max_segment_index == 1:
                route_candidates.extend([[row] for row in first_segments])
                continue

            for first_row in first_segments:
                build_from_chain([first_row], seat_class, 2)

        if route_candidates:
            return route_candidates

        # Fallback for irregular data: return per-row direct options instead of
        # emitting duplicated pseudo-segments.
        return [[row] for row in rows if int(row.segment_index) == 1]

    def _route_price(self, chain) -> Optional[float]:
        """Return the complete per-person price for one reconstructed route."""
        total = 0.0
        for row in chain:
            try:
                price = float(row.price)
            except (TypeError, ValueError):
                return None
            if price <= 0:
                return None
            total += price
        return total
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Execute train ticket query
        
        Args:
            params: Query parameters containing origin, destination, depDate
            
        Returns:
            JSON string of query results
        """
        # Verify parameter format
        params = self._verify_json_format_args(params)
        
        origin = self._normalize_city_query(params.get('origin'))
        destination = self._normalize_city_query(params.get('destination'))
        dep_date = params.get('depDate')
        seat_class = self._normalize_seat_class(params.get('seatClassName', ''))
        direct_only = self._to_bool(params.get('directOnly'))
        sort_by = str(params.get('sortBy', '') or '').strip().lower()
        train_type = str(params.get('trainType', '') or '').strip()
        dep_time_start = self._parse_clock_time(params.get('depTimeStart') or params.get('departureTimeStart'))
        dep_time_end = self._parse_clock_time(params.get('depTimeEnd') or params.get('departureTimeEnd'), is_end=True)
        arr_time_start = self._parse_clock_time(params.get('arrTimeStart') or params.get('arrivalTimeStart'))
        arr_time_end = self._parse_clock_time(params.get('arrTimeEnd') or params.get('arrivalTimeEnd'), is_end=True)
        
        # If database not loaded
        if self.data is None:
            return self.fields['no_info']
        
        query_result = self.data[
            (self.data['origin_city'] == origin) &
            (self.data['destination_city'] == destination) &
            (self.data['dep_date'] == dep_date)
        ]
        if seat_class:
            query_result = query_result[
                query_result['seat_class'].apply(self._normalize_seat_class) == seat_class
            ]
        if train_type:
            query_result = query_result[
                query_result['train_type'].astype(str).str.strip().str.lower() == train_type.lower()
            ]

        if query_result.empty:
            return self.fields['not_found'](origin, destination, dep_date)

        terminal_codes, terminal_names = self._terminal_station_sets(origin, destination)
        
        # Build result grouped by route_index, then reconstruct actual chains.
        routes = []
        for route_idx in sorted(query_result['route_index'].unique()):
            route_segments = query_result[query_result['route_index'] == route_idx].sort_values('segment_index')
            route_max_segment_index = int(route_segments['segment_index'].max())
            station_name_by_code = {}
            for station_row in route_segments.itertuples():
                for code, name in (
                    (station_row.dep_station_code, station_row.dep_station_name),
                    (station_row.arr_station_code, station_row.arr_station_name),
                ):
                    code_text = str(code or "").strip()
                    name_text = str(name or "").strip()
                    if not code_text or not name_text:
                        continue
                    existing = station_name_by_code.get(code_text)
                    if existing is None or len(name_text) < len(existing):
                        station_name_by_code[code_text] = name_text

            for chain in self._build_route_candidates(route_segments):
                if direct_only and len(chain) != 1:
                    continue
                if (
                    len(chain) == 1
                    and route_max_segment_index > 1
                    and not self._reaches_terminal_station(chain[0], terminal_codes, terminal_names)
                ):
                    continue
                route_data = {}
                route_price = self._route_price(chain)
                display_price = route_price if route_price is not None else 0

                for idx, row in enumerate(chain, 1):
                    dep_station_name = station_name_by_code.get(str(row.dep_station_code).strip(), row.dep_station_name)
                    arr_station_name = station_name_by_code.get(str(row.arr_station_code).strip(), row.arr_station_name)
                    dep_city_name = row.origin_city if idx == 1 else dep_station_name
                    arr_city_name = row.destination_city if idx == len(chain) else arr_station_name

                    segment = {
                        self.fields['segment'](idx): {
                            "arrCityName": arr_city_name,
                            "arrStationCode": row.arr_station_code,
                            "arrStationName": arr_station_name,
                            "depCityName": dep_city_name,
                            "depStationCode": row.dep_station_code,
                            "depStationName": dep_station_name,
                            "duration": int(row.duration),
                            "arrDateTime": row.arr_datetime,
                            "depDateTime": row.dep_datetime,
                            "marketingTransportName": row.train_type,
                            "marketingTransportNo": row.train_no,
                            "seatClassName": row.seat_class,
                            "price": display_price
                        }
                    }
                    route_data.update(segment)

                route_data["price"] = display_price
                route_data["segmentCount"] = len(chain)
                route_data["isDirect"] = len(chain) == 1
                first_segment = route_data.get(self.fields['segment'](1)) or {}
                last_segment = route_data.get(self.fields['segment'](len(chain))) or {}
                if not self._time_in_window(first_segment.get("depDateTime"), dep_time_start, dep_time_end):
                    continue
                if not self._time_in_window(last_segment.get("arrDateTime"), arr_time_start, arr_time_end):
                    continue
                routes.append([route_data])

        def route_item(raw_item):
            return raw_item[0] if isinstance(raw_item, list) and raw_item else raw_item

        if sort_by in {"price", "cheapest"}:
            def price_key(raw_item):
                item = route_item(raw_item)
                try:
                    price = float(item.get("price") or 0)
                except Exception:
                    price = 0
                return price if price > 0 else float("inf")

            routes.sort(
                key=lambda raw_item: (
                    price_key(raw_item),
                    int((route_item(raw_item) or {}).get("segmentCount") or 99),
                    str(((route_item(raw_item) or {}).get(self.fields['segment'](1)) or {}).get("depDateTime", "")),
                )
            )
        elif sort_by in {"departure_time", "deptime", "earliest"}:
            routes.sort(
                key=lambda raw_item: str(((route_item(raw_item) or {}).get(self.fields['segment'](1)) or {}).get("depDateTime", ""))
            )
        elif sort_by in {"arrival_time", "arrivaltime", "earliest_arrival"}:
            routes.sort(
                key=lambda raw_item: str(
                    ((route_item(raw_item) or {}).get(
                        self.fields['segment'](int((route_item(raw_item) or {}).get("segmentCount") or 1))
                    ) or {}).get("arrDateTime", "")
                )
            )
        elif sort_by in {"latest_arrival", "arrival_time_desc"}:
            routes.sort(
                key=lambda raw_item: str(
                    ((route_item(raw_item) or {}).get(
                        self.fields['segment'](int((route_item(raw_item) or {}).get("segmentCount") or 1))
                    ) or {}).get("arrDateTime", "")
                ),
                reverse=True,
            )
        elif sort_by in {"duration", "shortest_duration", "shortest"}:
            def duration_key(raw_item):
                segment = ((route_item(raw_item) or {}).get(self.fields['segment'](1)) or {})
                try:
                    return int(segment.get("duration") or 999999)
                except Exception:
                    return 999999

            routes.sort(
                key=lambda raw_item: (
                    duration_key(raw_item),
                    int((route_item(raw_item) or {}).get("segmentCount") or 99),
                    str(((route_item(raw_item) or {}).get(self.fields['segment'](1)) or {}).get("depDateTime", "")),
                )
            )

        if direct_only and not routes:
            return self.fields['not_found'](origin, destination, dep_date)
        
        return self.format_result_as_json(routes)
