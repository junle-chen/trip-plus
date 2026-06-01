"""
Hotel Query Tool - Query hotel information (English-only)
"""
import os
import re
from typing import Dict, Optional, Union

from .base_travel_tool import BaseTravelTool, register_tool


@register_tool('query_hotel_info')
class HotelQueryTool(BaseTravelTool):
    """Tool for querying hotel information (English-only)"""
    
    # English field mappings
    LANG_FIELDS = {
        'en': {
            'db_not_loaded': "Database not loaded",
            'not_found': lambda dest: f"No hotel information found in {dest}; please check parameters or reduce constraints",
        }
    }
    
    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.database_path = cfg.get('database_path') if cfg else None
        
        # Get English fields
        self.fields = self.LANG_FIELDS.get(self.language, self.LANG_FIELDS['en'])
        
        if self.database_path and os.path.exists(self.database_path):
            self.data = self.load_csv_database(self.database_path)
        else:
            self.data = None

    def _normalize_city_name(self, value: object) -> str:
        text = str(value or '').strip()
        if not text:
            return ''
        text = re.sub(r"[（(].*?[)）]", "", text).strip()
        return re.sub(r"\s+city$", "", text, flags=re.IGNORECASE)

    def _compact_hotel_name(self, value: object) -> str:
        text = str(value or '').strip().lower()
        return (
            text
            .replace(" ", "")
            .replace("　", "")
            .replace("（", "(")
            .replace("）", ")")
        )

    def _to_float(self, value: object, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return default if parsed != parsed else parsed

    def _merge_service_lists(self, rows) -> list[str]:
        services: list[str] = []
        seen = set()
        for _, row in rows.iterrows():
            services_field = row.get('services', None)
            if not isinstance(services_field, str) or not services_field.strip():
                continue
            for item in services_field.split(';'):
                item = item.strip()
                if item and item not in seen:
                    seen.add(item)
                    services.append(item)
        return services

    def _normalize_service_value(self, value: object) -> str:
        text = str(value or '').strip().lower()
        if not text:
            return ''
        text = text.replace('_', ' ')
        text = re.sub(r"\b(and|service|services|hotel)\b", "", text)
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)

    def _service_matches(self, services_field: object, required_service: object) -> bool:
        required = self._normalize_service_value(required_service)
        if not required:
            return True
        if not isinstance(services_field, str):
            return False
        for service in services_field.split(';'):
            candidate = self._normalize_service_value(service)
            if candidate and (candidate == required or candidate in required or required in candidate):
                return True
        return False
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Execute hotel query
        
        Args:
            params: Query parameters containing destination and optional filters:
                   hotelName, hotelStar, hotelBrands, minPrice, maxPrice
            
        Returns:
            JSON string of query results
        """
        params = self._verify_json_format_args(params)
        
        destination = params.get('destination')
        hotel_star = params.get('hotelStar', '')
        hotel_brands = params.get('hotelBrands', '')
        hotel_name = (
            params.get('hotelName')
            or params.get('hotel_name')
            or params.get('name')
            or ''
        )
        min_price = self._to_float(
            params.get('minPrice', params.get('priceMin', params.get('min_price', ''))),
            default=None,
        )
        max_price = self._to_float(
            params.get('maxPrice', params.get('priceMax', params.get('max_price', ''))),
            default=None,
        )
        required_service = (
            params.get('requiredService')
            or params.get('required_service')
            or params.get('service')
            or ''
        )
        sort_by = str(params.get('sortBy') or params.get('sort_by') or '').strip().lower()
        
        if self.data is None:
            return self.fields['db_not_loaded']

        # Filter by destination city when the dataset contains city metadata.
        query_result = self.data
        if destination and 'city' in query_result.columns:
            normalized_destination = self._normalize_city_name(destination)
            if normalized_destination:
                normalized_cities = query_result['city'].astype(str).map(self._normalize_city_name)
                city_filtered = query_result[normalized_cities == normalized_destination]
                if not city_filtered.empty:
                    query_result = city_filtered

        # Filter by optional parameters
        if hotel_name and 'name' in query_result.columns:
            target_name = self._compact_hotel_name(hotel_name)
            normalized_names = query_result['name'].astype(str).map(self._compact_hotel_name)
            exact_name = query_result[normalized_names == target_name]
            if not exact_name.empty:
                query_result = exact_name
            else:
                substring_name = query_result[
                    normalized_names.map(lambda item: bool(item) and (target_name in item or item in target_name))
                ]
                query_result = substring_name if not substring_name.empty else query_result.iloc[0:0]
        if hotel_star:
            query_result = query_result[query_result['hotel_star'] == hotel_star]
        if hotel_brands:
            query_result = query_result[query_result['brand'] == hotel_brands]
        if required_service and 'services' in query_result.columns:
            query_result = query_result[
                query_result['services'].apply(lambda value: self._service_matches(value, required_service))
            ]
        if (min_price is not None or max_price is not None) and 'price' in query_result.columns:
            numeric_price = query_result['price'].apply(lambda value: self._to_float(value, default=float('nan')))
            query_result = query_result[numeric_price == numeric_price]
            numeric_price = query_result['price'].apply(lambda value: self._to_float(value, default=float('nan')))
            if min_price is not None:
                query_result = query_result[numeric_price >= min_price]
                numeric_price = query_result['price'].apply(lambda value: self._to_float(value, default=float('nan')))
            if max_price is not None:
                query_result = query_result[numeric_price <= max_price]
        
        if query_result.empty:
            return self.fields['not_found'](destination)
        
        def is_nan(v):
            try:
                return v != v
            except Exception:
                return False

        def to_str(v: object) -> str:
            if v is None:
                return ''
            if is_nan(v):
                return ''
            return str(v)

        # Collapse city-database variants of the same hotel into one record.
        results = []
        if 'name' in query_result.columns:
            grouped_rows = [
                (_, group) for _, group in query_result.groupby('name', sort=False)
            ]
        else:
            grouped_rows = [(None, query_result)]

        def group_sort_key(item):
            _, group = item
            best_score = max(self._to_float(v, -1.0) for v in group.get('score', []))
            best_decoration = max(self._to_float(v, -1.0) for v in group.get('decoration_time', []))
            prices = [self._to_float(v, float('inf')) for v in group.get('price', [])]
            valid_prices = [price for price in prices if price == price and price > 0]
            cheapest_price = min(valid_prices) if valid_prices else float('inf')
            best_price = max([price for price in prices if price == price] or [-1.0])
            name = str(group.iloc[0].get('name', ''))
            if sort_by in {"price", "cheapest", "lowest_price"}:
                return (cheapest_price, -best_score, -best_decoration, name)
            if sort_by in {"rating", "score", "highest_rated"}:
                return (-best_score, cheapest_price, -best_decoration, name)
            if sort_by in {"decoration", "newest", "newest_decoration"}:
                return (-best_decoration, -best_score, cheapest_price, name)
            return (-best_score, -best_decoration, -best_price, name)

        for _, rows in sorted(grouped_rows, key=group_sort_key):
            if sort_by in {"price", "cheapest", "lowest_price"}:
                row = rows.sort_values(
                    by=['price', 'score', 'decoration_time', 'name'],
                    ascending=[True, False, False, True],
                    na_position='last',
                ).iloc[0]
            elif sort_by in {"rating", "score", "highest_rated"}:
                row = rows.sort_values(
                    by=['score', 'price', 'decoration_time', 'name'],
                    ascending=[False, True, False, True],
                    na_position='last',
                ).iloc[0]
            elif sort_by in {"decoration", "newest", "newest_decoration"}:
                row = rows.sort_values(
                    by=['decoration_time', 'score', 'price', 'name'],
                    ascending=[False, False, True, True],
                    na_position='last',
                ).iloc[0]
            else:
                row = rows.sort_values(
                    by=['score', 'decoration_time', 'price', 'name'],
                    ascending=[False, False, False, True],
                    na_position='last',
                ).iloc[0]
            raw_price = row.get('price', '')
            price_text = to_str(raw_price)
            price_missing = price_text.strip() == ''
            result = {
                "name": to_str(row.get('name', '')),
                "address": to_str(row.get('address', '')),
                "latitude": to_str(row.get('latitude', '')),
                "longitude": to_str(row.get('longitude', '')),
                "decorationTime": to_str(row.get('decoration_time', '')),
                "hotelStar": to_str(row.get('hotel_star', '')),
                "price": price_text,
                "score": to_str(row.get('score', '')),
                "brand": to_str(row.get('brand', '')),
            }
            if price_missing:
                result["price_missing"] = True

            if 'services' in rows.columns:
                services_list = self._merge_service_lists(rows)
                if services_list:
                    result['services'] = services_list

            results.append(result)

        return self.format_result_as_json(results)
