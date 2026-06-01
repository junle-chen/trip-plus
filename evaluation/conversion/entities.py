"""Sample-database entity names used to clean converted plans."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def _add_clean_name(names: set[str], value: Any) -> None:
    cleaned = re.sub(r"\s+", " ", str(value or "").strip())
    if cleaned:
        names.add(cleaned)


def _sample_parent_id(sample_id: Any) -> str:
    return re.sub(r"_turn_\d+$", "", str(sample_id or "").strip())


def _candidate_sample_ids(
    sample_id: Any, query_file: Optional[Path] = None
) -> List[str]:
    candidates: List[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if not text:
            return
        if text.startswith("id_"):
            text = text[3:]
        if text and text not in candidates:
            candidates.append(text)

    parent_id = _sample_parent_id(sample_id)
    add(sample_id)
    add(parent_id)
    if parent_id.startswith("mt_"):
        add(parent_id[3:])

    for value in list(candidates):
        match = re.fullmatch(r"(?:mt_)?single_(\d+)", value)
        if match:
            add(match.group(1))

    if query_file is not None and Path(query_file).exists():
        try:
            records = json.loads(Path(query_file).read_text(encoding="utf-8"))
        except Exception:
            records = []
        if isinstance(records, list):
            normalized = {item for item in candidates}
            for record in records:
                if not isinstance(record, dict):
                    continue
                record_id = str(record.get("id") or "").strip()
                base_query_id = str(record.get("base_query_id") or "").strip()
                if (
                    record_id in normalized
                    or record_id.replace("id_", "") in normalized
                ):
                    add(base_query_id)
                if parent_id and record_id == parent_id:
                    add(base_query_id)

    for value in list(candidates):
        match = re.fullmatch(r"(?:mt_)?single_(\d+)", value)
        if match:
            add(match.group(1))
    return candidates


def _existing_sample_db_candidates(
    sample_id: Any,
    database_dir: Optional[Path],
    language: str,
    query_file: Optional[Path] = None,
) -> List[Path]:
    if database_dir is None:
        return []
    root = Path(database_dir)
    candidate_ids = _candidate_sample_ids(sample_id, query_file)
    roots = [
        root,
        root / "sample" / language,
        root.parent / "sample" / language,
        root / f"database_{language}",
        root.parent / f"database_{language}",
    ]
    candidates: List[Path] = []
    for candidate_root in roots:
        for candidate_id in candidate_ids:
            names = [candidate_id]
            if not candidate_id.startswith("id_"):
                names.append(f"id_{candidate_id}")
            for name in names:
                path = candidate_root / name
                if path not in candidates:
                    candidates.append(path)
    if (root / "restaurants" / "restaurants.csv").exists() or (
        root / "locations" / "locations_coords.csv"
    ).exists():
        candidates.insert(0, root)
    return candidates


def _exact_entity_names_from_sample_db(db_path: Path) -> set[str]:
    names: set[str] = set()
    field_specs = (
        ("hotels/hotels.csv", ("name", "hotel_name")),
        ("attractions/attractions.csv", ("attraction_name", "name")),
        ("restaurants/restaurants.csv", ("restaurant_name", "name")),
        ("locations/locations_coords.csv", ("poi_name", "name")),
        ("locations/location_aliases.csv", ("canonical_name", "alias")),
        (
            "flights/flights.csv",
            ("dep_station_name", "arr_station_name", "origin_city", "destination_city"),
        ),
        (
            "trains/trains.csv",
            ("dep_station_name", "arr_station_name", "origin_city", "destination_city"),
        ),
    )
    for rel_path, fields in field_specs:
        for row in _read_csv_dicts(db_path / rel_path):
            for field in fields:
                _add_clean_name(names, row.get(field))
    return names


def _load_exact_entity_names_for_sample(
    sample_id: Any,
    database_dir: Optional[Path] = None,
    language: str = "en",
    query_file: Optional[Path] = None,
) -> set[str]:
    for candidate in _existing_sample_db_candidates(
        sample_id, database_dir, language, query_file
    ):
        if candidate.exists():
            names = _exact_entity_names_from_sample_db(candidate)
            if names:
                return names
    return set()


