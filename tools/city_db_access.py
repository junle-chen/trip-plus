from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _slugify_city_name(city_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", city_name.lower()).strip("_")
    return slug or "unknown_city"


def resolve_city_db_root(cfg: dict[str, Any] | None) -> Path | None:
    cfg = cfg or {}

    direct_root = str(cfg.get("city_db_root", "")).strip()
    if direct_root:
        path = Path(direct_root)
        if (path / "city_index.json").exists():
            return path

    sample_db_path = str(cfg.get("sample_db_path", "")).strip()
    if not sample_db_path:
        return None

    meta_path = Path(sample_db_path) / ".build_meta.json"
    if not meta_path.exists():
        return None

    try:
        meta = _read_json(meta_path)
    except Exception:
        return None

    city_db_root = str(meta.get("city_db_root", "")).strip()
    if not city_db_root:
        return None

    path = Path(city_db_root)
    return path if (path / "city_index.json").exists() else None


def load_city_index(city_db_root: Path) -> dict[str, dict[str, Any]]:
    raw = _read_json(city_db_root / "city_index.json")
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid city index: {city_db_root / 'city_index.json'}")
    return raw


def resolve_city_folder(city_db_root: Path, city_name: str) -> tuple[str, dict[str, Any]] | tuple[None, None]:
    target = str(city_name or "").strip()
    if not target:
        return None, None

    city_index = load_city_index(city_db_root)
    normalized = target.lower()
    slug = _slugify_city_name(target)

    for key, info in city_index.items():
        if not isinstance(info, dict):
            continue
        folder_name = str(info.get("folder_name", key)).strip()
        indexed_city_name = str(info.get("city_name", "")).strip()
        exact_candidates = {
            key.strip(),
            folder_name,
            indexed_city_name,
            key.strip().lower(),
            folder_name.lower(),
            indexed_city_name.lower(),
        }
        slug_candidates = {
            _slugify_city_name(key.strip()),
            _slugify_city_name(folder_name),
            _slugify_city_name(indexed_city_name),
        }
        if target in exact_candidates or normalized in exact_candidates:
            return folder_name or key, info
        if slug != "unknown_city" and slug in slug_candidates:
            return folder_name or key, info

    return None, None


def load_city_locations(city_db_root: Path, city_name: str) -> list[dict[str, str]]:
    folder_name, _ = resolve_city_folder(city_db_root, city_name)
    if not folder_name:
        return []
    csv_path = city_db_root / folder_name / "locations" / "locations_coords.csv"
    if not csv_path.exists():
        return []
    return _read_csv_rows(csv_path)


def load_city_location_entities(city_db_root: Path, city_name: str) -> list[dict[str, str]]:
    """Return canonical city entities as location-like rows.

    ``location_entities.csv`` is the canonical lookup layer built from
    locations, attractions, hotels, restaurants and transit hubs. Tool lookup
    still consumes the older ``poi_name`` shape, so normalize the schema here.
    """
    folder_name, _ = resolve_city_folder(city_db_root, city_name)
    if not folder_name:
        return []
    csv_path = city_db_root / folder_name / "locations" / "location_entities.csv"
    if not csv_path.exists():
        return []

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in _read_csv_rows(csv_path):
        name = str(row.get("canonical_name", "")).strip()
        lat = str(row.get("latitude", "")).strip()
        lon = str(row.get("longitude", "")).strip()
        if not name or not lat or not lon or name in seen:
            continue
        seen.add(name)
        rows.append(
            {
                "poi_name": name,
                "latitude": lat,
                "longitude": lon,
                "address": str(row.get("address", "")).strip(),
                "poi_type": str(row.get("entity_type", "")).strip(),
                "source": str(row.get("source", "")).strip() or "location_entities",
            }
        )
    return rows


def load_city_location_aliases(city_db_root: Path, city_name: str) -> dict[str, str]:
    folder_name, _ = resolve_city_folder(city_db_root, city_name)
    if not folder_name:
        return {}
    csv_path = city_db_root / folder_name / "locations" / "location_aliases.csv"
    if not csv_path.exists():
        return {}
    aliases: dict[str, str] = {}
    for row in _read_csv_rows(csv_path):
        alias = str(row.get("alias", "")).strip()
        canonical_name = str(row.get("canonical_name", "")).strip()
        if alias and canonical_name and alias != canonical_name:
            aliases.setdefault(alias, canonical_name)
            compact_alias = alias.replace(" ", "").replace("　", "").replace("（", "(").replace("）", ")")
            if compact_alias:
                aliases.setdefault(compact_alias, canonical_name)
    return aliases


def load_city_transportation(city_db_root: Path, city_name: str) -> list[dict[str, str]]:
    folder_name, _ = resolve_city_folder(city_db_root, city_name)
    if not folder_name:
        return []
    csv_path = city_db_root / folder_name / "transportation" / "distance_matrix.csv"
    if not csv_path.exists():
        return []
    return _read_csv_rows(csv_path)


def _candidate_city_db_roots(city_db_root: Path) -> list[Path]:
    candidates = [city_db_root]
    parent = city_db_root.parent
    for sibling_name in ("en",):
        sibling = parent / sibling_name
        if sibling != city_db_root and (sibling / "city_index.json").exists():
            candidates.append(sibling)
    return candidates


def _resolve_city_asset_json_path(city_db_root: Path, city_name: str, relative_path: str) -> tuple[Path | None, dict[str, Any] | None]:
    for candidate_root in _candidate_city_db_roots(city_db_root):
        folder_name, info = resolve_city_folder(candidate_root, city_name)
        if not folder_name:
            continue
        path = candidate_root / folder_name / relative_path
        if path.exists():
            return path, info
    return None, None


def _resolve_city_asset_csv_path(city_db_root: Path, city_name: str, relative_path: str) -> Path | None:
    for candidate_root in _candidate_city_db_roots(city_db_root):
        folder_name, _ = resolve_city_folder(candidate_root, city_name)
        if not folder_name:
            continue
        path = candidate_root / folder_name / relative_path
        if path.exists():
            return path
    return None


def load_city_subway(city_db_root: Path, city_name: str) -> dict[str, Any] | None:
    json_path, info = _resolve_city_asset_json_path(city_db_root, city_name, "subway/subway.json")
    if json_path is None:
        return None
    payload = _read_json(json_path)
    if isinstance(payload, dict) and "city" not in payload:
        payload["city"] = str((info or {}).get("city_name", city_name)).strip() or city_name
    return payload if isinstance(payload, dict) else None


def load_city_subway_stations(city_db_root: Path, city_name: str) -> list[dict[str, str]]:
    """Return subway stations as location-like rows.

    This is mainly a deterministic fallback for transport hubs. Some airport
    and railway station coordinates are present in subway data but not in
    locations_coords.csv, while flight/train tools still return those hub names.
    """
    payload = load_city_subway(city_db_root, city_name)
    if not payload:
        return []

    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    if isinstance(payload.get("lines"), list):
        for line in payload["lines"]:
            line_name = str(line.get("name", "")).strip()
            for station in line.get("stations", []):
                name = str(station.get("name", "")).strip()
                position = str(station.get("position", "")).strip()
                lon = station.get("longitude")
                lat = station.get("latitude")
                if position and "," in position:
                    lon, lat = [part.strip() for part in position.split(",", 1)]
                if not name or lon in (None, "") or lat in (None, ""):
                    continue
                key = name
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "poi_name": name,
                        "latitude": str(lat),
                        "longitude": str(lon),
                        "address": "",
                        "poi_type": "subway_station",
                        "source": "city_subway",
                        "line": line_name,
                    }
                )
        return rows

    amap_lines = payload.get("l") or []
    for line in amap_lines:
        line_name = str(line.get("ln", "")).strip()
        for station in line.get("st", []):
            name = str(station.get("n", "")).strip()
            position = str(station.get("sl", "")).strip()
            if not name or "," not in position:
                continue
            lon, lat = [part.strip() for part in position.split(",", 1)]
            key = name
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "poi_name": name,
                    "latitude": str(lat),
                    "longitude": str(lon),
                    "address": "",
                    "poi_type": "subway_station",
                    "source": "city_subway",
                    "line": line_name,
                }
            )
    return rows


def load_city_weather(city_db_root: Path, city_name: str) -> list[dict[str, str]]:
    csv_path = _resolve_city_asset_csv_path(city_db_root, city_name, "weather/weather_daily.csv")
    if csv_path is None:
        return []
    return _read_csv_rows(csv_path)
