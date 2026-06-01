"""Output helpers for flat and grouped multi-turn query files."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from query_generation.common import BASE_DIR, write_json
from query_generation.initial_query.config import INTERACTION_ARCHETYPE_WEIGHTS
from query_generation.multiturn_query.config import INTERACTION_DESCRIPTIONS, INTERACTION_LABELS


def write_grouped(records: list[dict[str, Any]], query_root: Path) -> None:
    query_root = query_root if query_root.is_absolute() else (BASE_DIR / query_root)
    query_root.mkdir(parents=True, exist_ok=True)
    items_dir = query_root / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in items_dir.glob("*.json"):
        stale_file.unlink()

    grouped: dict[str, list[dict[str, Any]]] = {label: [] for label, _ in INTERACTION_ARCHETYPE_WEIGHTS}
    item_files: dict[str, str] = {}
    for record in records:
        record_id = str(record.get("id", "")).strip()
        if record_id:
            item_path = items_dir / f"{record_id}.json"
            write_json(item_path, record)
            item_files[record_id] = str(item_path.relative_to(query_root))
        grouped.setdefault(str(record.get("interaction_type") or "unknown"), []).append(record)

    index_payload = {
        "total": len(records),
        "items_dir": "items",
        "items": item_files,
        "groups": {},
    }
    for label, _ in INTERACTION_ARCHETYPE_WEIGHTS:
        group_dir = query_root / label
        group_dir.mkdir(parents=True, exist_ok=True)
        group_items_dir = group_dir / "items"
        group_items_dir.mkdir(parents=True, exist_ok=True)
        for stale_file in group_items_dir.glob("*.json"):
            stale_file.unlink()
        group_records = grouped.get(label, [])
        group_item_files: dict[str, str] = {}
        for record in group_records:
            record_id = str(record.get("id", "")).strip()
            if not record_id:
                continue
            item_path = group_items_dir / f"{record_id}.json"
            write_json(item_path, record)
            group_item_files[record_id] = str(item_path.relative_to(query_root))
        write_json(group_dir / "queries.json", group_records)
        index_payload["groups"][label] = {
            "label": INTERACTION_LABELS[label],
            "description": INTERACTION_DESCRIPTIONS[label],
            "count": len(group_records),
            "file_name": "queries.json",
            "items_dir": f"{label}/items",
            "items": group_item_files,
        }
    write_json(query_root / "_index.json", index_payload)


def write_outputs(records: list[dict[str, Any]], output_path: Path, query_root: Path, environment_output: str = "") -> Counter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, records)
    if environment_output:
        environment_records = [record for record in records if record.get("interaction_type") == "environment_driven_replanning"]
        write_json(Path(environment_output), environment_records)
    write_grouped(records, query_root)
    return Counter(record["interaction_type"] for record in records)
