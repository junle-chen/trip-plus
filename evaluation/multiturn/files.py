"""File and database resolution for multi-turn evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.sample_db_resolver import resolve_sample_database_path_with_query


def _sample_id_variants(sample_id: object, base_query_id: object = None) -> List[str]:
    variants: List[str] = []
    for raw in (sample_id, base_query_id):
        text = str(raw or "").strip()
        if not text:
            continue
        for candidate in (
            text,
            text.replace("id_", ""),
            f"id_{text.replace('id_', '')}",
        ):
            if candidate not in variants:
                variants.append(candidate)
    return variants


def find_turn_plan(
    plans_dir: Path, sample_id: object, turn_id: object, base_query_id: object = None
) -> Optional[Path]:
    """Find a converted plan for a sample turn using common filename patterns."""
    turn_text = str(turn_id)
    candidates: List[Path] = []
    for sid in _sample_id_variants(sample_id, base_query_id):
        candidates.extend(
            [
                plans_dir / f"{sid}_turn_{turn_text}_converted.json",
                plans_dir / f"{sid}_t{turn_text}_converted.json",
                plans_dir / f"{sid}.turn_{turn_text}.converted.json",
                plans_dir / f"{sid}" / f"turn_{turn_text}_converted.json",
                plans_dir / f"{sid}" / f"t{turn_text}_converted.json",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_multiturn_database_path(
    record: Dict[str, Any],
    database_root: Path,
    language: str,
    query_file: Path,
) -> Optional[Path]:
    """Resolve the materialized DB for a multi-turn record.

    Multi-turn IDs are usually ``mt_*`` while DB caches are built for the
    underlying single-turn ``base_query_id``. Always go through the resolver so
    cache schema, query signature, and source-database freshness checks are
    applied consistently between inference tools and evaluation.
    """
    sample_id = str(record.get("base_query_id") or record.get("id") or "").strip()
    if not sample_id:
        return None
    try:
        return resolve_sample_database_path_with_query(
            sample_id=sample_id,
            database_root=database_root,
            language=language,
            query_file=query_file,
        )
    except Exception:
        return None
