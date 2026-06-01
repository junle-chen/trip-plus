from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any


def _sample_folder_names(sample_id: object) -> list[str]:
    text = str(sample_id or "").strip()
    if not text:
        return []
    names: list[str] = []

    def add(value: str) -> None:
        if value and value not in names:
            names.append(value)

    add(text)
    if text.startswith("id_"):
        return names
    match = re.search(r"(\d+)$", text)
    if match:
        add(f"id_{int(match.group(1)):04d}")
    add(f"id_{text}")
    return names


def _sample_id_variants(sample_id: object) -> list[str]:
    text = str(sample_id or "").strip()
    variants: list[str] = []

    def add(value: str) -> None:
        if value and value not in variants:
            variants.append(value)

    turn_match = re.match(r"(.+)_turn_\d+$", text)
    if turn_match:
        add(turn_match.group(1))
    add(text)
    return variants


def _base_query_id_from_query_file(sample_id: object, query_file: str | Path | None) -> str:
    if query_file is None:
        return ""
    path = Path(query_file)
    if not path.exists():
        return ""
    target_ids = set(_sample_id_variants(sample_id))
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(records, list):
        return ""
    for record in records:
        if not isinstance(record, dict):
            continue
        record_id = str(record.get("id") or "").strip()
        if record_id in target_ids:
            return str(record.get("base_query_id") or "").strip()
    return ""


def _candidate_roots(root: Path, language: str) -> list[Path]:
    if root.name == language and root.parent.name == "sample":
        return [root]
    if root.name == "sample":
        return [root / language, root]
    if root.name == language:
        return [root]
    return [root / "sample" / language, root / language, root]


def resolve_sample_database_path_with_query(
    sample_id: object,
    database_root: str | Path,
    language: str = "en",
    query_file: str | Path | None = None,
    **_: Any,
) -> Path:
    """Resolve the English sample cache directory for one query id."""
    if language != "en":
        raise ValueError(f"Unsupported language: {language!r}. This release only supports 'en'.")

    root = Path(database_root)
    ids = _sample_id_variants(sample_id)
    base_query_id = _base_query_id_from_query_file(sample_id, query_file)
    if base_query_id:
        ids.append(base_query_id)

    folders: list[str] = []
    for identifier in ids:
        for folder in _sample_folder_names(identifier):
            if folder not in folders:
                folders.append(folder)

    candidates: list[Path] = []
    for folder in folders:
        if root.name == folder:
            candidates.append(root)
    for candidate_root in _candidate_roots(root, language):
        for folder in folders:
            candidates.append(candidate_root / folder)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else root
