"""Sample-id parsing and output-completeness checks."""

from __future__ import annotations

import json
import re
from pathlib import Path


def sample_sort_key(sample_id):
    sample_id = str(sample_id)
    return (0, int(sample_id)) if sample_id.isdigit() else (1, sample_id)


def report_stem(sample_id) -> str:
    sample_id = str(sample_id)
    return f"id_{sample_id}" if sample_id.isdigit() else sample_id


def load_test_sample_ids(test_data_path: Path) -> list[str]:
    return [
        str(sample.get("id"))
        for sample in load_test_samples(test_data_path)
        if sample.get("id") is not None
    ]


def load_test_samples(test_data_path: Path) -> list[dict]:
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    if isinstance(test_data, list):
        return [sample for sample in test_data if isinstance(sample, dict)]
    if isinstance(test_data, dict):
        return [test_data]
    raise ValueError(f"Query file must contain a JSON object or list: {test_data_path}")


def expected_converted_output_ids(test_data_path: Path) -> list[str]:
    """Return expected converted-plan ids for single- and multi-turn data."""
    expected: list[str] = []
    for sample in load_test_samples(test_data_path):
        sample_id = str(sample.get("id") or "").strip()
        if not sample_id:
            continue
        turns = [turn for turn in sample.get("turns", []) if isinstance(turn, dict)]
        if turns:
            for turn_index, turn in enumerate(turns):
                turn_id = str(turn.get("turn_id", turn_index))
                expected.append(f"{sample_id}_turn_{turn_id}")
        else:
            expected.append(sample_id)
    return expected


def is_multiturn_test_data(test_data_path: Path) -> bool:
    return any(
        isinstance(sample, dict) and isinstance(sample.get("turns"), list)
        for sample in load_test_samples(test_data_path)
    )


def detect_missing_converted_outputs(directory: Path, expected_ids: list[str]) -> list[str]:
    """Detect missing converted plan files for numeric and string sample IDs."""
    if not directory.exists():
        return sorted(
            [str(sample_id) for sample_id in expected_ids], key=sample_sort_key
        )

    existing_names = {path.name for path in directory.iterdir() if path.is_file()}
    missing: list[str] = []
    for sample_id in expected_ids:
        sample_id = str(sample_id)
        expected_name = f"id_{sample_id}_converted.json"
        if expected_name not in existing_names:
            missing.append(sample_id)

    return sorted(missing, key=sample_sort_key)


def detect_missing_report_parent_ids(directory: Path, test_data_path: Path) -> list[str]:
    """Return parent sample ids whose single-turn or multi-turn reports are incomplete."""
    if not directory.exists():
        return sorted(load_test_sample_ids(test_data_path), key=sample_sort_key)

    existing_names = {path.name for path in directory.iterdir() if path.is_file()}
    missing_parent_ids: list[str] = []
    for sample in load_test_samples(test_data_path):
        sample_id = str(sample.get("id") or "").strip()
        if not sample_id:
            continue
        turns = [turn for turn in sample.get("turns", []) if isinstance(turn, dict)]
        if turns:
            expected_names = [
                f"{sample_id}_turn_{turn.get('turn_id', turn_index)}.txt"
                for turn_index, turn in enumerate(turns)
            ]
        else:
            expected_names = [f"{report_stem(sample_id)}.txt"]
        if any(name not in existing_names for name in expected_names):
            missing_parent_ids.append(sample_id)

    return sorted(missing_parent_ids, key=sample_sort_key)


def conversion_output_ids_for_rerun_ids(
    test_data_path: Path, rerun_ids: list
) -> list[str]:
    """Expand parent multi-turn sample IDs to their per-turn converted plan IDs."""
    requested = {str(sample_id) for sample_id in (rerun_ids or [])}
    output_ids: list[str] = []
    matched: set[str] = set()
    for sample in load_test_samples(test_data_path):
        sample_id = str(sample.get("id") or "").strip()
        if not sample_id:
            continue
        turns = [turn for turn in sample.get("turns", []) if isinstance(turn, dict)]
        if sample_id in requested:
            matched.add(sample_id)
            if turns:
                for turn_index, turn in enumerate(turns):
                    output_ids.append(
                        f"{sample_id}_turn_{turn.get('turn_id', turn_index)}"
                    )
            else:
                output_ids.append(sample_id)
            continue
        for turn_index, turn in enumerate(turns):
            turn_output_id = f"{sample_id}_turn_{turn.get('turn_id', turn_index)}"
            if turn_output_id in requested:
                matched.add(turn_output_id)
                output_ids.append(turn_output_id)
    output_ids.extend(sorted(requested - matched, key=sample_sort_key))
    return sorted(set(output_ids), key=sample_sort_key)


def parse_id_list(id_str: str) -> list:
    """Parse numeric, string, and prefixed range sample IDs."""
    if not id_str:
        return None

    ids = set()
    for part in id_str.split(","):
        part = part.strip()
        if "-" not in part:
            try:
                ids.add(int(part))
            except ValueError:
                ids.add(part)
            continue

        try:
            start_raw, end_raw = (piece.strip() for piece in part.split("-", 1))
            start_match = re.fullmatch(r"([A-Za-z_]+?)(\d+)", start_raw)
            end_match = re.fullmatch(r"([A-Za-z_]+?)(\d+)", end_raw)
            if start_match and end_match and start_match.group(1) == end_match.group(1):
                prefix = start_match.group(1)
                width = max(len(start_match.group(2)), len(end_match.group(2)))
                start = int(start_match.group(2))
                end = int(end_match.group(2))
                ids.update(f"{prefix}{i:0{width}d}" for i in range(start, end + 1))
            elif start_raw.isdigit() and end_raw.isdigit():
                ids.update(range(int(start_raw), int(end_raw) + 1))
            else:
                ids.add(part)
        except ValueError:
            ids.add(part)

    return sorted(list(ids), key=sample_sort_key)
