"""Observable traveler-profile sampler for English query generation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROFILE_PATH = Path(__file__).with_name("observable_profiles_en.json")


@dataclass(frozen=True)
class ObservableArchetype:
    persona_id: str
    name: str
    weight: int
    observable_template: dict[str, Any]


def _resolve_profile_path(path: Path | str | None = None) -> Path:
    if path is not None:
        return Path(path)
    return PROFILE_PATH


def load_observable_archetypes(
    path: Path | str | None = None,
) -> list[ObservableArchetype]:
    profile_path = _resolve_profile_path(path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    return [
        ObservableArchetype(
            persona_id=str(item["persona_id"]),
            name=str(item["name"]),
            weight=int(item["weight"]),
            observable_template=dict(item["observable_template"]),
        )
        for item in payload
    ]


LIST_OPTION_FIELDS = {
    "budget_range",
    "dietary",
    "mobility_constraints",
    "physical_rules",
    "schedule_rules",
    "interest_tags",
    "hate_tags",
    "transport_preferences",
    "rest_preferences",
}


def _pick(value: Any, rng: random.Random, field_name: str = "") -> Any:
    if isinstance(value, list):
        if not value:
            return []
        if field_name in LIST_OPTION_FIELDS:
            picked = rng.choice(value)
            if isinstance(picked, list):
                return list(picked)
            return picked
        if all(isinstance(item, (str, int, float, bool)) for item in value):
            return rng.choice(value)
        return _pick(rng.choice(value), rng, field_name=field_name)
    if isinstance(value, dict):
        if set(value.keys()) == {"age", "note"}:
            age_range = value["age"]
            note_value = value["note"]
            age = rng.randint(int(age_range[0]), int(age_range[1]))
            note = rng.choice(note_value) if isinstance(note_value, list) else str(note_value)
            return {"age": age, "note": note}
        if set(value.keys()) == {"age", "mobility_note"}:
            age_range = value["age"]
            note_value = value["mobility_note"]
            age = rng.randint(int(age_range[0]), int(age_range[1]))
            note = rng.choice(note_value) if isinstance(note_value, list) else str(note_value)
            return {"age": age, "mobility_note": note}
        if set(value.keys()) == {"adults", "children", "elders"}:
            adults = rng.choice(value["adults"])
            children = [_pick(item, rng) for item in value["children"]]
            elders = [_pick(item, rng) for item in value["elders"]]
            return {"adults": adults, "children": children, "elders": elders}
        return {key: _pick(item, rng, field_name=key) for key, item in value.items()}
    return value


class ObservableProfileSampler:
    def __init__(
        self,
        archetypes: list[ObservableArchetype] | None = None,
        *,
        profile_path: Path | str | None = None,
    ):
        self.archetypes = archetypes or load_observable_archetypes(profile_path)

    def sample(self, rng: random.Random) -> dict[str, Any]:
        picked = rng.choices(self.archetypes, weights=[item.weight for item in self.archetypes], k=1)[0]
        observable = _pick(picked.observable_template, rng)
        return {
            "persona_id": picked.persona_id,
            "persona_name": picked.name,
            "observable": observable,
        }
