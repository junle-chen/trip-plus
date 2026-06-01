"""Optional LLM rendering for non-initial multi-turn user messages."""

from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


def _build_turn_render_prompt(record: dict[str, Any], turn: dict[str, Any]) -> list[dict[str, str]]:
    payload = {
        "base_query": record.get("query"),
        "previous_turns": [
            item.get("utterance")
            for item in record.get("turns", [])
            if isinstance(item, dict) and int(item.get("turn_id") or 0) < int(turn.get("turn_id") or 0)
        ],
        "draft_utterance": turn.get("utterance"),
        "must_preserve": turn.get("must_preserve"),
        "must_update": turn.get("must_update"),
        "response_expectation": turn.get("response_expectation"),
    }
    return [
        {
            "role": "system",
            "content": (
                "Rewrite the current user turn as natural English chat. Preserve all facts, dates, numbers, "
                "entity names, constraints, and response expectations. Output only the rewritten user message."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def _llm_render_record(record: dict[str, Any], args: Any) -> dict[str, Any]:
    from agent.call_llm import call_llm

    rendered = copy.deepcopy(record)
    for turn in rendered.get("turns", []):
        if not isinstance(turn, dict) or int(turn.get("turn_id") or 0) == 0:
            continue
        response = call_llm(
            config_name=args.turn_render_model,
            messages=_build_turn_render_prompt(rendered, turn),
            request_overrides={
                "temperature": float(args.turn_render_temperature),
                "max_tokens": int(args.turn_render_max_tokens),
            },
        )
        turn["utterance"] = str(response.choices[0].message.content).strip().strip('"')
    return rendered


def llm_render_multiturn_surfaces(records: list[dict[str, Any]], args: Any) -> list[dict[str, Any]]:
    if not args.llm_render_turns or not records:
        return records
    workers = max(1, min(int(args.turn_render_workers or 1), len(records)))
    if workers == 1:
        return [_llm_render_record(record, args) for record in records]
    rendered: list[dict[str, Any] | None] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {executor.submit(_llm_render_record, record, args): idx for idx, record in enumerate(records)}
        for future in as_completed(future_to_idx):
            rendered[future_to_idx[future]] = future.result()
    return [record or records[idx] for idx, record in enumerate(rendered)]
