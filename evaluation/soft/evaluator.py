"""Public entry points for deterministic User Alignment scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .budget import _score_budget, _score_hotel
from .comfort import _score_mobility, _score_schedule, _score_weather
from .common import (
    _evidence_basis_for_rule,
    _hard_ratio,
    _preference_family_for_rule,
    _rule_metadata_by_canonical,
    _safe_average,
    _source_rule_ids,
)
from .config import SOFT_CHECK_RULE_IDS, SOFT_PREFERENCE_FAMILY_ORDER
from .interest import _score_interest
from .transport import _score_transport


def _rule_ids_from_meta(meta: Dict[str, Any]) -> List[str]:
    """Extract active profile rule ids from single-turn and multi-turn metadata.

    Single-turn samples store rules under ``meta.user_profile.rule_ids``. For
    multi-turn samples, the active turn can add rules through
    ``turn_ground_truth.verification_oracle`` or oracle state deltas.
    """
    rule_ids: List[str] = []
    user_profile = meta.get("user_profile") or {}
    if isinstance(user_profile, dict):
        rule_ids.extend(
            str(item)
            for item in user_profile.get("rule_ids") or []
            if str(item).strip()
        )
        for rule in user_profile.get("rules") or []:
            if isinstance(rule, dict) and str(rule.get("rule_id") or "").strip():
                rule_ids.append(str(rule["rule_id"]))

    turn_gt = meta.get("turn_ground_truth") or {}
    if isinstance(turn_gt, dict):
        oracle = turn_gt.get("verification_oracle") or {}
        if isinstance(oracle, dict):
            rule_ids.extend(
                str(item)
                for item in oracle.get("profile_rule_ids") or []
                if str(item).strip()
            )
        state = turn_gt.get("oracle_state_after_turn") or {}
        if isinstance(state, dict):
            for delta in state.get("active_profile_deltas") or []:
                if isinstance(delta, dict):
                    pref = delta.get("profile_preference") or {}
                    if isinstance(pref, dict):
                        rule_ids.extend(
                            str(item)
                            for item in pref.get("rule_ids") or []
                            if str(item).strip()
                        )
            for delta in state.get("active_user_state_deltas") or []:
                if isinstance(delta, dict):
                    pref = delta.get("profile_preference") or {}
                    if isinstance(pref, dict):
                        rule_ids.extend(
                            str(item)
                            for item in pref.get("rule_ids") or []
                            if str(item).strip()
                        )

    return list(dict.fromkeys(rule_ids))


def _active_soft_rule_ids(
    meta: Dict[str, Any], rule_ids: Optional[Iterable[str]] = None
) -> List[str]:
    return list(
        dict.fromkeys(
            str(item)
            for item in (rule_ids or _rule_ids_from_meta(meta))
            if str(item).strip()
        )
    )


def compile_soft_checks(
    meta: Dict[str, Any], rule_ids: Optional[Iterable[str]] = None
) -> List[Dict[str, Any]]:
    """Compile active profile rules into deterministic soft checks.

    The optional ``rule_ids`` argument is used by multi-turn evaluation when a
    turn should check only one preference family, such as transport or hotel.
    Unsupported rules are intentionally skipped instead of being guessed.
    """
    active = _active_soft_rule_ids(meta, rule_ids)
    metadata_by_rule = _rule_metadata_by_canonical(meta)
    canonical_sources: Dict[str, List[str]] = {}
    for raw_rule_id in active:
        canonical_rule_id = raw_rule_id
        if canonical_rule_id not in SOFT_CHECK_RULE_IDS:
            continue
        canonical_sources.setdefault(canonical_rule_id, []).append(raw_rule_id)

    checks: List[Dict[str, Any]] = []
    for rule_id, input_rule_ids in canonical_sources.items():
        checks.append(
            {
                "rule_id": rule_id,
                "canonical_rule_id": rule_id,
                "input_rule_ids": input_rule_ids,
                "source_rule_ids": _source_rule_ids(
                    input_rule_ids, metadata_by_rule.get(rule_id)
                ),
                "rule_metadata": metadata_by_rule.get(rule_id, []),
                "preference_family": _preference_family_for_rule(rule_id),
                "source": "user_profile.rules",
                "score_type": "0_0.5_1",
                "evidence_basis": _evidence_basis_for_rule(rule_id),
            }
        )
    return checks


def evaluate_soft_checks(
    plan: Dict[str, Any],
    meta: Dict[str, Any],
    rule_ids: Optional[Iterable[str]] = None,
    database_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Evaluate all active deterministic soft-preference checks.

    The returned score is the average of applicable checks. When no supported
    or applicable soft rules are active, the score is ``None`` and callers
    should omit the soft component from the final requirement score.
    """
    requested_rule_ids = _active_soft_rule_ids(meta, rule_ids)
    unsupported_rule_ids = [
        item for item in requested_rule_ids if item not in SOFT_CHECK_RULE_IDS
    ]
    checks = compile_soft_checks(meta, rule_ids=rule_ids)
    results: List[Dict[str, Any]] = []
    for check in checks:
        rule_id = check["rule_id"]
        if rule_id == "schedule_pacing":
            result = _score_schedule(
                plan,
                rule_id,
                source_rule_ids=check["source_rule_ids"],
                rule_metadata=check["rule_metadata"],
            )
        elif rule_id == "mobility_accessibility":
            result = _score_mobility(
                plan,
                rule_id,
                database_dir=database_dir,
                source_rule_ids=check["source_rule_ids"],
                rule_metadata=check["rule_metadata"],
            )
        elif rule_id.startswith("weather_"):
            result = _score_weather(plan, meta, rule_id, database_dir=database_dir)
        elif rule_id.startswith("transport_"):
            result = _score_transport(plan, rule_id)
        elif rule_id.startswith("hotel_"):
            result = _score_hotel(
                plan,
                rule_id,
                database_dir=database_dir,
                source_rule_ids=check["source_rule_ids"],
                rule_metadata=check["rule_metadata"],
            )
        elif rule_id.startswith("budget_") or rule_id == "meal_avoid_expensive":
            result = _score_budget(plan, meta, rule_id, database_dir=database_dir)
        elif rule_id.startswith("interest_"):
            result = _score_interest(
                plan,
                rule_id,
                database_dir=database_dir,
                source_rule_ids=check["source_rule_ids"],
                rule_metadata=check["rule_metadata"],
            )
        else:
            continue
        result.update(
            {
                "canonical_rule_id": check["canonical_rule_id"],
                "input_rule_ids": check["input_rule_ids"],
                "source_rule_ids": check["source_rule_ids"],
                "preference_family": check["preference_family"],
                "source": check["source"],
                "score_type": check["score_type"],
                "evidence_basis": check["evidence_basis"],
            }
        )
        results.append(result)

    family_scores: List[Dict[str, Any]] = []
    for family in SOFT_PREFERENCE_FAMILY_ORDER:
        family_items = [
            item
            for item in results
            if item.get("preference_family") == family
            and item.get("applicable", True)
            and item.get("score") is not None
        ]
        if not family_items:
            continue
        family_score = _safe_average(
            [float(item["score"]) for item in family_items], default=None
        )
        family_scores.append(
            {
                "family": family,
                "score": round(float(family_score), 4)
                if family_score is not None
                else None,
                "evaluated_count": len(family_items),
                "rule_ids": [str(item["rule_id"]) for item in family_items],
            }
        )

    applicable_scores = [
        float(item["score"])
        for item in results
        if item.get("applicable", True) and item.get("score") is not None
    ]
    score = _safe_average(
        [
            float(item["score"])
            for item in family_scores
            if item.get("score") is not None
        ],
        default=None,
    )
    return {
        "score": round(score, 4) if score is not None else None,
        "checks": results,
        "preference_family_scores": family_scores,
        "active_rule_ids": [item["rule_id"] for item in results],
        "requested_rule_ids": requested_rule_ids,
        "unsupported_rule_ids": unsupported_rule_ids,
        "evaluated_count": len(applicable_scores),
        "evaluated_family_count": len(family_scores),
        "compiled_count": len(results),
        "not_applicable_count": sum(
            1 for item in results if not item.get("applicable", True)
        ),
    }


def _strict_soft_preference_score(soft: Dict[str, Any]) -> Optional[float]:
    applicable_checks = [
        item
        for item in soft.get("checks") or []
        if isinstance(item, dict)
        and item.get("applicable", True)
        and item.get("score") is not None
    ]
    if not applicable_checks:
        return None
    for item in applicable_checks:
        try:
            if float(item["score"]) < 1.0 - 1e-9:
                return 0.0
        except (TypeError, ValueError):
            return 0.0
    return 1.0


def calculate_user_alignment(
    plan: Dict[str, Any],
    meta: Dict[str, Any],
    hard_result: Dict[str, Any],
    *,
    rule_ids: Optional[Iterable[str]] = None,
    database_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Combine hard-constraint and soft-preference satisfaction.

    This is the public entry point used by single-turn evaluation. It reports
    the strict hard all-pass score, the continuous hard score, and the compiled soft
    checks, then averages the applicable components into ``requirement_score``.
    """
    strict_hard = float(hard_result.get("score", 0.0))
    hard_ratio, hard_passed, hard_total = _hard_ratio(hard_result)
    soft = evaluate_soft_checks(
        plan, meta, rule_ids=rule_ids, database_dir=database_dir
    )
    soft_score = soft["score"] if soft["evaluated_family_count"] else None
    strict_soft = _strict_soft_preference_score(soft)
    components = [{"name": "hard_constraint_score", "score": hard_ratio, "weight": 1.0}]
    if soft_score is not None:
        components.append(
            {"name": "soft_preference_score", "score": soft_score, "weight": 1.0}
        )
        score = _safe_average([hard_ratio, float(soft_score)], default=hard_ratio)
    else:
        score = hard_ratio
    return {
        "score": round(float(score), 4),
        "strict_hard_constraint": strict_hard,
        "hard_constraint_score": round(hard_ratio, 4),
        "hard_constraint_ratio": round(hard_ratio, 4),
        "hard_constraints_passed": hard_passed,
        "hard_constraints_total": hard_total,
        "strict_soft_preference": strict_soft,
        "soft_preference_score": soft_score,
        "soft_preferences": soft,
        "components": components,
        "constraints": hard_result.get("constraints", {}),
        "hard_constraints": hard_result,
    }
