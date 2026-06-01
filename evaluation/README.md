# Evaluation

`evaluation/` converts agent reports into structured plans and scores those
plans against fixed benchmark rules.

## Main Entry

The integrated path is `run.py` with the released multi-turn query file:

```bash
python run.py --test-data query/query_en/multiturn/query.json
```

Conversion can also be called directly:

```bash
python -m evaluation.conversion --result-dir result/<run>/<model>_en
```

## Top-Level Files

- `single_turn.py`: orchestrates single-turn feasibility, hard, and soft checks.
  It is still used as the underlying per-plan evaluator for converted plans.
- `response_cases.py`: scores non-itinerary responses such as clarification and
  no-solution.
- `costing.py`: computes estimated costs from structured plan activities.
- `output.py`: writes score payloads and summary files.
- `summary.py`: aggregates evaluation records for reporting.
- `runtime.py`: extracts runtime and token metadata from trajectories.
- `scoring_config.py`: shared score labels, weights, and thresholds.
- `utils.py`: evaluation-only parsing and normalization helpers.

## Subpackages

- `conversion/`: deterministic report parsing, JSON extraction, entity-name
  cleanup, LLM-backed conversion, and conversion CLI.
- `feasibility/`: structure, evidence, timing, and budget feasibility checks.
- `hard/`: explicit hard-constraint checks from the query.
- `soft/`: traveler-profile preference checks.
- `multiturn/`: multi-turn fulfillment, response-mode, and preservation checks.

For multi-turn reporting, request fulfillment is derived from current-turn
updates (`must_update` plus inferred updates from active state deltas), while
preservation checks earlier active commitments.

## Boundary

- Evaluation reads structured plans and benchmark evidence.
- Evaluation should not call planner tools or regenerate queries.
- LLM user simulation is standalone under `simulation/`.
