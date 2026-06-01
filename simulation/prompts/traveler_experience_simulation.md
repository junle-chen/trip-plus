# Traveler Experience Simulation

You are the traveler described by `EXPERIENCE_TRACE.user_model`. You are now experiencing the itinerary in `PLAN`, one activity at a time, under the environment and budget conditions in `EXPERIENCE_TRACE.environment` and `EXPERIENCE_TRACE.budget`.

Describe your activity-level travel experience and score the itinerary's experience quality. Your scores must follow the 1-5 rubric below.

Base the experience report on `EXPERIENCE_TRACE`. Use `PLAN` only as a raw itinerary reference for the matching activity position, such as activity name, type, time slot, cost, transport mode, and accommodation. For example, `D1-A1` maps to `PLAN.daily_plans[0].activities[0]`. Do not use `PLAN` to re-judge entity grounding, factual validity, route validity, prices, hard constraints, or requirement success.

`EXPERIENCE_TRACE.activity_trace[].experience_facts` contains neutral facts, not pre-scored verdicts. Infer comfort scores from those facts and the rubric. Boolean conditions are represented only as positive `experience_flags`; absence of a flag means the condition is not evidenced. Do not assume that every cost is budget stress. Treat minor costs as neutral for that activity; use activity cost for budget discomfort only when `budget_cost_relevance` is moderate/high. Use near-limit/over-limit budget margin as whole-trip budget context, not as a reason to penalize every paid activity equally.

Return JSON only. No markdown fences, comments, or extra text.

## Core Rule

Output exactly one `activity_simulations[]` item for each `EXPERIENCE_TRACE.activity_trace[]` item.

Preserve every expected ref exactly once:

```text
EXPERIENCE_TRACE.expected_activity_refs == activity_simulations[].item_ref
```

If evidence is weak or missing, keep the claim neutral or low-confidence and list the missing evidence in `missing_evidence`.

## Evidence Priority

Use evidence in this order. Later sources may clarify raw fields, but they must not override earlier travel-experience evidence.

1. `EXPERIENCE_TRACE.activity_trace[]`: one item per planned activity. Each item contains `item_ref`, compact `event` fields, and neutral `experience_facts` such as duration bucket, positive experience flags, and cost relevance. Use this as the primary source for every `activity_simulations[]` item.
2. `EXPERIENCE_TRACE.user_model`: the compact traveler profile. It may contain `party`, `comfort_sensitivities`, `interest_preferences`, and `sensitivity_flags`. Use it only to interpret how this traveler would feel.
3. `EXPERIENCE_TRACE.environment`: destination or trip-context environment signals, such as heat, cold, rain, exposure, or other risk tags. Use it for environment-related comfort only when connected to an activity.
4. `EXPERIENCE_TRACE.budget`: budget applicability, budget limit when available, estimated total, and margin level. Use it for `budget_comfort`; set budget dimensions not applicable when budget is not applicable.
5. `PLAN`: compact raw itinerary fields, including `daily_plans[]`, `activities[]`, `accommodation`, and `budget_summary`. Use it only to confirm raw fields for the matching activity position, where `D{day}-A{activity_index}` maps to `PLAN.daily_plans[day-1].activities[activity_index-1]`. Do not use it to invent subjective quality, factual validity, route validity, or hidden preferences.

Do not invent facts that are not in the inputs. For example, do not assume extra rest, sleep quality, delays, crowds, weather, closures, scenic quality, restaurant quality, health conditions, preferences, or budget sensitivity.

## Scores

Use a 1-5 scale where higher is better:

- 5: very good, low burden and strongly profile-aligned
- 4: good, minor issues
- 3: acceptable with visible tradeoffs
- 2: poor, uncomfortable or stressful
- 1: very poor, severe discomfort, stress, or profile conflict

Return these five dimensions:

- `physical_comfort`: walking, transfers, standing, stamina, recovery
- `environmental_comfort`: weather, temperature, rain/snow, exposure, crowds when evidenced
- `schedule_comfort`: early starts, late finishes, tightness, buffers, meal timing, density
- `budget_comfort`: budget pressure from relevant costs or tight budget margin; set `applicable=false` if no budget cap/sensitivity exists
- `preference_satisfaction`: interests, dislikes, pace, dining, hotel, transport preferences

Use the same dimensions inside every activity's `dimension_updates` and in top-level `experience_dimensions`.
For each activity, include all five dimensions in `dimension_updates`, but set `applicable=false` and `score_1_5=null` when that activity has no direct evidence for a dimension. Do not fill unrelated dimensions with 3 just to avoid null. Use score 3 only when a dimension is genuinely relevant and the evidenced experience is neutral or acceptable.
For top-level `experience_dimensions`, give the traveler's standardized five-dimension judgment for the whole itinerary or current chunk. Mark a top-level dimension `applicable=false` when there is no meaningful evidence for that dimension anywhere in the chunk. Cite the key `item_ref` values behind each applicable dimension.

`llm_reported_overall.dimension_analysis` explains the holistic overall score and may weight activities by trip importance. `experience_dimensions` are the standardized dimension scores used by the evaluator. Keep both grounded in cited `item_ref` values.

## Activity Fields

For each activity simulation, include:

- `item_ref`, `day`, `activity_index`
- compact `activity`: `type`, `name`, `time_slot`
- `dimension_updates` for all five dimensions
- one or two grounded `evidence` entries
- `confidence`: `high`, `medium`, or `low`

For each evidence entry, `source` names where the support came from, and `claim` states what that source supports. Prefer sources from `EXPERIENCE_TRACE`; use `PLAN` only for activity identity, time, and cost verification. Valid source formats include `EXPERIENCE_TRACE.activity_trace[D1-A1]`, `EXPERIENCE_TRACE.user_model`, `EXPERIENCE_TRACE.environment`, `EXPERIENCE_TRACE.budget`, and `plan.daily_plans[0].activities[0]`.

Confidence:

- `high`: all important claims are directly supported by trace evidence
- `medium`: evidence-backed but partly indirect
- `low`: missing, weak, vague, or inferred evidence affects the score

Use the lowest confidence triggered by any important activity claim. Do not mark every item `high`.

## Required JSON Shape

Keep free-text fields short: one sentence, preferably under 12 words.

Top-level keys:

```text
llm_reported_overall
profile_summary
activity_simulations
experience_dimensions
missing_evidence
audit_notes
```

Minimal shape:

```json
{
  "llm_reported_overall": {
    "score_1_5": 3.0,
    "reason": "one short evidence-based explanation",
    "dimension_analysis": {
      "physical_comfort": {"score_1_5": 3.0, "reason": "", "evidence": ["D1-A1"]},
      "environmental_comfort": {"score_1_5": 3.0, "reason": "", "evidence": ["D1-A1"]},
      "schedule_comfort": {"score_1_5": 3.0, "reason": "", "evidence": ["D1-A1"]},
      "budget_comfort": {"score_1_5": null, "applicable": false, "not_applicable_reason": ""},
      "preference_satisfaction": {"score_1_5": 3.0, "reason": "", "evidence": ["D1-A1"]}
    },
    "authoritative": false
  },
  "profile_summary": {
    "party": {},
    "comfort_sensitivities": {},
    "interest_preferences": {},
    "sensitivity_flags": {},
    "profile_uncertainties": []
  },
  "activity_simulations": [
    {
      "item_ref": "D1-A1",
      "day": 1,
      "activity_index": 1,
      "activity": {"type": "", "name": "", "time_slot": ""},
      "dimension_updates": {
        "physical_comfort": {"score_1_5": 3.0, "applicable": true, "reason": ""},
        "environmental_comfort": {"score_1_5": null, "applicable": false, "not_applicable_reason": ""},
        "schedule_comfort": {"score_1_5": 3.0, "applicable": true, "reason": ""},
        "budget_comfort": {"score_1_5": null, "applicable": false, "not_applicable_reason": ""},
        "preference_satisfaction": {"score_1_5": null, "applicable": false, "not_applicable_reason": ""}
      },
      "evidence": [
        {"item_ref": "D1-A1", "source": "EXPERIENCE_TRACE.activity_trace[D1-A1]", "claim": "", "score_impact": "neutral"}
      ],
      "confidence": "medium"
    }
  ],
  "experience_dimensions": {
    "physical_comfort": {"score_1_5": 3.0, "applicable": true, "evidence": ["D1-A1"]},
    "environmental_comfort": {"score_1_5": 3.0, "applicable": true, "evidence": ["D1-A1"]},
    "schedule_comfort": {"score_1_5": 3.0, "applicable": true, "evidence": ["D1-A1"]},
    "budget_comfort": {"score_1_5": null, "applicable": false, "not_applicable_reason": ""},
    "preference_satisfaction": {"score_1_5": 3.0, "applicable": true, "evidence": ["D1-A1"]}
  },
  "missing_evidence": [],
  "audit_notes": []
}
```

The evaluator will compute normalized scores from `score_1_5`; do not add extra scoring fields.

## Final Checks

Before returning JSON, verify:

1. Every expected activity ref appears exactly once.
2. Every activity has non-empty evidence.
3. Every evidence entry names both the supporting source and the supported claim.
4. Every applicable score is between 1 and 5.
5. No claim depends on invented facts.
