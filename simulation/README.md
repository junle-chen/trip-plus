# User Simulation

`simulation/` reruns the LLM traveler-experience simulator on existing
converted multi-turn plans. It does not rerun planning, conversion, or
deterministic evaluation.

Simulation expects completed multi-turn result directories produced from
`query/query_en/multiturn/query.json`. It does not need the optional
single-turn query directory or split query views.

Standalone simulator artifacts are written under `result/user_simulation/` by
default, not under the `simulation/` source package.

## Main Entry

Run all simulator judges and aggregate the median:

```bash
python -m simulation.run_user_simulation \
  --all-judges \
  --result-dir result/example_run/qwen3.6-27b-vllm_en
```

Run one simulator judge on completed results:

```bash
python -m simulation.run_user_simulation \
  --simulator-model qwen \
  --result-dir result/example_run/qwen3.6-27b-vllm_en
```

Aggregate the four-judge median after artifacts are present. The
`--result-dir` is the source planner run being judged; the judge models are
selected separately:

```bash
python -m simulation.aggregate_median \
  --result-dir result/example_run/qwen3.6-27b-vllm_en \
  --judge gpt-5.4-nano \
  --judge claude-haiku-4-5-20251001 \
  --judge gemini-3.1-flash-lite \
  --judge qwen
```

## Layout

- `run_user_simulation.py`: single-judge and all-judge simulator CLI.
- `batch_runner.py`: result discovery, artifact reuse, per-sample execution,
  and run summaries.
- `simulator.py`: one-itinerary LLM simulation orchestration.
- `prompting.py`: prompt loading and compact input construction.
- `experience_trace.py`: deterministic activity trace from plan and query.
- `scoring.py`: output normalization, dimension scoring, and validation checks.
- `chunking.py`: bounded activity-chunk simulation for long itineraries.
- `aggregate_median.py`: median aggregation across simulator judges.
- `prompts/`: simulator prompt text.

## Judges

The released config includes local `qwen3.6-27b-vllm`. The intended median
metric uses four judges: `gpt-5.4-nano`, `claude-haiku-4-5-20251001`,
`gemini-3.1-flash-lite`, and `qwen`.
