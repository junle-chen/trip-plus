# Query Files

`query/` contains released benchmark inputs consumed by `run.py`. This release
is multi-turn focused; treat these files as benchmark artifacts, not source
code.

## Layout

- `query_en/multiturn/query.json`: cleaned English multi-turn benchmark
  records used by default runs and evaluation.
- `query_en/multiturn/query_raw.json`: audit copy of the multi-turn records
  before metadata cleanup.

## Boundary

- Regenerate queries through `query_generation/`.
- Run benchmark queries through `run.py`.
- Keep model outputs under `result/`, not in this directory.
