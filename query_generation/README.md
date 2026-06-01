# Query Generation

`query_generation/` builds English Trip-Plus query files and per-query sample
databases. It samples grounded trip frames from `database/en`, attaches traveler
profiles and hidden references, renders English user requests, and writes
benchmark-ready JSON.

The released benchmark is multi-turn focused. The checked-in runnable query is
`query/query_en/multiturn/query.json`; intermediate generation files should be
written to temporary or ignored paths.

## Main Entry

Use the shell wrappers from the repository root:

```bash
bash scripts/generate_multiturn_queries.sh
```

`generate_multiturn_queries.sh` expects a single-turn input file. To rebuild
from scratch, first create that input with `generate_single_queries.sh` into a
temporary or ignored path.

For a small deterministic smoke run:

```bash
QUERY_SKIP_LLM=true QUERY_COUNT=2 QUERY_DESTINATION_COVERAGE=off \
QUERY_OUTPUT=tmp/trip-plus-smoke/single/query.json \
QUERY_OUTPUT_DB_ROOT=tmp/trip-plus-smoke/sample/en \
QUERY_ROOT=tmp/trip-plus-smoke/single \
bash scripts/generate_single_queries.sh
```

## Layout

- `initial_query/`: single-turn query construction.
- `multiturn_query/`: deterministic expansion from single-turn records into
  multi-turn scenarios.
- `user_profile/`: observable profile sampling and hidden profile-rule
  derivation.
- `sample_database.py`: materializes per-query sample databases.
- `city_database.py`: loads city databases and route options.
- `common.py`: shared generation helpers.

## Outputs

Default generation targets in the generator code:

- `query/query_en/single/query.json`
- `query/query_en/multiturn/query.json`
- `database/sample/en/`

For this release, keep the published benchmark surface to:

- `query/query_en/multiturn/query.json`
- `query/query_en/multiturn/query_raw.json`

Use explicit output paths for smoke tests so released artifacts are not
overwritten.

## Boundary

- Query generation creates benchmark inputs.
- Planner execution belongs in `agent/` and `run.py`.
- Evaluation rules belong in `evaluation/`.
