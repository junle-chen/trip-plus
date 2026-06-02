# Database

`database/` contains the fixed English travel evidence used by planner tools,
query generation, and evaluation.

## Layout

- `en/`: city-level English database and indexes.
- `sample/en/`: per-query sample databases materialized by query generation.

The public reviewer release keeps `database/en/beijing/` as a city-level source
database example. Other city folders under `database/en/` remain external unless
you provide the full database package locally.

Each sample database contains the evidence visible to tools and evaluators for
one query. Runtime outputs should not be written here.

The current release runs the cleaned multi-turn query file by default. Its
records resolve evidence from `database/sample/en/` through the sample database
resolver; ignored single-turn query files are not required at runtime.

## Boundary

- Keep raw or generated travel evidence here.
- Put database readers and lookup behavior in `tools/`.
- Put query construction and sample materialization in `query_generation/`.
- Put scoring rules in `evaluation/`.
