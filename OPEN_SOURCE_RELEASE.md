# Open-source release scope

This release should expose the smallest English-only path needed to run the
released multi-turn planning benchmark and inspect the released query/dataset
contracts.

## Keep

Core runner:

- `run.py`
- `models_config.json`
- `env.example`

Agent planning:

- `agent/`

Travel tools and schemas:

- `tools/`
- `tools/tool_schema_en.json`

Evaluation:

- `evaluation/*.py`
- `simulation/prompts/traveler_experience_simulation.md`

Standalone user simulation:

- `simulation/`

Dataset and queries:

- `query/query_en/multiturn/query.json`
- `query/query_en/multiturn/query_raw.json`
- `database_mini/`
- full `database/sample/en/` as an external package if full benchmark
  execution is required

Reviewer scripts:

- `scripts/run.sh`
- `scripts/run_batch.sh`
- `scripts/vllm.sh`
- `scripts/generate_single_queries.sh`
- `scripts/generate_multiturn_queries.sh`
- helper files referenced by `scripts/vllm.sh` only if those model modes are released

Released query construction surface:

- `query_generation/README.md`
- `query_generation/city_database.py`
- `query_generation/common.py`
- `query_generation/sample_database.py`
- `query_generation/initial_query/`
- `query_generation/multiturn_query/`
- `query_generation/user_profile/`
- `query_generation/user_profile/observable_profiles_en.json`

The current release should treat `query/query_en/multiturn/query.json` as the
released benchmark input. The included generation code keeps the English
query-generation path for rebuilding benchmark inputs into a temporary or
ignored location. Legacy demos, Chinese profile data, preview dumps, full
database builders, crawlers, repair scripts, audit scripts, and internal
translation utilities are not included.

## Do not include

Historical outputs and experiment artifacts:

- `result/`
- `log/`
- `*_simulation/`
- `tmp/`
- `archive/`
- `Fig/`
- `output.png`

Chinese or migration-only assets:

- `query/query_zh/`
- `database/zh/`
- `database/database_generated_single_zh/`
- `tools/tool_schema_zh.json`
- `scripts/translate_queries_zh_to_en.py`
- `scripts/translate_query_tree_zh_to_en.py`
- `scripts/audit_multiturn_zh_en_alignment.py`
- `scripts/backfill_en_transport_hubs_from_zh.py`
- `query_generation/user_profile/observable_profiles.json`
- `query_generation/user_profile/profile_*`
- `query_generation/runner.py`
- `query_generation/single_turn.py`
- `query_generation/multi_turn.py`
- `query_generation/_multi_turn_core.py`
- `query_generation/user_profile/export_profile_derivations.py`
- `query_generation/user_profile/generate_queries_from_derivations.py`
- `query_generation/user_profile/multi_turn_roleplay.py`
- `query_generation/user_profile/run_demo_single_turn.py`

Research maintenance scripts not needed by reviewers:

- `scripts/audit_*.py`
- `scripts/repair_*.py`
- `scripts/normalize_*.py`
- `scripts/canonicalize_*.py`
- `scripts/rerun_*.sh`
- `scripts/utrip_*.sh`
- `evaluation/prompts/travel_feel_judge.md`
- `evaluation/resources/dedupe_price_aliases.json`

Paper and internal docs:

- `paper/`
- `doc/` except a concise release README if needed

The release path does not depend on database builders or crawlers. Generated
query sample-database materialization lives in `query_generation/sample_database.py`.

## Recommended reviewer commands

Start a local vLLM server:

```bash
bash scripts/vllm.sh
```

Run a small multi-turn subset:

```bash
RERUN_IDS=mt_single_0007_turn_0-mt_single_0007_turn_3 \
TEST_DATA=query/query_en/multiturn/query.json \
DATABASE_DIR=database/sample/en \
LANGUAGE=en \
bash scripts/run_batch.sh
```

Run the default configured benchmark:

```bash
LANGUAGE=en bash scripts/run_batch.sh
```
