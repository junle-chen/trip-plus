# Trip-Plus

Trip-Plus is a travel-planning benchmark for evaluating LLM agents with
database-backed tools, itinerary conversion, deterministic rule-based scoring,
and LLM-based user simulation.

## Quick Start

```bash
conda create -n trip-plus python=3.10 -y
conda activate trip-plus
pip install -r requirements.txt
cp env.example .env
```

Edit `.env` with the API keys and endpoints used by `models_config.json`.

This anonymous release includes the full query file at
`query/query_en/multiturn/query.json`. The complete database is too large to
commit directly to GitHub, so this repository tracks only two smoke-test sample
databases under `database/sample/en/` plus the city-level Beijing database
under `database/en/beijing/` for reviewers to inspect the source data format.
Due to repository size limits, Beijing is included as one representative city
from the 40-city source database.

If you need a local OpenAI-compatible vLLM server, start it before running
samples. The helper script takes the serving preset as its first argument:

```bash
bash scripts/vllm.sh qwen
```

The default local endpoint is:

```text
http://127.0.0.1:8000/v1
```

Run one mini sample:

```bash
TEST_DATA=query/query_en/multiturn/query.json \
DATABASE_DIR=database/sample/en \
RERUN_IDS=mt_single_0005 \
bash scripts/run_batch.sh qwen3.6-27b-vllm
```

Run the full benchmark after placing the full database package at
`database/`:

```bash
bash scripts/run_batch.sh
```

Run with a vLLM model alias:

```bash
bash scripts/run_batch.sh qwen3.6-27b-vllm
```

Run with Gemini Flash:

```bash
bash scripts/run_batch.sh gemini-3-flash-preview
```

You can also run several models with the same benchmark setup:

```bash
bash scripts/run_batch.sh "kimi deepseek doubao minimax"
```

Outputs are written under `result/` by default. A generated run path looks like
`result/<model_slug>/<run_name>_<timestamp>/<model>_en/`.

## Model Configuration

Model aliases are defined in `models_config.json`. Each entry specifies the
model name, base URL, and API-key environment variable.

## Running

`scripts/run_batch.sh` is the main batch entry point. It runs:

1. Planner inference (generate trajectories and final reports)
2. Report conversion (parse reports into structured itinerary JSON to evaluate)
3. Evaluation (score feasibility, constraints, and preferences)

The released query set is `query/query_en/multiturn/query.json`. The bundled
sample databases support smoke tests for `mt_single_0005` and
`mt_single_0006`. Running the full query set requires the full
`database/sample/en/` package.

Run the full benchmark:

```bash
bash scripts/run_batch.sh qwen3.6-27b-vllm
```

Run a subset after the full-run command is working:

```bash
RERUN_IDS=mt_single_0007_turn_0-mt_single_0007_turn_3 \
TEST_DATA=query/query_en/multiturn/query.json \
DATABASE_DIR=database/sample/en \
bash scripts/run_batch.sh qwen3.6-27b-vllm
```

Run the bundled smoke-test samples:

```bash
TEST_DATA=query/query_en/multiturn/query.json \
DATABASE_DIR=database/sample/en \
RERUN_IDS=mt_single_0005 \
bash scripts/run_batch.sh qwen3.6-27b-vllm
```

Resume from conversion:

```bash
START_FROM=conversion \
OUTPUT_DIR=result/example_run \
bash scripts/run_batch.sh qwen3.6-27b-vllm
```

Resume from evaluation:

```bash
START_FROM=evaluation \
OUTPUT_DIR=result/example_run \
bash scripts/run_batch.sh qwen3.6-27b-vllm
```

The Python entry point is `run.py`:

```bash
python run.py \
  --model qwen3.6-27b-vllm \
  --test-data query/query_en/multiturn/query.json \
  --database-dir database/sample/en \
  --workers 2
```

Common options:

- `--start-from inference|conversion|evaluation`
- `--rerun-ids 0,5,10` or `--rerun-ids 0-10`
- `--output-dir result/example_run`
- `--conversion-model qwen3.6-27b-vllm`
- `--local-vllm-worker-cap <n>`

When `OUTPUT_DIR` or `--output-dir` is set, it is treated as an output root and
the runner still appends `<model>_en/` under it.

## Database Directories

Trip-Plus uses two database layouts for different stages:

- `database/sample/en/`: per-query sample databases used by planner tools,
  conversion, evaluation, and user simulation. This repository tracks only
  `id_0005` and `id_0006` for smoke tests; the remaining sample IDs stay
  ignored unless you provide the full database package locally.
- `database/en/`: city-level source database used by query generation. The
  release includes `database/en/beijing/` as one concrete city example with
  local POIs, weather, transportation, train, and flight tables; the full
  40-city source database remains external due to repository size limits. This
  is not the runtime `DATABASE_DIR` for released benchmark evaluation.

For query generation, `QUERY_CITY_DB_ROOT` points to the city-level source
database and `QUERY_OUTPUT_DB_ROOT` is where generated per-query sample
databases are written.

## Evaluation

The integrated run already performs conversion and evaluation. To rerun
evaluation for an existing full result directory:

```bash
START_FROM=evaluation \
OUTPUT_DIR=result/example_run \
bash scripts/run_batch.sh qwen3.6-27b-vllm
```

Evaluate a subset:

```bash
START_FROM=evaluation \
OUTPUT_DIR=result/example_run \
RERUN_IDS=mt_single_0007_turn_0-mt_single_0007_turn_3 \
bash scripts/run_batch.sh qwen3.6-27b-vllm
```

Convert existing reports manually:

```bash
python -m evaluation.conversion \
  --result-dir result/example_run/qwen3.6-27b-vllm_en \
  --model qwen3.6-27b-vllm \
  --database-dir database/sample/en \
  --query-file query/query_en/multiturn/query.json
```

Evaluation reads `converted_plans/`, query metadata, and the sample database.
It writes detailed scoring artifacts under `evaluation/` inside the result
directory.

## User Simulation

User simulation scores completed multi-turn plans with an LLM judge.
In the examples below, `--result-dir` points to the plans generated by
`qwen3.6-27b-vllm`; the simulator judges evaluate those plans.

Run all four simulator judges and aggregate the median:

```bash
python -m simulation.run_user_simulation \
  --all-judges \
  --result-dir result/example_run/qwen3.6-27b-vllm_en
```

Run one judge only, for debugging or backfilling:

```bash
python -m simulation.run_user_simulation \
  --simulator-model qwen \
  --result-dir result/example_run/qwen3.6-27b-vllm_en
```

The `--result-dir` above is the source planner run being judged. For example,
`result/example_run/qwen3.6-27b-vllm_en` means "evaluate plans generated by
`qwen3.6-27b-vllm`"; it is not the judge-model list.

The median simulation metric is computed from four judges:

- `gpt-5.4-nano`
- `claude-haiku-4-5-20251001`
- `gemini-3.1-flash-lite`
- `qwen`

If judge outputs were produced separately, aggregate them with:

```bash
python -m simulation.aggregate_median \
  --result-dir result/example_run/qwen3.6-27b-vllm_en \
  --judge gpt-5.4-nano \
  --judge claude-haiku-4-5-20251001 \
  --judge gemini-3.1-flash-lite \
  --judge qwen
```

## Query Generation

The released benchmark data is the cleaned multi-turn file:

```text
query/query_en/multiturn/query.json
```

`query/query_en/multiturn/query_raw.json` is kept as an audit copy before
metadata cleanup. Split views and single-turn intermediates are optional local
artifacts and are ignored by git in this release.

Regenerate multi-turn queries from source code if needed:

```bash
bash scripts/generate_multiturn_queries.sh
```

The multi-turn generator expects a single-turn input file. If you need to
rebuild from scratch, first generate that intermediate into a temporary
location:

```bash
QUERY_SKIP_LLM=true QUERY_COUNT=4 QUERY_DESTINATION_COVERAGE=off \
QUERY_OUTPUT=tmp/trip-plus-smoke/single/query.json \
QUERY_OUTPUT_DB_ROOT=tmp/trip-plus-smoke/sample/en \
QUERY_ROOT=tmp/trip-plus-smoke/single \
bash scripts/generate_single_queries.sh
```

Then pass the temporary single-turn file to the multi-turn generator.

## Outputs

A typical result directory looks like:

```text
result/example_run/qwen3.6-27b-vllm_en/
├── trajectories/
├── reports/
├── converted_plans/
└── evaluation/
```

- `trajectories/`: planner messages and tool calls.
- `reports/`: final text reports.
- `converted_plans/`: structured itinerary JSON.
- `evaluation/`: deterministic score details and summaries.
- `result/user_simulation/<judge>/qwen3_6_27b_vllm/`: standalone user-simulation artifacts.

## Repository Layout

```text
.
├── agent/             # Planner runtime, prompts, LLM calls, and tool orchestration
├── database/          # Smoke-test sample DBs plus one Beijing source-data example
├── evaluation/        # Conversion, hard/soft checks, scoring, and summaries
├── query/             # Released benchmark queries
├── query_generation/  # Query generation and sample database materialization
├── runner/            # Shared run configuration, ID handling, and reporting helpers
├── scripts/           # Shell entry points for runs, generation, and vLLM serving
├── simulation/        # LLM user simulation and median aggregation
├── tools/             # Travel tools backed by the local databases
├── util/              # Small shared utilities
├── models_config.json # Model aliases and endpoint configuration
├── env.example        # Environment variable template
├── requirements.txt   # Python dependencies
└── run.py             # Integrated pipeline entry point
```

Most top-level subdirectories include a local `README.md` with module-specific
notes.
