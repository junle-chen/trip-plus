# Scripts

`scripts/` contains reviewer-facing shell entry points and vLLM parser helpers.

## Runtime

- `run_batch.sh`: default benchmark launcher.
- `run.sh`: lower-level launcher used by `run_batch.sh`.
- `vllm.sh`: starts a local OpenAI-compatible vLLM endpoint.

Run a small multi-turn subset:

```bash
RERUN_IDS=mt_single_0007_turn_0-mt_single_0007_turn_3 \
TEST_DATA=query/query_en/multiturn/query.json \
DATABASE_DIR=database/sample/en \
bash scripts/run_batch.sh
```

## Query Generation

- `generate_multiturn_queries.sh`: English multi-turn query generator.
- `generate_single_queries.sh`: optional single-turn query and sample-database
  generator used when rebuilding the multi-turn source input from scratch.

The released query surface is multi-turn only:

```text
query/query_en/multiturn/query.json
query/query_en/multiturn/query_raw.json
```

Single-turn outputs and split/group query views are ignored by git.

## vLLM Helpers

- `functiongemma_safe_tool_parser.py`
- `glm4_0414_tool_parser.py`
- `glm4_0414_tool_chat_template.jinja`

These are optional parser helpers referenced by `vllm.sh`.
