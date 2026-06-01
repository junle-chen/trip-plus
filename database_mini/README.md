# Database Mini

`database_mini/` is a two-sample subset of the full database. The full database
is too large to commit directly to GitHub, so this subset is included for table
inspection and smoke testing.

Included sample databases:

| Base sample DB | Matching query ID |
| --- | --- |
| `sample/en/id_0005` | `mt_single_0005` |
| `sample/en/id_0006` | `mt_single_0006` |

Run a mini smoke test from the repository root:

```bash
TEST_DATA=query/query_en/multiturn/query.json \
DATABASE_DIR=database_mini/sample/en \
RERUN_IDS=mt_single_0005 \
bash scripts/run_batch.sh qwen3.6-27b-vllm
```

Running the full released query set requires the complete
`database/sample/en/` package.
