# Runner Helpers

`runner/` contains small orchestration helpers used by the top-level `run.py`.
It does not contain planner behavior, tool implementations, query generation,
or scoring rules.

The default release path is multi-turn:

```text
query/query_en/multiturn/query.json
```

Split query views and single-turn intermediates are ignored local artifacts and
are not required by the runner.

## Files

- `config.py`: parses `run.py` arguments and resolves model, path, worker, and
  stage configuration.
- `ids.py`: handles sample IDs, multi-turn parent/turn IDs, missing-output
  detection, and rerun expansion.
- `reporting.py`: prints final summaries and writes multi-turn evaluation
  details.
- `__init__.py`: marks the package.

## Boundary

- Keep only runner orchestration support here.
- Put model planning in `agent/`.
- Put scoring in `evaluation/`.
- Put query construction in `query_generation/`.
- Put shared low-level helpers in `util/`.
