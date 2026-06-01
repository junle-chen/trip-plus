# Utility Helpers

`util/` contains small, dependency-light helpers shared by runtime and
maintenance code.

This package is independent of the released query layout. Keep benchmark
schema handling in `agent/`, `runner/`, or `evaluation/`.

## Files

- `env.py`: `.env` loading.
- `io.py`: lightweight file I/O helpers.
- `numeric.py`: numeric parsing helpers.

## Boundary

- Keep this package generic and low-level.
- Put domain logic in `agent/`, `tools/`, `evaluation/`, or
  `query_generation/`.
