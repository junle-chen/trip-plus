# Agent Runtime

`agent/` contains the planner runtime used by `run.py`. It prepares benchmark
records, calls an OpenAI-compatible model, executes travel tools, and writes
raw trajectories plus final reports.

## Main Entry

Use the repository runner rather than importing this package directly:

```bash
python run.py --test-data query/query_en/multiturn/query.json
```

## Files

- `batch_runner.py`: runs benchmark records in batch, including multi-turn
  turn IDs such as `mt_single_0007_turn_2`.
- `planner.py`: ReAct-style loop for one planning request.
- `tooling.py`: tool-call parsing, argument normalization, duplicate-call keys,
  and compact tool-result views.
- `input_processing.py`: converts query records into agent-ready messages and
  resolves sample metadata.
- `output_parser.py`: extracts `<plan>`, `<clarification>`, and
  `<no_solution>` responses.
- `prompts.py`: English planner and conversion prompts.
- `call_llm.py`: model-provider wrapper for OpenAI-compatible chat APIs.

## Boundary

- Planning behavior belongs here.
- Travel evidence lookup belongs in `tools/`.
- Scoring belongs in `evaluation/`.
- Query construction belongs in `query_generation/`.
