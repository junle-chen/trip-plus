#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MULTITURN_INPUT="${MULTITURN_INPUT:-query/query_en/single/query.json}"
MULTITURN_OUTPUT="${MULTITURN_OUTPUT:-query/query_en/multiturn/query.json}"
MULTITURN_QUERY_ROOT="${MULTITURN_QUERY_ROOT:-query/query_en/multiturn}"
MULTITURN_DATABASE_ROOT="${MULTITURN_DATABASE_ROOT:-database/sample/en}"
MULTITURN_GENERATION_LANGUAGE="${MULTITURN_GENERATION_LANGUAGE:-en}"
MULTITURN_COUNT="${MULTITURN_COUNT:-0}"
MULTITURN_SEED="${MULTITURN_SEED:-20260427}"
MULTITURN_REFRESH_ENVIRONMENT_FROM="${MULTITURN_REFRESH_ENVIRONMENT_FROM:-}"
MULTITURN_ENVIRONMENT_OUTPUT="${MULTITURN_ENVIRONMENT_OUTPUT:-}"
MULTITURN_LLM_RENDER_TURNS="${MULTITURN_LLM_RENDER_TURNS:-false}"
MULTITURN_TURN_RENDER_MODEL="${MULTITURN_TURN_RENDER_MODEL:-qwen3.6-27b-vllm}"
MULTITURN_TURN_RENDER_LANGUAGE="${MULTITURN_TURN_RENDER_LANGUAGE:-en}"
MULTITURN_TURN_RENDER_TEMPERATURE="${MULTITURN_TURN_RENDER_TEMPERATURE:-1.0}"
MULTITURN_TURN_RENDER_MAX_TOKENS="${MULTITURN_TURN_RENDER_MAX_TOKENS:-260}"
MULTITURN_TURN_RENDER_WORKERS="${MULTITURN_TURN_RENDER_WORKERS:-1}"

if [[ "$MULTITURN_GENERATION_LANGUAGE" != "en" ]]; then
  echo "Unsupported MULTITURN_GENERATION_LANGUAGE=$MULTITURN_GENERATION_LANGUAGE; this release only supports en." >&2
  exit 2
fi

echo "================================"
echo "Generating multi-turn queries"
echo "Input:        $MULTITURN_INPUT"
echo "Output:       $MULTITURN_OUTPUT"
echo "Grouped root: $MULTITURN_QUERY_ROOT"
echo "Database:     $MULTITURN_DATABASE_ROOT"
echo "Language:     $MULTITURN_GENERATION_LANGUAGE"
echo "Count:        $MULTITURN_COUNT (0 means all)"
echo "Turns:        type-specific fixed counts"
echo "Seed:         $MULTITURN_SEED"
if [[ -n "$MULTITURN_REFRESH_ENVIRONMENT_FROM" && -f "$MULTITURN_REFRESH_ENVIRONMENT_FROM" ]]; then
  echo "Env refresh:  $MULTITURN_REFRESH_ENVIRONMENT_FROM -> $MULTITURN_ENVIRONMENT_OUTPUT"
fi
echo "LLM render:   $MULTITURN_LLM_RENDER_TURNS"
if [[ "$MULTITURN_LLM_RENDER_TURNS" == "true" || "$MULTITURN_LLM_RENDER_TURNS" == "1" ]]; then
  echo "Render model: $MULTITURN_TURN_RENDER_MODEL"
  echo "Render lang:  $MULTITURN_TURN_RENDER_LANGUAGE"
  echo "Temperature:  $MULTITURN_TURN_RENDER_TEMPERATURE"
  echo "Workers:      $MULTITURN_TURN_RENDER_WORKERS"
fi
echo "================================"

EXTRA_ARGS=()
if [[ "$MULTITURN_LLM_RENDER_TURNS" == "true" || "$MULTITURN_LLM_RENDER_TURNS" == "1" ]]; then
  EXTRA_ARGS+=(
    --llm-render-turns
    --turn-render-model "$MULTITURN_TURN_RENDER_MODEL"
    --turn-render-language "$MULTITURN_TURN_RENDER_LANGUAGE"
    --turn-render-temperature "$MULTITURN_TURN_RENDER_TEMPERATURE"
    --turn-render-max-tokens "$MULTITURN_TURN_RENDER_MAX_TOKENS"
    --turn-render-workers "$MULTITURN_TURN_RENDER_WORKERS"
  )
fi
if [[ -n "$MULTITURN_REFRESH_ENVIRONMENT_FROM" && -f "$MULTITURN_REFRESH_ENVIRONMENT_FROM" ]]; then
  EXTRA_ARGS+=(--refresh-environment-from "$MULTITURN_REFRESH_ENVIRONMENT_FROM")
fi
if [[ -n "$MULTITURN_ENVIRONMENT_OUTPUT" ]]; then
  EXTRA_ARGS+=(
    --environment-output "$MULTITURN_ENVIRONMENT_OUTPUT"
  )
fi
python -m query_generation.multiturn_query.generate \
  --input "$MULTITURN_INPUT" \
  --output "$MULTITURN_OUTPUT" \
  --query-root "$MULTITURN_QUERY_ROOT" \
  --database-root "$MULTITURN_DATABASE_ROOT" \
  --generation-language "$MULTITURN_GENERATION_LANGUAGE" \
  --count "$MULTITURN_COUNT" \
  --seed "$MULTITURN_SEED" \
  "${EXTRA_ARGS[@]}"
