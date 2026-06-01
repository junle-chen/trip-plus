#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

QUERY_COUNT="${QUERY_COUNT:-160}"
QUERY_MODEL="${QUERY_MODEL:-qwen3.6-27b-vllm}"
QUERY_RENDER_WORKERS="${QUERY_RENDER_WORKERS:-4}"
QUERY_SEED="${QUERY_SEED:-20260422}"
QUERY_CITY_DB_ROOT="${QUERY_CITY_DB_ROOT:-database/en}"
QUERY_OUTPUT="${QUERY_OUTPUT:-query/query_en/single/query.json}"
QUERY_OUTPUT_DB_ROOT="${QUERY_OUTPUT_DB_ROOT:-database/sample/en}"
QUERY_ROOT="${QUERY_ROOT:-query/query_en/single}"
QUERY_MIN_DAYS="${QUERY_MIN_DAYS:-2}"
QUERY_MAX_DAYS="${QUERY_MAX_DAYS:-7}"
QUERY_DATE_POLICY="${QUERY_DATE_POLICY:-curated}"
QUERY_SKIP_LLM="${QUERY_SKIP_LLM:-false}"
QUERY_NO_ROUTE_MANIFEST="${QUERY_NO_ROUTE_MANIFEST:-false}"
QUERY_DESTINATION_COVERAGE="${QUERY_DESTINATION_COVERAGE:-reachable}"
QUERY_MIN_DEPART_DATE="${QUERY_MIN_DEPART_DATE:-}"
QUERY_MAX_DEPART_DATE="${QUERY_MAX_DEPART_DATE:-}"
QUERY_CURATED_DATE_WINDOWS="${QUERY_CURATED_DATE_WINDOWS:-}"
QUERY_ROUTE_MANIFESTS="${QUERY_ROUTE_MANIFESTS:-}"

ARGS=(
  --count "$QUERY_COUNT"
  --seed "$QUERY_SEED"
  --model "$QUERY_MODEL"
  --city-db-root "$QUERY_CITY_DB_ROOT"
  --output "$QUERY_OUTPUT"
  --output-db-root "$QUERY_OUTPUT_DB_ROOT"
  --query-root "$QUERY_ROOT"
  --min-days "$QUERY_MIN_DAYS"
  --max-days "$QUERY_MAX_DAYS"
  --date-policy "$QUERY_DATE_POLICY"
  --destination-coverage "$QUERY_DESTINATION_COVERAGE"
  --render-workers "$QUERY_RENDER_WORKERS"
)

if [ -n "$QUERY_MIN_DEPART_DATE" ]; then
  ARGS+=(--min-depart-date "$QUERY_MIN_DEPART_DATE")
fi

if [ -n "$QUERY_MAX_DEPART_DATE" ]; then
  ARGS+=(--max-depart-date "$QUERY_MAX_DEPART_DATE")
fi

if [ "$QUERY_SKIP_LLM" = "true" ]; then
  ARGS+=(--skip-llm)
fi

if [ "$QUERY_NO_ROUTE_MANIFEST" = "true" ]; then
  ARGS+=(--no-route-manifest)
fi

if [ -n "$QUERY_CURATED_DATE_WINDOWS" ]; then
  read -ra WINDOWS <<< "$QUERY_CURATED_DATE_WINDOWS"
  for WINDOW in "${WINDOWS[@]}"; do
    ARGS+=(--curated-date-window "$WINDOW")
  done
fi

if [ -n "$QUERY_ROUTE_MANIFESTS" ]; then
  read -ra MANIFESTS <<< "$QUERY_ROUTE_MANIFESTS"
  for MANIFEST in "${MANIFESTS[@]}"; do
    ARGS+=(--route-manifest "$MANIFEST")
  done
fi

if [[ "$QUERY_SKIP_LLM" != "true" && "$QUERY_MODEL" == *"vllm"* ]]; then
  export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
  export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,0.0.0.0"
  export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost,0.0.0.0"
  ENDPOINT=$(python3 -c "import json; from pathlib import Path; cfg=json.loads(Path('models_config.json').read_text(encoding='utf-8')); base=cfg.get('models',{}).get('$QUERY_MODEL',{}).get('base_url','').rstrip('/'); print((base + '/models') if base else '')")
  if [ -z "$ENDPOINT" ]; then
    echo "Missing base_url for model '$QUERY_MODEL' in models_config.json"
    exit 1
  fi
  if ! curl --noproxy '*' -fsS --max-time 5 "$ENDPOINT" >/dev/null 2>&1; then
    echo "vLLM endpoint unreachable for '$QUERY_MODEL': $ENDPOINT"
    echo "Start it first, for example: bash scripts/vllm.sh"
    exit 1
  fi
fi

echo "================================"
echo "Generating single-turn queries"
echo "Model:          $QUERY_MODEL"
echo "Count:          $QUERY_COUNT"
echo "Render workers: $QUERY_RENDER_WORKERS"
echo "Date policy:    $QUERY_DATE_POLICY"
echo "Dest coverage:  $QUERY_DESTINATION_COVERAGE"
echo "Output:         $QUERY_OUTPUT"
echo "DB output:      $QUERY_OUTPUT_DB_ROOT"
echo "Grouped output: $QUERY_ROOT"
echo "================================"

python -m query_generation.initial_query.cli "${ARGS[@]}"
