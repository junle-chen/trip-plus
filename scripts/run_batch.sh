#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================================================
# Default English multi-turn benchmark launcher
# ============================================================================
# Usage:
#   bash scripts/run_batch.sh qwen3.6-27b-vllm
#   bash scripts/run_batch.sh kimi
#   bash scripts/run_batch.sh "kimi deepseek doubao minimax"
#   bash scripts/run_batch.sh deepseek-v4-pro
#   bash scripts/run_batch.sh "gpt5.4-mini gpt5.4 gemini3-flash"
#   bash scripts/run_batch.sh "Qwen/Qwen3.6-27B Qwen/Qwen3.6-35B-A3B"
#   bash scripts/run_batch.sh gemma26
#   bash scripts/run_batch.sh glm5.1
#   bash scripts/run_batch.sh hy3
#   bash scripts/run_batch.sh "qwen3.6-27b-vllm glm4"
#   bash scripts/run_batch.sh glm4
# Advanced parameters remain available in scripts/run.sh.

if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/run_batch.sh [model-or-quoted-model-list]"
  echo "Example: bash scripts/run_batch.sh qwen3.6-27b-vllm"
  echo "Example: bash scripts/run_batch.sh kimi"
  echo "Example: bash scripts/run_batch.sh \"kimi deepseek doubao minimax\""
  echo "Example: bash scripts/run_batch.sh deepseek-v4-pro"
  echo "Example: bash scripts/run_batch.sh \"gpt5.4-mini gpt5.4 gemini3-flash\""
  echo "Example: bash scripts/run_batch.sh \"Qwen/Qwen3.6-27B Qwen/Qwen3.6-35B-A3B\""
  echo "Example: bash scripts/run_batch.sh gemma26"
  echo "Example: bash scripts/run_batch.sh glm5.1"
  echo "Example: bash scripts/run_batch.sh hy3"
  echo "Example: bash scripts/run_batch.sh glm4"
  echo "Example: bash scripts/run_batch.sh \"qwen3.6-27b-vllm glm4\""
  exit 2
fi

normalize_model_alias() {
  case "$1" in
    kimi|kimi-k2|kimi-k2.6)
      printf '%s\n' "kimi-k2.6"
      ;;
    deepseek|deepseek-v3|deepseek-v3.2|ds|ds-v3|ds-v3.2)
      printf '%s\n' "deepseek-v3.2"
      ;;
    deepseek-v4|deepseek-v4-pro|deepseekv4|deepseekv4pro|ds-v4|ds-v4-pro)
      printf '%s\n' "deepseek-v4-pro"
      ;;
    doubao|doubao-pro|doubao-seed|doubao-seed-2.0|doubao-seed-2.0-pro)
      printf '%s\n' "doubao-seed-2.0-pro"
      ;;
    minimax|minimax-m2|minimax-m2.7)
      printf '%s\n' "minimax-m2.7"
      ;;
    gpt54mini|gpt5.4mini|gpt5.4-mini|gpt-5.4-mini)
      printf '%s\n' "gpt-5.4-mini"
      ;;
    gpt54|gpt5.4|gpt-5.4)
      printf '%s\n' "gpt-5.4"
      ;;
    gemini3flash|gemini3-flash|gemini-3-flash|gemini_3_flash|gemini-3-flash-preview)
      printf '%s\n' "gemini-3-flash-preview"
      ;;
    glm51|glm5.1|glm-5.1|glm5|glm-5)
      printf '%s\n' "glm-5.1"
      ;;
    hy3|hy3-preview|hunyuan3|hunyuan-3|tencent-hy3|tencent/Hy3-preview)
      printf '%s\n' "hy3-preview"
      ;;
    qwen36-27b|qwen3.6-27b|qwen3.6-27b-vllm|Qwen/Qwen3.6-27B)
      printf '%s\n' "qwen3.6-27b-vllm"
      ;;
    qwen36-35b-a3b|qwen3.6-35b-a3b|qwen35ba3b|qwen3.6-35b-a3b-vllm|Qwen/Qwen3.6-35B-A3B)
      printf '%s\n' "qwen3.6-35b-a3b-vllm"
      ;;
    qwen36|qwen36-8010|qwen3.6-27b-8010|qwen3.6-27b-vllm-8010)
      printf '%s\n' "qwen3.6-27b-vllm-8010"
      ;;
    qwen35-9b|qwen3.5-9b|qwen9b|qwen3.5-9b-vllm|Qwen/Qwen3.5-9B)
      printf '%s\n' "qwen3.5-9b-vllm"
      ;;
    qwen35-27b|qwen3.5-27b|qwen27b|qwen3.5-27b-vllm|Qwen/Qwen3.5-27B)
      printf '%s\n' "qwen3.5-27b-vllm"
      ;;
    qwen-a10b|qwen122a10b|qwen35a3b|qwen3.5-122b-a10b-fp8|qwen3.5-122b-a10b-fp8-vllm|Qwen/Qwen3.5-122B-A10B-FP8)
      printf '%s\n' "qwen3.5-122b-a10b-fp8-vllm"
      ;;
    gemma26|gemma4-26b|gemma-4-26b|gemma-4-26b-a4b|gemma-4-26b-a4b-vllm|google/gemma-4-26B-A4B-it)
      printf '%s\n' "gemma-4-26b-a4b-vllm"
      ;;
    glm|glm4|glm-4|glm-4-32b|glm-4-32b-0414|zai-org/GLM-4-32B-0414)
      printf '%s\n' "glm-4-32b-0414-vllm"
      ;;
    *)
      printf '%s\n' "$1"
      ;;
  esac
}

model_result_slug() {
  local slug="${1%-vllm}"
  slug="${slug//-/_}"
  printf '%s' "$slug" | sed -E 's/([0-9]+)b($|_)/\1B\2/g'
}

RAW_MODEL="${1:-${MODEL:-qwen3.6-27b-vllm}}"
RAW_MODEL="${RAW_MODEL//gpt 5.4 mini/gpt-5.4-mini}"
RAW_MODEL="${RAW_MODEL//gpt 5.4-mini/gpt-5.4-mini}"
RAW_MODEL="${RAW_MODEL//gpt 5.4/gpt-5.4}"
RAW_MODEL="${RAW_MODEL//glm 5.1/glm-5.1}"
RAW_MODEL="${RAW_MODEL//glm 5/glm-5}"
read -ra RAW_MODELS <<< "$RAW_MODEL"
NORMALIZED_MODELS=()
for MODEL_NAME in "${RAW_MODELS[@]}"; do
  NORMALIZED_MODELS+=("$(normalize_model_alias "$MODEL_NAME")")
done
if [[ ${#NORMALIZED_MODELS[@]} -eq 0 ]]; then
  NORMALIZED_MODELS=("qwen3.6-27b-vllm")
fi
MODEL="${NORMALIZED_MODELS[*]}"
FIRST_MODEL="${NORMALIZED_MODELS[0]}"
INFERENCE_MODEL="$MODEL"
CONVERSION_MODEL="${CONVERSION_MODEL:-}"
EVALUATION_MODEL="${EVALUATION_MODEL:-$FIRST_MODEL}"
LLM_USER_SIMULATOR_MODEL=""

LANGUAGE="en"
TEST_DATA="${TEST_DATA:-query/query_en/multiturn/query.json}"
DATABASE_DIR="${DATABASE_DIR:-database/sample/en}"
RERUN_IDS="${RERUN_IDS:-}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
if [[ -z "${RUN_NAME:-}" ]]; then
  case "$TEST_DATA" in
    query/query_*/single/query.json) RUN_NAME="query_en_single" ;;
    query/query_*/multiturn/query.json) RUN_NAME="query_en_multiturn" ;;
    *) RUN_NAME="$(basename "${TEST_DATA%.json}")" ;;
  esac
fi

OUTPUT_DIR_EXPLICIT=false
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR_EXPLICIT=true
fi

MODEL_SLUG="$(model_result_slug "$FIRST_MODEL")"
OUTPUT_DIR_TEMPLATE=""
if [[ "$OUTPUT_DIR_EXPLICIT" == true ]]; then
  OUTPUT_DIR_ROOTS=("$OUTPUT_DIR")
else
  OUTPUT_DIR="result/${MODEL_SLUG}/${RUN_NAME}_${STAMP}"
  OUTPUT_DIR_ROOTS=()
  if [[ ${#NORMALIZED_MODELS[@]} -gt 1 ]]; then
    OUTPUT_DIR_TEMPLATE="result/{model_slug}/${RUN_NAME}_${STAMP}"
    for MODEL_NAME in "${NORMALIZED_MODELS[@]}"; do
      OUTPUT_DIR_ROOTS+=("result/$(model_result_slug "$MODEL_NAME")/${RUN_NAME}_${STAMP}")
    done
  else
    OUTPUT_DIR_ROOTS=("$OUTPUT_DIR")
  fi
fi

WORKERS="${WORKERS:-10}"
LOCAL_VLLM_WORKER_CAP="${LOCAL_VLLM_WORKER_CAP:-4}"
MAX_LLM_CALLS="${MAX_LLM_CALLS:-100}"

# Optional controls.
START_FROM="${START_FROM:-inference}"  # inference | conversion | evaluation
VERBOSE="${VERBOSE:-true}"
mkdir -p result
for OUTPUT_ROOT in "${OUTPUT_DIR_ROOTS[@]}"; do
  mkdir -p "$OUTPUT_ROOT"
done

TEST_DATA_FOR_QUERY_CHECK="$TEST_DATA" python - <<'PY'
import json
import os
from pathlib import Path


path = Path(os.environ["TEST_DATA_FOR_QUERY_CHECK"])
records = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(records, list):
    raise SystemExit(f"bad query file {path}: top-level JSON is not a list")
ids = [record.get("id") for record in records if isinstance(record, dict)]
if len(ids) != len(records) or len(set(ids)) != len(ids):
    raise SystemExit(f"bad query file {path}: records={len(records)} unique_ids={len(set(ids))}")
if not ids:
    raise SystemExit(f"bad query file {path}: no records")
print(f"Using existing {path}: records={len(records)} first={ids[0]} last={ids[-1]}")
PY

# ============================================================================
# Bridge simple variables into the full runner.
# ============================================================================

export BENCHMARK_MODEL="$MODEL"
export BENCHMARK_INFERENCE_MODEL="$INFERENCE_MODEL"
export BENCHMARK_EVALUATION_MODEL="$EVALUATION_MODEL"
if [[ -n "$CONVERSION_MODEL" ]]; then
  export BENCHMARK_CONVERSION_MODEL="$CONVERSION_MODEL"
else
  unset BENCHMARK_CONVERSION_MODEL
fi
unset BENCHMARK_LLM_USER_SIMULATOR_MODEL
unset BENCHMARK_DISABLE_LLM_USER_SIMULATOR
unset BENCHMARK_REQUIRE_LLM_USER_SIMULATOR

export BENCHMARK_TEST_DATA="$TEST_DATA"
export BENCHMARK_DATABASE_DIR="$DATABASE_DIR"
if [[ -n "$OUTPUT_DIR_TEMPLATE" ]]; then
  unset BENCHMARK_OUTPUT_DIR
  export BENCHMARK_OUTPUT_DIR_TEMPLATE="$OUTPUT_DIR_TEMPLATE"
else
  export BENCHMARK_OUTPUT_DIR="$OUTPUT_DIR"
  unset BENCHMARK_OUTPUT_DIR_TEMPLATE
fi
export BENCHMARK_WORKERS="$WORKERS"
export BENCHMARK_LOCAL_VLLM_WORKER_CAP="$LOCAL_VLLM_WORKER_CAP"
export BENCHMARK_MAX_LLM_CALLS="$MAX_LLM_CALLS"
export BENCHMARK_START_FROM="$START_FROM"
export BENCHMARK_VERBOSE="$VERBOSE"

if [[ -n "$RERUN_IDS" ]]; then
    export BENCHMARK_RERUN_IDS="$RERUN_IDS"
else
    unset BENCHMARK_RERUN_IDS
fi

python - <<'PY'
import sys

try:
    import pandas  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "Current Python cannot import pandas, so travel CSV tools will fail with "
        f"tool_init_failed. Python: {sys.executable}. Error: {exc}. "
        "Run the benchmark from an environment with pandas, for example: "
        "conda run -n trip-plus bash scripts/run_batch.sh <model>."
    )
PY

echo "================================"
echo "Trip-Plus run"
echo "Model:                    $MODEL"
echo "Inference model(s):       $INFERENCE_MODEL"
echo "Conversion model:         ${CONVERSION_MODEL:-per inference model}"
echo "Evaluation model:         $EVALUATION_MODEL"
echo "LLM user simulator:       standalone only"
if [[ ${#OUTPUT_DIR_ROOTS[@]} -eq 1 ]]; then
  echo "Output dir:               ${OUTPUT_DIR_ROOTS[0]}"
else
  echo "Output dirs:"
  for OUTPUT_ROOT in "${OUTPUT_DIR_ROOTS[@]}"; do
    echo "  - $OUTPUT_ROOT"
  done
fi
echo "Test data:                $TEST_DATA"
echo "Database dir:             $DATABASE_DIR"
echo "Language:                 $LANGUAGE"
echo "Rerun IDs:                ${RERUN_IDS:-all}"
echo "Workers:                  $WORKERS"
echo "Local vLLM worker cap:    $LOCAL_VLLM_WORKER_CAP"
echo "Max LLM calls:            $MAX_LLM_CALLS"
echo "Start from:               $START_FROM"
echo "================================"

RUN_STATUS=0
bash "$SCRIPT_DIR/run.sh" || RUN_STATUS=$?

RUN_BATCH_OUTPUT_DIRS="$(IFS=:; printf '%s' "${OUTPUT_DIR_ROOTS[*]}")" python - <<'PY'
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _fmt_score(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{numeric:.4f} ({numeric:.2%})"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _iter_summary_files(output_dir: Path) -> Iterable[Path]:
    for evaluation_dir in sorted(output_dir.glob("*/evaluation")):
        for name in ("_summary.json", "multiturn_summary.json", "evaluation_summary.json"):
            path = evaluation_dir / name
            if path.exists():
                yield path
                break


def _print_multiturn_summary(model_dir: Path, data: Dict[str, Any], path: Path) -> None:
    metric_groups = data.get("metric_groups") or {}
    all_responses = metric_groups.get("all_responses") or {}
    successful_plans = metric_groups.get("successful_plans") or {}
    diagnostics = data.get("diagnostics") or {}
    metrics = data.get("metrics") or {}

    plan_count = successful_plans.get("evaluated_turn_count")
    final_plan_count = successful_plans.get("llm_user_simulation_denominator")
    scored_count = (successful_plans.get("score_counts") or {}).get("llm_user_simulation_score")
    failed_count = diagnostics.get("llm_user_simulation_failed_count", 0)
    score_successful = successful_plans.get("llm_user_simulation_score")
    score_all = all_responses.get("llm_user_simulation_score")
    score_non_null = metrics.get("llm_user_simulation_score")

    print(f"Model: {model_dir.name}")
    print(f"  Summary: {path}")
    print(f"  Evaluated plan turns: {plan_count if plan_count is not None else 'N/A'}")
    print(f"  Final successful plans for simulation: {final_plan_count if final_plan_count is not None else 'N/A'}")
    print(f"  LLM user simulation scored: {scored_count if scored_count is not None else 0}")
    print(f"  LLM user simulation failed: {failed_count}")
    print(f"  Score over scored successful final plans: {_fmt_score(score_successful)}")
    print(f"  Score over scored eligible final turns: {_fmt_score(score_all)}")
    print(f"  Mean over non-null simulator scores: {_fmt_score(score_non_null)}")


def _print_single_summary(model_dir: Path, data: Dict[str, Any], path: Path) -> None:
    metrics = data.get("metrics") or {}
    diagnostics = data.get("diagnostics") or {}
    plan_count = diagnostics.get("delivered_itinerary_count")
    if plan_count is None:
        plan_count = data.get("evaluation_success_count")
    scored_count = diagnostics.get("llm_user_simulation_success_count", 0)
    failed_count = diagnostics.get("llm_user_simulation_failed_count", 0)

    print(f"Model: {model_dir.name}")
    print(f"  Summary: {path}")
    print(f"  Evaluated plans: {plan_count if plan_count is not None else 'N/A'}")
    print(f"  LLM user simulation scored: {scored_count}")
    print(f"  LLM user simulation failed: {failed_count}")
    print(f"  LLM user simulation score: {_fmt_score(metrics.get('llm_user_simulation_score'))}")


output_dirs = [
    Path(item).resolve()
    for item in os.environ.get("RUN_BATCH_OUTPUT_DIRS", "").split(os.pathsep)
    if item
]
summary_files = []
for output_dir in output_dirs:
    summary_files.extend(_iter_summary_files(output_dir))

print("")
print("================================")
print("LLM USER SIMULATION SUMMARY")
if len(output_dirs) == 1:
    print(f"Output dir: {output_dirs[0]}")
else:
    print("Output dirs:")
    for output_dir in output_dirs:
        print(f"  - {output_dir}")

if not summary_files:
    print("No evaluation summary found yet.")
else:
    for index, summary_path in enumerate(summary_files):
        if index:
            print("")
        data = _load_json(summary_path)
        if data is None:
            print(f"Skipped unreadable summary: {summary_path}")
            continue
        model_dir = summary_path.parent.parent
        if summary_path.name in {"_summary.json", "multiturn_summary.json"}:
            _print_multiturn_summary(model_dir, data, summary_path)
        else:
            _print_single_summary(model_dir, data, summary_path)
print("================================")
PY

exit "$RUN_STATUS"
