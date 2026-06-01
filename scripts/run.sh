#!/bin/bash

# Advanced runner. For routine experiments, edit scripts/run_batch.sh and let it
# pass the small set of important controls into this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Model settings. Routine runs should set these through scripts/run_batch.sh.
MODEL="${BENCHMARK_MODEL:-qwen3.6-27b-vllm}"
INFERENCE_MODEL="${BENCHMARK_INFERENCE_MODEL:-$MODEL}"
CONVERSION_MODEL="${BENCHMARK_CONVERSION_MODEL:-$INFERENCE_MODEL}"
EVALUATION_MODEL="${BENCHMARK_EVALUATION_MODEL:-$INFERENCE_MODEL}"
LLM_USER_SIMULATOR_MODEL=""

LANGUAGE="en"

# Parallel workers
# Can be overridden by BENCHMARK_WORKERS environment variable
WORKERS="${BENCHMARK_WORKERS:-10}"

# Local vLLM worker cap for inference/conversion. Set to 0 to disable.
# Can be overridden by BENCHMARK_LOCAL_VLLM_WORKER_CAP environment variable
LOCAL_VLLM_WORKER_CAP="${BENCHMARK_LOCAL_VLLM_WORKER_CAP:-10}"

# Max LLM calls per task
# Can be overridden by BENCHMARK_MAX_LLM_CALLS environment variable
MAX_LLM_CALLS="${BENCHMARK_MAX_LLM_CALLS:-100}"

# ReAct tool budget. These remain overridable for short debugging runs.
export TOOL_CALL_HARD_LIMIT="${TOOL_CALL_HARD_LIMIT:-64}"
export TOOL_BUDGET_WARNING_THRESHOLD="${TOOL_BUDGET_WARNING_THRESHOLD:-45}"

# Start point: inference, conversion, evaluation
# Can be overridden by BENCHMARK_START_FROM environment variable
START_FROM="${BENCHMARK_START_FROM:-inference}"

# Output directory
# Can be overridden by BENCHMARK_OUTPUT_DIR environment variable
OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-}"
OUTPUT_DIR_TEMPLATE="${BENCHMARK_OUTPUT_DIR_TEMPLATE:-}"

# Test data path
# Can be overridden by BENCHMARK_TEST_DATA environment variable
TEST_DATA="${BENCHMARK_TEST_DATA:-}"

# Rerun specific IDs
# Can be overridden by BENCHMARK_RERUN_IDS environment variable
RERUN_IDS="${BENCHMARK_RERUN_IDS:-}"

# Database mode / path
# BENCHMARK_DATABASE_MODE: id | city
# BENCHMARK_DATABASE_DIR: explicit database directory, higher priority than mode
DATABASE_MODE="${BENCHMARK_DATABASE_MODE:-id}"
DATABASE_DIR="${BENCHMARK_DATABASE_DIR:-}"

# Verbose mode
# Can be overridden by BENCHMARK_VERBOSE environment variable
VERBOSE="${BENCHMARK_VERBOSE:-false}"

# Debug mode
# Can be overridden by BENCHMARK_DEBUG environment variable
DEBUG="${BENCHMARK_DEBUG:-false}"
PERSIST_LOG_DIR="${BENCHMARK_PERSIST_LOG_DIR:-}"
PERSIST_LOG_TIME="${BENCHMARK_PERSIST_LOG_TIME:-$(date +%Y%m%d_%H%M%S)}"

model_result_slug() {
    local slug="${1%-vllm}"
    slug="${slug//-/_}"
    printf '%s' "$slug" | sed -E 's/([0-9]+)b($|_)/\1B\2/g'
}

model_output_root() {
    local model_name="$1"
    if [ -n "$OUTPUT_DIR" ]; then
        printf '%s\n' "$OUTPUT_DIR"
        return
    fi
    if [ -n "$OUTPUT_DIR_TEMPLATE" ]; then
        local model_slug
        local output_root
        model_slug="$(model_result_slug "$model_name")"
        output_root="$OUTPUT_DIR_TEMPLATE"
        output_root="${output_root//\{model_slug\}/$model_slug}"
        output_root="${output_root//\{model\}/$model_name}"
        printf '%s\n' "$output_root"
        return
    fi
    printf '\n'
}
               

read -ra MODELS <<< "$INFERENCE_MODEL"
TOTAL=${#MODELS[@]}

# Minimal vLLM health checks for models that the requested start stage can call.
case "$START_FROM" in
    inference)
        HEALTH_CHECK_MODELS="$INFERENCE_MODEL $CONVERSION_MODEL"
        ;;
    conversion)
        HEALTH_CHECK_MODELS="$CONVERSION_MODEL"
        ;;
    evaluation)
        HEALTH_CHECK_MODELS=""
        ;;
    *)
        HEALTH_CHECK_MODELS="$INFERENCE_MODEL $CONVERSION_MODEL"
        ;;
esac

if [[ "$HEALTH_CHECK_MODELS" == *"vllm"* ]]; then
    export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
    CONFIG_PATH="$PROJECT_ROOT/models_config.json"

    read -ra CHECK_MODELS <<< "$HEALTH_CHECK_MODELS"
    for MODEL_NAME in "${CHECK_MODELS[@]}"; do
        if [[ "$MODEL_NAME" != *"vllm"* ]]; then
            continue
        fi

        ENDPOINT=$(python3 -c "import json; from pathlib import Path; cfg=json.loads(Path('$CONFIG_PATH').read_text(encoding='utf-8')); base=cfg.get('models',{}).get('$MODEL_NAME',{}).get('base_url','').rstrip('/'); print((base + '/models') if base else '')")

        if [ -z "$ENDPOINT" ]; then
            echo "❌ Missing base_url for model '$MODEL_NAME' in $CONFIG_PATH"
            exit 1
        fi

        CURL_ARGS=(-fsS --max-time 5)
        if [[ "$ENDPOINT" == http://127.0.0.1:* || "$ENDPOINT" == http://localhost:* ]]; then
            CURL_ARGS+=(--noproxy '*')
            export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
            export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"
        fi

        if ! curl "${CURL_ARGS[@]}" "$ENDPOINT" >/dev/null 2>&1; then
            echo "❌ vLLM endpoint unreachable for '$MODEL_NAME': $ENDPOINT"
            echo "   Example:"
            case "$MODEL_NAME" in
                qwen3.6-27b-vllm)
                    echo "   bash scripts/vllm.sh qwen"
                    ;;
                qwen3.6-27b-vllm-8010)
                    echo "   bash scripts/vllm.sh qwen36"
                    ;;
                qwen3.6-35b-a3b-vllm)
                    echo "   bash scripts/vllm.sh qwen3.6-35b-a3b"
                    ;;
                qwen3.5-9b-vllm)
                    echo "   bash scripts/vllm.sh qwen3.5-9b"
                    ;;
                qwen3.5-27b-vllm)
                    echo "   bash scripts/vllm.sh qwen3.5-27b"
                    ;;
                qwen3.5-122b-a10b-fp8-vllm)
                    echo "   bash scripts/vllm.sh qwen-a10b"
                    ;;
                gemma-4-31b-vllm)
                    echo "   bash scripts/vllm.sh gemma"
                    ;;
                gemma-4-26b-a4b-vllm)
                    echo "   bash scripts/vllm.sh gemma26"
                    ;;
                glm-4-32b-0414-vllm)
                    echo "   bash scripts/vllm.sh glm4"
                    ;;
                *)
                    echo "   bash scripts/vllm.sh"
                    ;;
            esac
            exit 1
        fi
    done
fi

LANGUAGE_DISPLAY="$LANGUAGE"

# ---------------- Concurrent Execution Mode ----------------
LOG_DIR=$(mktemp -d)

# Trap Ctrl+C signal, cleanup background processes and temp directory
trap "echo '🛑 Caught Ctrl+C, stopping all tasks...'; for pid in \${PIDS[@]}; do pkill -P \$pid 2>/dev/null; kill \$pid 2>/dev/null; done; pkill -P \$\$; rm -rf $LOG_DIR; exit 1" INT

declare -a PIDS
declare -A PID_TO_MODEL

echo "================================"
echo "🚀 Starting Concurrent Evaluation"
echo "Total Inference Models: ${#MODELS[@]}"
echo "Inference Model(s): $INFERENCE_MODEL"
echo "Conversion Model:   $CONVERSION_MODEL"
echo "Evaluation Model:   $EVALUATION_MODEL"
echo "LLM User Simulator: standalone only"
echo "Language: $LANGUAGE_DISPLAY"
echo "Workers: $WORKERS"
echo "Local vLLM Worker Cap: $LOCAL_VLLM_WORKER_CAP"
echo "Max LLM Calls: $MAX_LLM_CALLS"
echo "Tool Call Hard Limit: $TOOL_CALL_HARD_LIMIT"
echo "Tool Warning Threshold: $TOOL_BUDGET_WARNING_THRESHOLD"
echo "Start From: $START_FROM"
echo "Test Data: ${TEST_DATA:-auto}"
echo "Database Mode: $DATABASE_MODE"
echo "Database Dir: ${DATABASE_DIR:-auto}"
if [ -n "$OUTPUT_DIR" ]; then
    echo "Output Dir: $OUTPUT_DIR"
elif [ -n "$OUTPUT_DIR_TEMPLATE" ]; then
    echo "Output Dir Template: $OUTPUT_DIR_TEMPLATE"
else
    echo "Output Dir: auto"
fi
echo "================================"
echo ""

if [ -z "$DATABASE_DIR" ]; then
    if [ "$DATABASE_MODE" = "city" ]; then
        DATABASE_DIR="database"
    else
        DATABASE_DIR=""
    fi
fi

# ---------------- Pre-check missing IDs and filter completed models ----------------
declare -a MODELS_TO_RUN
declare -A MODEL_SKIP_REASON
declare -A MODEL_START_FROM  # Record which step each model should start from

if [ "$START_FROM" == "inference" ]; then
    # Pre-calculate target IDs for pre-check display to be dynamic
    EXPECTED_IDS_JSON=$(python3 -c "
import json, re, sys
from pathlib import Path
def parse_ids(id_str):
    if not id_str: return None
    ids = []
    for part in id_str.split(','):
        part = part.strip()
        if not part: continue
        if '-' in part:
            s, e = part.split('-', 1)
            sm = re.fullmatch(r'([A-Za-z_]+?)(\d+)', s)
            em = re.fullmatch(r'([A-Za-z_]+?)(\d+)', e)
            if sm and em and sm.group(1) == em.group(1):
                prefix = sm.group(1)
                width = max(len(sm.group(2)), len(em.group(2)))
                ids.extend(f'{prefix}{i:0{width}d}' for i in range(int(sm.group(2)), int(em.group(2)) + 1))
                continue
            try:
                start, end = int(s), int(e)
                ids.extend(str(i) for i in range(start, end + 1))
            except Exception:
                ids.append(part)
        else:
            ids.append(str(int(part)) if part.isdigit() else part)
    return sorted(set(ids))

test_data = '$TEST_DATA'
lang = '${LANGUAGE:-en}'
rerun_ids_str = '$BENCHMARK_RERUN_IDS'

rerun_ids = parse_ids(rerun_ids_str)
if rerun_ids is not None:
    print(json.dumps(rerun_ids))
else:
    if not test_data:
        test_data = f'query/query_{lang}/single/query.json'
    try:
        with open(test_data, 'r') as f:
            records = json.load(f)
            print(json.dumps([str(item.get('id', i)) for i, item in enumerate(records)]))
    except Exception:
        print(json.dumps([str(i) for i in range(120)]))
")
    TARGET_COUNT=$(echo "$EXPECTED_IDS_JSON" | python3 -c "import json, sys; print(len(json.load(sys.stdin)))")

    echo "🔍 Pre-check: Detecting status for $TARGET_COUNT target samples..."
    echo ""
    
    for MODEL_NAME in "${MODELS[@]}"; do
        MODEL_OUTPUT_DIR="$(model_output_root "$MODEL_NAME")"
        SHOULD_SKIP=false
        SKIP_REASON=""
        MODEL_START="inference"  # Default start from inference
        
            # Check specified language
            if [ -n "$MODEL_OUTPUT_DIR" ]; then
                REPORTS_DIR="$MODEL_OUTPUT_DIR/${MODEL_NAME}_${LANGUAGE}/reports"
                PLANS_DIR="$MODEL_OUTPUT_DIR/${MODEL_NAME}_${LANGUAGE}/converted_plans"
            else
                REPORTS_DIR="results/${MODEL_NAME}_${LANGUAGE}/reports"
                PLANS_DIR="results/${MODEL_NAME}_${LANGUAGE}/converted_plans"
            fi
            
            # Check reports
            if [ -d "$REPORTS_DIR" ]; then
                REPORTS_MISSING=$(python3 -c "
import json, re, sys
from pathlib import Path
expected = set(json.loads('$EXPECTED_IDS_JSON'))
existing = set()
for f in Path('$REPORTS_DIR').glob('id_*.txt'):
    name = f.stem
    existing.add(name[3:] if name.startswith('id_') else name)
for f in Path('$REPORTS_DIR').glob('*.txt'):
    name = f.stem
    existing.add(name[3:] if name.startswith('id_') else name)
print(len(expected - existing))
")
            else
                REPORTS_MISSING=$TARGET_COUNT
            fi
            
            # Check converted_plans
            if [ -d "$PLANS_DIR" ]; then
                PLANS_MISSING=$(python3 -c "
import json, re, sys
from pathlib import Path
expected = set(json.loads('$EXPECTED_IDS_JSON'))
existing = set()
for f in Path('$PLANS_DIR').glob('id_*_converted.json'):
    name = f.stem
    if name.startswith('id_'):
        name = name[3:]
    if name.endswith('_converted'):
        name = name[:-10]
    existing.add(name)
print(len(expected - existing))
")
            else
                PLANS_MISSING=$TARGET_COUNT
            fi
            
            # Display status
            if [ "$REPORTS_MISSING" -eq 0 ] && [ "$PLANS_MISSING" -eq 0 ]; then
                echo "  ✅ $MODEL_NAME: All complete (reports + plans)"
                SHOULD_SKIP=true
                SKIP_REASON="All reports and plans exist for language $LANGUAGE"
            elif [ "$REPORTS_MISSING" -eq 0 ] && [ "$PLANS_MISSING" -gt 0 ]; then
                echo "  📝 $MODEL_NAME: Reports ✅ | Plans: $PLANS_MISSING missing"
                MODEL_START="conversion"
            elif [ "$REPORTS_MISSING" -gt 0 ]; then
                echo "  📝 $MODEL_NAME: Reports: $REPORTS_MISSING missing | Plans: $PLANS_MISSING missing"
                MODEL_START="inference"
            fi
        
        # Decide whether to add to run list
        if [ "$SHOULD_SKIP" = true ]; then
            MODEL_SKIP_REASON[$MODEL_NAME]="$SKIP_REASON"
        else
            MODELS_TO_RUN+=("$MODEL_NAME")
            MODEL_START_FROM[$MODEL_NAME]="$MODEL_START"
        fi
    done
    echo ""
    
    # If there are skipped models, display information
    if [ ${#MODEL_SKIP_REASON[@]} -gt 0 ]; then
        echo "⏭️  Skipping models (already complete):"
        for MODEL_NAME in "${!MODEL_SKIP_REASON[@]}"; do
            echo "   - $MODEL_NAME: ${MODEL_SKIP_REASON[$MODEL_NAME]}"
        done
        echo ""
    fi
else
    # If not starting from inference, run all models
    MODELS_TO_RUN=("${MODELS[@]}")
    for MODEL_NAME in "${MODELS[@]}"; do
        MODEL_START_FROM[$MODEL_NAME]="$START_FROM"
    done
fi

# Update total count
TOTAL=${#MODELS_TO_RUN[@]}

if [ $TOTAL -eq 0 ]; then
    echo "✅ All models are already complete. Nothing to run!"
    exit 0
fi

# Count models starting from different steps
INFERENCE_COUNT=0
CONVERSION_COUNT=0
EVALUATION_COUNT=0
for MODEL_NAME in "${MODELS_TO_RUN[@]}"; do
    case "${MODEL_START_FROM[$MODEL_NAME]}" in
        conversion)
            CONVERSION_COUNT=$((CONVERSION_COUNT + 1))
            ;;
        evaluation)
            EVALUATION_COUNT=$((EVALUATION_COUNT + 1))
            ;;
        *)
            INFERENCE_COUNT=$((INFERENCE_COUNT + 1))
            ;;
    esac
done

echo "================================"
echo "📊 Will run $TOTAL models (skipped ${#MODEL_SKIP_REASON[@]})"
if [ $INFERENCE_COUNT -gt 0 ]; then
    echo "   - From inference: $INFERENCE_COUNT models"
fi
if [ $CONVERSION_COUNT -gt 0 ]; then
    echo "   - From conversion: $CONVERSION_COUNT models (reports complete, only convert plans)"
fi
if [ $EVALUATION_COUNT -gt 0 ]; then
    echo "   - From evaluation: $EVALUATION_COUNT models (converted plans complete, only evaluate)"
fi
echo "================================"
echo ""

# ---------------- Auto-fix permissions (simple method) ----------------
if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
    echo "🔧 Fixing permissions for output directory..."
    chmod -R u+rwX "$OUTPUT_DIR" 2>/dev/null || true
    echo "   ✅ Permissions fixed"
    echo ""
elif [ -n "$OUTPUT_DIR_TEMPLATE" ]; then
    for MODEL_NAME in "${MODELS_TO_RUN[@]}"; do
        MODEL_OUTPUT_DIR="$(model_output_root "$MODEL_NAME")"
        if [ -n "$MODEL_OUTPUT_DIR" ] && [ -d "$MODEL_OUTPUT_DIR" ]; then
            chmod -R u+rwX "$MODEL_OUTPUT_DIR" 2>/dev/null || true
        fi
    done
fi

# Start all models concurrently
for i in "${!MODELS_TO_RUN[@]}"; do
    MODEL_NAME="${MODELS_TO_RUN[$i]}"
    MODEL_LOG_DIR="log/${MODEL_NAME}"
    mkdir -p "$MODEL_LOG_DIR"
    LOG_FILE="$MODEL_LOG_DIR/run_${PERSIST_LOG_TIME}.log"
    
    # Get the starting step for this model
    MODEL_START="${MODEL_START_FROM[$MODEL_NAME]:-$START_FROM}"
    MODEL_OUTPUT_DIR="$(model_output_root "$MODEL_NAME")"
    
    echo "[STARTED] $MODEL_NAME (start-from: $MODEL_START) ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "   📝 Log: $LOG_FILE"
    if [ -n "$MODEL_OUTPUT_DIR" ]; then
        echo "   📁 Output root: $MODEL_OUTPUT_DIR"
    fi
    
    (
        python run.py \
            --model "$MODEL_NAME" \
            --inference-model "$MODEL_NAME" \
            --conversion-model "$CONVERSION_MODEL" \
            --evaluation-model "$EVALUATION_MODEL" \
            --workers $WORKERS \
            --local-vllm-worker-cap $LOCAL_VLLM_WORKER_CAP \
            --max-llm-calls $MAX_LLM_CALLS \
            --start-from "$MODEL_START" \
            ${TEST_DATA:+--test-data "$TEST_DATA"} \
            ${MODEL_OUTPUT_DIR:+--output-dir "$MODEL_OUTPUT_DIR"} \
            ${DATABASE_DIR:+--database-dir "$DATABASE_DIR"} \
            ${RERUN_IDS:+--rerun-ids "$RERUN_IDS"} \
            $([ "$VERBOSE" = "true" ] && echo "--verbose") \
            $([ "$DEBUG" = "true" ] && echo "--debug") > "$LOG_FILE" 2>&1
        
        echo $? > "$LOG_DIR/${MODEL_NAME}.exit"
    ) &
    
    PID=$!
    PIDS+=($PID)
    PID_TO_MODEL[$PID]="$MODEL_NAME"
done

echo ""
echo "All models started, waiting for completion..."
echo ""

# ---------------- Wait for tasks to complete and print in real-time ----------------
COMPLETED=0
SUCCESS=0
FAILED=0
FAILED_MODELS=()
declare -A PROCESSED_PIDS

while [ $COMPLETED -lt $TOTAL ]; do
    # Poll check each process status
    for PID in "${PIDS[@]}"; do
        # Skip already processed PIDs
        if [ -n "${PROCESSED_PIDS[$PID]}" ]; then
            continue
        fi
        
        # Check if process is still running
        if ! kill -0 $PID 2>/dev/null; then
            # Process ended, mark as processed
            PROCESSED_PIDS[$PID]=1
            
            MODEL_NAME="${PID_TO_MODEL[$PID]}"
            if [ -n "$MODEL_NAME" ]; then
                # Wait for process to fully end and get exit code
                wait $PID 2>/dev/null
                EXIT_CODE=$?
                
                # Also try to read from file (fallback)
                if [ -f "$LOG_DIR/${MODEL_NAME}.exit" ]; then
                    FILE_EXIT_CODE=$(cat "$LOG_DIR/${MODEL_NAME}.exit")
                    if [ -n "$FILE_EXIT_CODE" ]; then
                        EXIT_CODE=$FILE_EXIT_CODE
                    fi
                fi
                
                COMPLETED=$((COMPLETED + 1))
                
                if [ "$EXIT_CODE" -eq 0 ]; then
                    SUCCESS=$((SUCCESS + 1))
                    echo "[$COMPLETED/$TOTAL] ✅ $MODEL_NAME - Completed Successfully ($(date '+%H:%M:%S'))"
                    
                    # Extract summary from log if available
                    MODEL_LOG_DIR="log/${MODEL_NAME}"
                    LOG_FILE="$MODEL_LOG_DIR/run_${PERSIST_LOG_TIME}.log"
                    if [ -f "$LOG_FILE" ]; then
                        # Try to extract "Model 'xxx' | Language 'xxx' completed" line
                        COMPLETION_LINE=$(grep -E "Model.*Language.*completed" "$LOG_FILE" | tail -n 1)
                        if [ -n "$COMPLETION_LINE" ]; then
                            echo "   $COMPLETION_LINE"
                        fi
                    fi
                else
                    FAILED=$((FAILED + 1))
                    FAILED_MODELS+=("$MODEL_NAME")
                    echo "[$COMPLETED/$TOTAL] ❌ $MODEL_NAME - Failed (exit code: $EXIT_CODE) ($(date '+%H:%M:%S'))"
                    echo "   See log: $MODEL_LOG_DIR/run_${PERSIST_LOG_TIME}.log"
                fi
            fi
        fi
    done
    
    # Avoid high CPU usage, brief sleep
    if [ $COMPLETED -lt $TOTAL ]; then
        sleep 1
    fi
done

echo ""

# ---------------- Summary ----------------
echo "================================"
echo "📊 BATCH EVALUATION SUMMARY"
echo "Total: $TOTAL | Success: $SUCCESS | Failed: $FAILED"

# (Logs are now directly saved to log/<MODEL_NAME>/)
if [ -n "$PERSIST_LOG_DIR" ]; then
    echo "Note: PERSIST_LOG_DIR is set, but logs are now saved to log/<model_name>/"
fi

if [ $FAILED -gt 0 ]; then
    echo "Failed models: ${FAILED_MODELS[*]}"
    echo ""
    echo "Log directory: $LOG_DIR"
else
    # Clean up temp directory
    rm -rf $LOG_DIR
fi
echo "================================"
