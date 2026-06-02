#!/bin/bash

MODE=${1:-qwen}
DEFAULT_MAX_MODEL_LEN=${DEFAULT_MAX_MODEL_LEN:-32768}
MAX_MODEL_LEN_OVERRIDE=${MAX_MODEL_LEN:-}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.94}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-65536}
MAX_NUM_PARTIAL_PREFILLS=${MAX_NUM_PARTIAL_PREFILLS:-1}
MAX_LONG_PARTIAL_PREFILLS=${MAX_LONG_PARTIAL_PREFILLS:-1}
LONG_PREFILL_TOKEN_THRESHOLD=${LONG_PREFILL_TOKEN_THRESHOLD:-16384}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GEMMA26_MODEL_PATH="${GEMMA26_MODEL_PATH:-google/gemma-4-26B-A4B-it}"
GEMMA31_MODEL_PATH="${GEMMA31_MODEL_PATH:-google/gemma-4-31B-it}"

if [ "$MODE" = "qwen" ]; then
  MODEL_MAX_MODEL_LEN=${MAX_MODEL_LEN_OVERRIDE:-$DEFAULT_MAX_MODEL_LEN}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
  export CUDA_VISIBLE_DEVICES

  TP=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.6-27B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size "$TP" \
    --dtype bfloat16 \
    --max-model-len "$MODEL_MAX_MODEL_LEN" \
    --reasoning-parser qwen3 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --max-num-partial-prefills "$MAX_NUM_PARTIAL_PREFILLS" \
    --max-long-partial-prefills "$MAX_LONG_PARTIAL_PREFILLS" \
    --long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD" \
    --generation-config vllm \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder

elif [ "$MODE" = "qwen36" ] || [ "$MODE" = "qwen36-8010" ] || [ "$MODE" = "qwen3.6-27b-8010" ] || [ "$MODE" = "qwen3.6-27b-vllm-8010" ]; then
  MODEL_MAX_MODEL_LEN=${MAX_MODEL_LEN_OVERRIDE:-$DEFAULT_MAX_MODEL_LEN}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
  export CUDA_VISIBLE_DEVICES

  TP=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.6-27B \
    --host 0.0.0.0 \
    --port 8010 \
    --tensor-parallel-size "$TP" \
    --dtype bfloat16 \
    --max-model-len "$MODEL_MAX_MODEL_LEN" \
    --reasoning-parser qwen3 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --max-num-partial-prefills "$MAX_NUM_PARTIAL_PREFILLS" \
    --max-long-partial-prefills "$MAX_LONG_PARTIAL_PREFILLS" \
    --long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD" \
    --generation-config vllm \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder

elif [ "$MODE" = "qwen3.6-35b-a3b" ] || [ "$MODE" = "qwen3.6-35b-a3b-vllm" ] || [ "$MODE" = "qwen36-35b-a3b" ] || [ "$MODE" = "qwen35ba3b" ]; then
  MODEL_MAX_MODEL_LEN=${MAX_MODEL_LEN_OVERRIDE:-$DEFAULT_MAX_MODEL_LEN}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
  export CUDA_VISIBLE_DEVICES

  TP=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.6-35B-A3B \
    --served-model-name Qwen/Qwen3.6-35B-A3B \
    --host 0.0.0.0 \
    --port 8007 \
    --tensor-parallel-size "$TP" \
    --dtype bfloat16 \
    --max-model-len "$MODEL_MAX_MODEL_LEN" \
    --reasoning-parser qwen3 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --max-num-partial-prefills "$MAX_NUM_PARTIAL_PREFILLS" \
    --max-long-partial-prefills "$MAX_LONG_PARTIAL_PREFILLS" \
    --long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD" \
    --generation-config vllm \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder

elif [ "$MODE" = "qwen3.5-9b" ] || [ "$MODE" = "qwen35-9b" ] || [ "$MODE" = "qwen9b" ]; then
  MODEL_MAX_MODEL_LEN=${MAX_MODEL_LEN_OVERRIDE:-$DEFAULT_MAX_MODEL_LEN}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
  export CUDA_VISIBLE_DEVICES

  TP=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-9B \
    --served-model-name Qwen/Qwen3.5-9B \
    --host 0.0.0.0 \
    --port 8004 \
    --tensor-parallel-size "$TP" \
    --dtype bfloat16 \
    --max-model-len "$MODEL_MAX_MODEL_LEN" \
    --reasoning-parser qwen3 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --max-num-partial-prefills "$MAX_NUM_PARTIAL_PREFILLS" \
    --max-long-partial-prefills "$MAX_LONG_PARTIAL_PREFILLS" \
    --long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD" \
    --generation-config vllm \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder

elif [ "$MODE" = "qwen3.5-27b" ] || [ "$MODE" = "qwen35-27b" ] || [ "$MODE" = "qwen27b" ]; then
  MODEL_MAX_MODEL_LEN=${MAX_MODEL_LEN_OVERRIDE:-$DEFAULT_MAX_MODEL_LEN}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
  export CUDA_VISIBLE_DEVICES

  TP=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-27B \
    --served-model-name Qwen/Qwen3.5-27B \
    --host 0.0.0.0 \
    --port 8006 \
    --tensor-parallel-size "$TP" \
    --dtype bfloat16 \
    --max-model-len "$MODEL_MAX_MODEL_LEN" \
    --reasoning-parser qwen3 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --max-num-partial-prefills "$MAX_NUM_PARTIAL_PREFILLS" \
    --max-long-partial-prefills "$MAX_LONG_PARTIAL_PREFILLS" \
    --long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD" \
    --generation-config vllm \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder

elif [ "$MODE" = "qwen-a10b" ] || [ "$MODE" = "qwen122a10b" ]; then
  MODEL_MAX_MODEL_LEN=${MAX_MODEL_LEN_OVERRIDE:-$DEFAULT_MAX_MODEL_LEN}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5}
  export CUDA_VISIBLE_DEVICES

  TP=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-122B-A10B-FP8 \
    --served-model-name Qwen/Qwen3.5-122B-A10B-FP8 \
    --host 0.0.0.0 \
    --port 8003 \
    --tensor-parallel-size "$TP" \
    --dtype bfloat16 \
    --max-model-len "$MODEL_MAX_MODEL_LEN" \
    --reasoning-parser qwen3 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --max-num-partial-prefills "$MAX_NUM_PARTIAL_PREFILLS" \
    --max-long-partial-prefills "$MAX_LONG_PARTIAL_PREFILLS" \
    --long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD" \
    --generation-config vllm \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder

elif [ "$MODE" = "gemma26" ] || [ "$MODE" = "gemma-4-26b" ] || [ "$MODE" = "gemma-4-26b-a4b" ] || [ "$MODE" = "gemma-4-26b-a4b-vllm" ]; then
  MODEL_MAX_MODEL_LEN=${MAX_MODEL_LEN_OVERRIDE:-$DEFAULT_MAX_MODEL_LEN}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}
  export CUDA_VISIBLE_DEVICES

  TP=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
  if [ ! -f "$GEMMA26_MODEL_PATH/config.json" ]; then
    GEMMA26_MODEL_PATH="google/gemma-4-26B-A4B-it"
  fi

  python -m vllm.entrypoints.openai.api_server \
    --model "$GEMMA26_MODEL_PATH" \
    --served-model-name google/gemma-4-26B-A4B-it \
    --host 0.0.0.0 \
    --port 8008 \
    --tensor-parallel-size "$TP" \
    --dtype bfloat16 \
    --max-model-len "$MODEL_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --max-num-partial-prefills "$MAX_NUM_PARTIAL_PREFILLS" \
    --max-long-partial-prefills "$MAX_LONG_PARTIAL_PREFILLS" \
    --long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD" \
    --generation-config vllm \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-parser-plugin "${SCRIPT_DIR}/functiongemma_safe_tool_parser.py" \
    --tool-call-parser functiongemma_safe

elif [ "$MODE" = "gemma" ]; then
  MODEL_MAX_MODEL_LEN=${MAX_MODEL_LEN_OVERRIDE:-$DEFAULT_MAX_MODEL_LEN}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
  export CUDA_VISIBLE_DEVICES

  TP=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
  if [ ! -f "$GEMMA31_MODEL_PATH/config.json" ]; then
    GEMMA31_MODEL_PATH="google/gemma-4-31B-it"
  fi

  python -m vllm.entrypoints.openai.api_server \
    --model "$GEMMA31_MODEL_PATH" \
    --served-model-name google/gemma-4-31B-it \
    --host 0.0.0.0 \
    --port 8001 \
    --tensor-parallel-size "$TP" \
    --dtype bfloat16 \
    --max-model-len "$MODEL_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --max-num-partial-prefills "$MAX_NUM_PARTIAL_PREFILLS" \
    --max-long-partial-prefills "$MAX_LONG_PARTIAL_PREFILLS" \
    --long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD" \
    --generation-config vllm \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-parser-plugin "${SCRIPT_DIR}/functiongemma_safe_tool_parser.py" \
    --tool-call-parser functiongemma_safe

elif [ "$MODE" = "glm4" ]; then
  MODEL_MAX_MODEL_LEN=${MAX_MODEL_LEN_OVERRIDE:-$DEFAULT_MAX_MODEL_LEN}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
  export CUDA_VISIBLE_DEVICES

  TP=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')

  python -m vllm.entrypoints.openai.api_server \
    --model zai-org/GLM-4-32B-0414 \
    --host 0.0.0.0 \
    --port 8002 \
    --tensor-parallel-size "$TP" \
    --dtype bfloat16 \
    --max-model-len "$MODEL_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enable-chunked-prefill \
    --max-num-partial-prefills "$MAX_NUM_PARTIAL_PREFILLS" \
    --max-long-partial-prefills "$MAX_LONG_PARTIAL_PREFILLS" \
    --long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD" \
    --generation-config vllm \
    --enable-prefix-caching \
    --chat-template "${SCRIPT_DIR}/glm4_0414_tool_chat_template.jinja" \
    --enable-auto-tool-choice \
    --tool-parser-plugin "${SCRIPT_DIR}/glm4_0414_tool_parser.py" \
    --tool-call-parser glm4_0414

else
  echo "Usage: bash scripts/vllm.sh [qwen|qwen36|qwen36-8010|qwen3.6-35b-a3b|qwen3.5-9b|qwen3.5-27b|qwen-a10b|gemma|gemma26|glm4]"
  exit 1
fi
