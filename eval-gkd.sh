#!/bin/bash
BASE_DIR="/PROJECT/0325120095_A/BASE/rex/LLM"
LOG_DIR="${BASE_DIR}/logs/eval"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_PATH="${BASE_DIR}/output/gemma-3-4b-pt/kd/run_20260310_002255/final_model"
LOG_FILE="${LOG_DIR}/kmmlu-gkd-${TIMESTAMP}.log"

echo "[GKD eval] START: $(date)" | tee "$LOG_FILE"
echo "MODEL: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "LOG: $LOG_FILE" | tee -a "$LOG_FILE"

# /xtmp 에서 실행: /tmp(noexec) 충돌 방지 + triton 캐시 실행 허용
mkdir -p /xtmp/triton_cache_gkd
export TORCHINDUCTOR_CACHE_DIR=/xtmp/triton_cache_gkd
export TRITON_CACHE_DIR=/xtmp/triton_cache_gkd

cd /xtmp

CUDA_VISIBLE_DEVICES=3 /PROJECT/0325120095_A/BASE/rex/LLM/kmmlu/bin/lm_eval \
  --model vllm \
  --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.85,block_size=32" \
  --tasks kmmlu \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path "${LOG_DIR}/kmmlu-gkd-${TIMESTAMP}" \
  --log_samples \
  2>&1 | tee -a "$LOG_FILE"

echo "[GKD eval] END: $(date)" | tee -a "$LOG_FILE"
