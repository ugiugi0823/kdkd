#!/bin/bash
BASE_DIR="/PROJECT/0325120095_A/BASE/rex/LLM"
LOG_DIR="${BASE_DIR}/logs/eval"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ADAPTER_PATH="${BASE_DIR}/output/gemma-3-4b-pt/gold/run_20260306_141411/final_model"
MERGED_PATH="${BASE_DIR}/output/gemma-3-4b-pt/gold/run_20260306_141411/merged_model"
LOG_FILE="${LOG_DIR}/kmmlu-gold-${TIMESTAMP}.log"

echo "[GOLD eval] START: $(date)" | tee "$LOG_FILE"
echo "ADAPTER: $ADAPTER_PATH" | tee -a "$LOG_FILE"
echo "MERGED:  $MERGED_PATH" | tee -a "$LOG_FILE"
echo "LOG: $LOG_FILE" | tee -a "$LOG_FILE"

# Step 1: LoRA 어댑터 머지
if [ ! -d "$MERGED_PATH" ]; then
  echo "[GOLD eval] Merging LoRA adapter into base model..." | tee -a "$LOG_FILE"
  CUDA_VISIBLE_DEVICES=3 ${BASE_DIR}/venv2/bin/python3 ${BASE_DIR}/src/merge_gold_adapter.py 2>&1 | tee -a "$LOG_FILE"

  if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[GOLD eval] ERROR: Merge failed. Exiting." | tee -a "$LOG_FILE"
    exit 1
  fi
  echo "[GOLD eval] Merge done." | tee -a "$LOG_FILE"
else
  echo "[GOLD eval] Merged model already exists, skipping merge." | tee -a "$LOG_FILE"
fi

# Step 2: KMMLU 평가
mkdir -p /xtmp/triton_cache_gold
export TORCHINDUCTOR_CACHE_DIR=/xtmp/triton_cache_gold
export TRITON_CACHE_DIR=/xtmp/triton_cache_gold

cd /xtmp

echo "[GOLD eval] Running KMMLU evaluation..." | tee -a "$LOG_FILE"
CUDA_VISIBLE_DEVICES=3 ${BASE_DIR}/kmmlu/bin/lm_eval \
  --model vllm \
  --model_args "pretrained=${MERGED_PATH},dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.85,block_size=32" \
  --tasks kmmlu \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path "${LOG_DIR}/kmmlu-gold-${TIMESTAMP}" \
  --log_samples \
  2>&1 | tee -a "$LOG_FILE"

echo "[GOLD eval] END: $(date)" | tee -a "$LOG_FILE"
