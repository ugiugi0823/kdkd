#!/bin/bash
# ============================================================================
# Corpus Pretrain — 단일 서버 (1 Node, GPU × 8) 실행 스크립트
# H100 80GB × 8, NVLink, DeepSpeed ZeRO3
# ============================================================================
# 사용법:
#   bash sh/corpus/run-single.sh                          # 기본 pretrain (NLL)
#   bash sh/corpus/run-single.sh pretrain-gkd.yaml        # KD pretrain (JSD)
# ============================================================================

set -e

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

ACCELERATE_CONFIG="config/zero3-single.yaml"
TRAIN_SCRIPT="src/train_pretrain.py"

# 인자로 config 파일명 지정 가능 (기본: pretrain.yaml)
CONFIG_NAME="${1:-pretrain.yaml}"
CONFIG_FILE="config/${CONFIG_NAME}"

# ─── 환경 설정 ────────────────────────────────────────────────────────────────
source "${PROJECT_DIR}/venv/bin/activate"

# ─── NCCL 최적화 (단일 노드 — NVLink, InfiniBand 없음) ───────────────────────
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=OFF

export NCCL_TIMEOUT=3600
export NCCL_P2P_DISABLE=0       # NVLink P2P 활성화
export NCCL_SHM_DISABLE=0       # SHM 고속 버퍼 활성화
export NCCL_IB_DISABLE=1        # InfiniBand 없음 (단일 노드)

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0

# ─── CUDA 메모리 최적화 ───────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_DISABLE_CACHING=1

# ─── 사전 확인 ────────────────────────────────────────────────────────────────
if [ ! -f "${ACCELERATE_CONFIG}" ]; then
    echo "❌ Accelerate config 없음: ${ACCELERATE_CONFIG}"
    exit 1
fi

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "❌ 학습 config 없음: ${CONFIG_FILE}"
    echo "   사용 가능한 config:"
    ls config/*.yaml | sed 's/^/     /'
    exit 1
fi

if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "❌ 학습 스크립트 없음: ${TRAIN_SCRIPT}"
    exit 1
fi

DATA_PATHS=$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('${CONFIG_FILE}'))
for p in cfg.get('data_paths', []):
    print(p)
" 2>/dev/null)

MISSING=0
while IFS= read -r data_path; do
    if [ -n "${data_path}" ] && [ ! -d "${data_path}" ]; then
        echo "❌ 데이터 없음: ${data_path}"
        MISSING=1
    fi
done <<< "${DATA_PATHS}"

if [ "${MISSING}" -eq 1 ]; then
    echo "   python download_datasets.py 로 먼저 다운로드하세요"
    exit 1
fi

# ─── 로그 ─────────────────────────────────────────────────────────────────────
mkdir -p ./logs
CONFIG_STEM="${CONFIG_NAME%.yaml}"
LOG_FILE="./logs/pretrain_single_${CONFIG_STEM}_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================"
echo "  Corpus Pretrain  —  Single Node"
echo "  Framework : Accelerate + DeepSpeed ZeRO3"
echo "  GPUs      : H100 80GB × 8"
echo "  Config    : ${CONFIG_FILE}"
echo "  Log       : ${LOG_FILE}"
echo "================================================================"

# ─── 실행 ─────────────────────────────────────────────────────────────────────
nohup accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    "${TRAIN_SCRIPT}" \
    --config "${CONFIG_FILE}" \
    >> "${LOG_FILE}" 2>&1 &

PID=$!
echo "[STARTED] PID=${PID} | Log=${LOG_FILE}"
echo "📋 실시간 로그: tail -f ${LOG_FILE}"
