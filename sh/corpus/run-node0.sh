#!/bin/bash
# ============================================================================
# Corpus Pretrain — 서버1 (Node 0 / Master) 실행 스크립트
# H100 80GB × 8, InfiniBand, DeepSpeed ZeRO3
# ============================================================================
# 사용법:
#   서버1에서: bash sh/corpus/run-node0.sh                     # 기본 pretrain
#   서버1에서: bash sh/corpus/run-node0.sh pretrain-gkd.yaml   # KD pretrain
#   서버2에서: bash sh/corpus/run-node1.sh [same config]       ← 동시에 실행
# ============================================================================

set -e

# ─── 경로 설정 ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

ACCELERATE_CONFIG="config/zero3-node0.yaml"
TRAIN_SCRIPT="src/train_pretrain.py"

# 인자로 config 파일명 지정 가능 (기본: pretrain.yaml)
CONFIG_NAME="${1:-pretrain.yaml}"
CONFIG_FILE="config/${CONFIG_NAME}"

# ─── 환경 설정 ────────────────────────────────────────────────────────────────
source /home/rex/workspace/orca/tr/venv/bin/activate

# ─── NCCL / InfiniBand 최적화 ─────────────────────────────────────────────────
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=OFF

export NCCL_TIMEOUT=7200
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_RETRY_CNT=7

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

if [ ! -d "/xtmp/jp1924_DomainSpecificCorpus" ]; then
    echo "❌ 데이터 없음: /xtmp/jp1924_DomainSpecificCorpus"
    echo "   python download_datasets.py 로 먼저 다운로드하세요"
    exit 1
fi

# ─── 로그 ─────────────────────────────────────────────────────────────────────
mkdir -p ./logs
CONFIG_STEM="${CONFIG_NAME%.yaml}"
LOG_FILE="./logs/pretrain_node0_${CONFIG_STEM}_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================"
echo "  Corpus Pretrain  —  Node 0 (Master)"
echo "  Framework : Accelerate + DeepSpeed ZeRO3"
echo "  Nodes     : 2  (master: 10.34.1.11:29500)"
echo "  GPUs      : H100 80GB × 8 (이 서버)"
echo "  Config    : ${CONFIG_FILE}"
echo "  Log       : ${LOG_FILE}"
echo "================================================================"
echo ""
echo "⚠️  서버2에서도 아래 명령을 동시에 실행하세요:"
echo "     bash sh/corpus/run-node1.sh ${CONFIG_NAME}"
echo ""

# ─── 실행 ─────────────────────────────────────────────────────────────────────
nohup accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    "${TRAIN_SCRIPT}" \
    --config "${CONFIG_FILE}" \
    >> "${LOG_FILE}" 2>&1 &

PID=$!
echo "[STARTED] PID=${PID} | Log=${LOG_FILE}"
echo "📋 실시간 로그: tail -f ${LOG_FILE}"
