# GKD 학습 이력 및 평가 계획

## 1. 실험 개요

| 항목 | 내용 |
|------|------|
| 방법론 | Corpus Knowledge Distillation (Off-policy GKD) |
| Loss | Jensen-Shannon Divergence (JSD, symmetric, β=0.5) |
| Student | `google/gemma-3-4b-pt` (4B, text-only) |
| Teacher | `google/gemma-3-12b-pt` (12B, text-only) |
| 학습 데이터 | 한국어 금융 도메인 코퍼스 2종 |
| 평가 목표 | KMMLU (Korean Massive Multitask Language Understanding) |

---

## 2. 학습 데이터

| 데이터셋 | 경로 | train | validation |
|----------|------|------:|----------:|
| jp1924/MultilingualCorpusInFinancialSector | `/xtmp/jp1924_MultilingualCorpusInFinancialSector` | 821,752건 | 500건 |
| ugiugi/korean_financial_corpus | `/xtmp/ugiugi_korean_financial_corpus` | 7,579건 | 500건 |
| **합계** | | **829,331건** | **1,000건** |

- `text_field`: `sentence_ls`
- `text_join_sep`: `\n`

---

## 3. 모델 정보

### Student: Gemma-3-4B-PT (학습 대상)

| 항목 | 값 |
|------|-----|
| 로컬 경로 | `/PROJECT/0325120095_A/BASE/rex/LLM/models/input/google/gemma-3-4b-pt` |
| 아키텍처 | `Gemma3ForCausalLM` (text-only, VLM 체크포인트에서 language_model 추출) |
| hidden_size | 2,560 |
| num_layers | 34 |
| vocab_size | 262,208 |
| dtype | BF16 |

### Teacher: Gemma-3-12B-PT (고정, 지식 제공)

| 항목 | 값 |
|------|-----|
| 로컬 경로 | `/PROJECT/0325120095_A/BASE/rex/LLM/models/input/google/gemma-3-12b-pt` |
| 아키텍처 | `Gemma3ForCausalLM` (text-only) |
| 역할 | Inference-only (requires_grad=False) |

---

## 4. 학습 설정 (`config/pretrain-gkd.yaml`)

### 핵심 하이퍼파라미터

| 파라미터 | 값 |
|----------|-----|
| num_train_epochs | 1 |
| per_device_train_batch_size | 1 |
| gradient_accumulation_steps | 4 |
| learning_rate | 2.0e-6 |
| lr_scheduler_type | cosine |
| warmup_ratio | 0.03 (≈ 32 steps) |
| max_length | 8,192 |
| optim | adamw_8bit |
| max_grad_norm | 1.0 |
| use_peft | false (Full Finetuning) |

### KD 설정

| 파라미터 | 값 |
|----------|-----|
| beta | 0.5 (symmetric JSD) |
| temperature | 1.0 |
| Loss 수식 | `JSD(S‖T) = 0.5·KL(S‖M) + 0.5·KL(T‖M)`, M = (S+T)/2 |

### 인프라

| 파라미터 | 값 |
|----------|-----|
| GPU | NVIDIA B200 × 3 (GPU 3, 4, 5) |
| 분산 학습 | DeepSpeed ZeRO3 + CPU offload |
| attention | flash_attention_2 |
| liger_kernel | ✅ (apply_liger_kernel_to_gemma3_text) |
| packing | ✅ |
| bf16 / tf32 | ✅ / ✅ |
| venv | `/PROJECT/0325120095_A/BASE/rex/LLM/venv` |

---

## 5. 학습 결과

### 실행 정보

| 항목 | 값 |
|------|-----|
| 시작 시각 | 2026-03-05 16:29:09 |
| timestamp | `20260305_162921` |
| 총 step | 1,056 / 1,056 ✅ |
| 총 소요 시간 | **2시간 30분 09초** (9,009초) |
| 처리 토큰 | 약 **1.003억 토큰** |
| 처리 속도 | ~11,140 tokens/sec |
| WandB Run | [corpus-kd @ wandb](https://wandb.ai/ugiugi/LLM/runs/97haw6sd) |

### Loss 추이

| epoch | train loss (logged) | raw JSD (per-token) | eval loss (CE) |
|-------|--------------------:|--------------------:|---------------:|
| 0.000 | — | — | 2.731 |
| 0.070 | ~0.00090 | ~0.00022 | 2.731 |
| 0.210 | ~0.00130 | ~0.00033 | 2.730 |
| 0.420 | ~0.00155 | ~0.00039 | 2.729 |
| 0.490 | ~0.00148 | ~0.00035 | 2.732 |
| 0.700 | ~0.00130 | ~0.00030 | 2.730 |
| 0.981 | ~0.00110 | — | 2.730 |
| **1.000** | **0.000839** | — | **2.730** |

> - **logged train loss** = raw JSD × gradient_accumulation_steps(4)
> - **eval loss** = CE loss (standard cross-entropy, evaluation용)
> - LR이 cosine 감소하면서 후반부 loss 안정적 수렴 확인

### 저장된 체크포인트

| 경로 | step | epoch |
|------|-----:|------:|
| `output/gemma-3-4b-pt/kd/run_20260305_162921/checkpoint-444` | 444 | 0.42 |
| `output/gemma-3-4b-pt/kd/run_20260305_162921/checkpoint-1036` | 1,036 | 0.98 |
| `output/gemma-3-4b-pt/kd/run_20260305_162921/checkpoint-1056` | 1,056 | 1.00 |
| **`output/gemma-3-4b-pt/kd/run_20260305_162921/final_model`** | **1,056** | **1.00** ✅ |

### 최종 모델

```
/PROJECT/0325120095_A/BASE/rex/LLM/output/gemma-3-4b-pt/kd/run_20260305_162921/final_model/
├── config.json
├── generation_config.json
├── model.safetensors   (~8.5GB, BF16)
├── tokenizer.json
├── tokenizer_config.json
└── training_args.bin
```

---

## 6. 평가 계획 (KMMLU)

### 평가 대상 모델

| 모델 | 경로 | 비고 |
|------|------|------|
| **GKD 학습 모델** | `output/gemma-3-4b-pt/kd/run_20260305_162921/final_model` | 주 평가 대상 |
| Baseline (미학습) | `models/input/google/gemma-3-4b-pt` | 비교 기준 |

### 평가 벤치마크: KMMLU

- **태스크**: `kmmlu` (Korean Massive Multitask Language Understanding)
- **출처**: [HAERAE-HUB/KMMLU](https://huggingface.co/datasets/HAERAE-HUB/KMMLU)
- **구성**: 한국어 45개 전문 과목 (법률, 의학, 금융, 공학 등)
- **평가 방식**: 5-shot, multiple choice (A/B/C/D)
- **지표**: `acc` (정확도)

### 실행 환경

| 항목 | 값 |
|------|-----|
| venv | `/PROJECT/0325120095_A/BASE/rex/LLM/kmmlu/bin/activate` |
| 평가 프레임워크 | `lm-evaluation-harness` |
| 추론 백엔드 | `vllm` (고속 추론) |
| torch | 2.9.1+cu128 |
| vllm | 0.16.0 |

### 실행 명령 (예정)

```bash
source /PROJECT/0325120095_A/BASE/rex/LLM/kmmlu/bin/activate
cd /PROJECT/0325120095_A/BASE/rex/LLM

MODEL_PATH="output/gemma-3-4b-pt/kd/run_20260305_162921/final_model"
LOG_DIR="logs/eval"
mkdir -p "$LOG_DIR"

# GKD 학습 모델 평가
lm_eval \
  --model vllm \
  --model_args "pretrained=${MODEL_PATH},dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.85" \
  --tasks kmmlu \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path "${LOG_DIR}/kmmlu-gkd-$(date +%Y%m%d_%H%M%S)" \
  --log_samples \
  2>&1 | tee "${LOG_DIR}/kmmlu-gkd-$(date +%Y%m%d_%H%M%S).log"

# Baseline 평가 (비교용)
BASELINE_PATH="models/input/google/gemma-3-4b-pt"

lm_eval \
  --model vllm \
  --model_args "pretrained=${BASELINE_PATH},dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.85" \
  --tasks kmmlu \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path "${LOG_DIR}/kmmlu-baseline-$(date +%Y%m%d_%H%M%S)" \
  --log_samples \
  2>&1 | tee "${LOG_DIR}/kmmlu-baseline-$(date +%Y%m%d_%H%M%S).log"
```

### 평가 시 주의사항

1. **lm-evaluation-harness 미설치**: kmmlu venv에 `lm_eval` 패키지 설치 필요
   ```bash
   pip install lm-eval[vllm]
   ```
2. **CUDA 호환성**: kmmlu venv는 `torch 2.9.1+cu128` (CUDA 12.8) — 학습 venv(CUDA 13.0)과 다름. vllm 추론은 별도 GPU 할당 권장
3. **vllm & Gemma3**: vllm이 `gemma3_text` 아키텍처를 지원하는지 확인 필요 (지원하지 않을 경우 `--model hf` 모드로 전환)
4. **금융 관련 과목 집중 분석**: KMMLU 45개 과목 중 `accounting`, `taxation`, `economics` 등 금융/경제 관련 과목에서 KD 효과가 더 두드러질 것으로 예상

---

## 7. KMMLU 평가 결과

### 평가 실행 정보

| 항목 | GKD 모델 | Baseline |
|------|----------|---------|
| 시작 | 2026-03-05 23:49:34 | 2026-03-05 23:47:58 |
| 종료 | 2026-03-06 00:12:09 | 2026-03-06 00:12:33 |
| 소요 시간 | 약 22분 | 약 24분 |
| 총 요청 수 | 140,120 | 140,120 |
| 추론 속도 (peak) | ~14,700 it/s | ~12,200 it/s |
| GPU | CUDA 3 (B200) | CUDA 4 (B200) |
| 로그 | `logs/eval/kmmlu-gkd-20260305_234934.log` | `logs/eval/kmmlu-baseline-20260305_234758.log` |

### 그룹별 결과 (5-shot, acc)

| 그룹 | GKD 모델 | Baseline | 차이 (GKD - Base) |
|------|:--------:|:--------:|:-----------------:|
| **kmmlu (전체)** | **0.3986** | **0.3975** | **+0.0011 (+0.28%)** |
| kmmlu_applied_science | 0.3859 | 0.3839 | +0.0020 (+0.52%) |
| kmmlu_humss | 0.4021 | 0.4012 | +0.0009 (+0.22%) |
| kmmlu_other | 0.4029 | 0.4031 | -0.0002 (-0.05%) |
| kmmlu_stem | 0.4079 | 0.4067 | +0.0012 (+0.30%) |

### 과목별 상세 결과

#### Applied Science (응용과학)

| 과목 | GKD | Baseline | 차이 |
|------|----:|--------:|-----:|
| aviation_engineering_and_maintenance | 0.4000 | 0.3970 | +0.0030 |
| electronics_engineering | 0.4640 | 0.4610 | +0.0030 |
| energy_management | **0.3100** | **0.3040** | **+0.0060** |
| environmental_science | 0.2910 | 0.2950 | -0.0040 |
| gas_technology_and_engineering | **0.3370** | **0.3300** | **+0.0070** |
| geomatics | 0.3880 | 0.3900 | -0.0020 |
| industrial_engineer | 0.4000 | 0.4010 | -0.0010 |
| machine_design_and_manufacturing | 0.3870 | 0.3890 | -0.0020 |
| maritime_engineering | 0.4117 | 0.4100 | +0.0017 |
| nondestructive_testing | **0.4060** | **0.3980** | **+0.0080** |
| railway_and_automotive_engineering | 0.3460 | 0.3410 | +0.0050 |
| telecommunications_and_wireless_technology | 0.5010 | 0.5010 | 0.0000 |

#### HUMSS (인문·사회)

| 과목 | GKD | Baseline | 차이 |
|------|----:|--------:|-----:|
| accounting | 0.3100 | 0.3100 | 0.0000 |
| criminal_law | 0.3250 | 0.3300 | -0.0050 |
| economics | 0.4308 | **0.4538** | -0.0230 |
| education | **0.5100** | **0.4900** | **+0.0200** |
| korean_history | 0.2700 | 0.2700 | 0.0000 |
| law | 0.3830 | 0.3830 | 0.0000 |
| management | 0.4420 | 0.4400 | +0.0020 |
| political_science_and_sociology | 0.4367 | 0.4433 | -0.0066 |
| psychology | 0.3790 | 0.3770 | +0.0020 |
| social_welfare | 0.4310 | 0.4270 | +0.0040 |
| **taxation** | **0.3350** | **0.3300** | **+0.0050** |

#### Other (기타 전문)

| 과목 | GKD | Baseline | 차이 |
|------|----:|--------:|-----:|
| agricultural_sciences | 0.3200 | 0.3210 | -0.0010 |
| construction | 0.3130 | 0.3150 | -0.0020 |
| fashion | 0.4050 | 0.4020 | +0.0030 |
| food_processing | 0.3560 | 0.3530 | +0.0030 |
| health | 0.4900 | 0.5100 | -0.0200 |
| interior_architecture_and_design | **0.4840** | **0.4770** | **+0.0070** |
| marketing | 0.6980 | 0.7010 | -0.0030 |
| patent | 0.3600 | 0.3800 | -0.0200 |
| public_safety | 0.3370 | 0.3420 | -0.0050 |
| real_estate | **0.3550** | **0.3450** | **+0.0100** |
| refrigerating_machinery | 0.3150 | 0.3170 | -0.0020 |

#### STEM (과학·기술·공학·수학)

| 과목 | GKD | Baseline | 차이 |
|------|----:|--------:|-----:|
| biology | 0.3230 | 0.3170 | +0.0060 |
| chemical_engineering | 0.3990 | 0.3970 | +0.0020 |
| chemistry | 0.3833 | 0.3933 | -0.0100 |
| civil_engineering | 0.3540 | 0.3490 | +0.0050 |
| computer_science | 0.5900 | 0.5940 | -0.0040 |
| ecology | 0.4390 | 0.4410 | -0.0020 |
| electrical_engineering | 0.2940 | 0.2920 | +0.0020 |
| information_technology | 0.6010 | 0.6030 | -0.0020 |
| materials_engineering | 0.3750 | 0.3770 | -0.0020 |
| **math** | **0.3200** | **0.3033** | **+0.0167** |
| mechanical_engineering | 0.3370 | 0.3290 | +0.0080 |

### 평가 결과 분석

1. **전체 성능**: GKD 모델이 Baseline 대비 **+0.11%p** (39.86% vs 39.75%) 향상
2. **유의미한 개선 과목**:
   - `math`: +0.0167 (가장 큰 향상)
   - `education`: +0.0200
   - `real_estate`: +0.0100
   - `nondestructive_testing`: +0.0080
   - `interior_architecture_and_design`: +0.0070
   - `gas_technology_and_engineering`: +0.0070
   - `energy_management`: +0.0060
3. **금융 도메인 효과** (학습 데이터 연관):
   - `taxation` (세무): +0.0050 ✓ (예상대로 소폭 개선)
   - `accounting` (회계): 0.0000 (변화 없음)
   - `economics` (경제): -0.0230 (오히려 하락)
4. **주요 하락 과목**: `economics` (-0.0230), `patent` (-0.0200), `health` (-0.0200)
5. **결론**: Off-policy GKD 1 epoch 학습으로 전반적인 소폭 성능 향상 확인. 금융 도메인 특화 효과는 예상보다 미미하며, 더 많은 epoch이나 더 큰 teacher 모델이 필요할 수 있음.

> 출력 경로:
> - `logs/eval/kmmlu-gkd-20260305_234934/` (GKD 모델 결과 JSON)
> - `logs/eval/kmmlu-baseline-20260305_234758/` (Baseline 결과 JSON)

---

## 8. 향후 계획

| 단계 | 내용 | 상태 |
|------|------|------|
| ✅ | GKD 학습 완료 (1 epoch, JSD loss) | 완료 |
| ✅ | KMMLU 평가 (GKD 모델 vs Baseline) | **완료** |
| 🔜 | GOLD 학습 (Cross-tokenizer ULD, Qwen3.5-9B teacher) | 대기 |
| 🔜 | GOLD 학습 완료 후 KMMLU 재평가 (GKD vs GOLD vs Baseline) | 대기 |

---

*생성일: 2026-03-05*
*WandB: https://wandb.ai/ugiugi/LLM*
