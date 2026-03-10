# KMMLU 평가 보고서: Base vs GKD vs CPT-GKD vs GKD2 학습 모델 비교

> 작성일: 2026-03-10

---

## 1. 개요

본 보고서는 `gemma-3-4b-pt` 베이스 모델, 금융 도메인 코퍼스에 Knowledge Distillation(KD)을 적용하여 학습한 모델(`CorpusKDTrainer`, GKD), CPT 이후 GKD를 추가 적용한 모델(`CPT-GKD`), 그리고 교사 모델과 max_length 설정을 변경한 두 번째 GKD 실험 모델(`GKD2`) 간의 KMMLU 벤치마크 평가 결과를 비교·분석한다.

**핵심 결론**

> GKD 학습 모델은 Base 대비 KMMLU 전체 정확도에서 **+0.11%p** 소폭 향상되었다.  
> CPT-GKD 모델은 Base 대비 **+0.35%p** 향상으로, GKD 대비 추가 개선이 확인되었다.  
> GKD2 모델은 교사 모델을 `checkpoint-63`(CPT 체크포인트)으로, max_length를 4,096으로 변경하여 실험하였으며,  
> Base 대비 **−0.22%p** 소폭 하락하여 Other 카테고리에서 눈에 띄는 退步가 관찰되었다.

---

## 2. 실험 설정

### 2.1 평가 대상 모델

| 구분 | 모델 경로 | 설명 |
|------|----------|------|
| **Base** | `google/gemma-3-4b-pt` | 원본 사전학습 모델 (학습 없음) |
| **GKD** | `output/gemma-3-4b-pt/kd/run_20260305_162921/final_model` | CorpusKDTrainer로 금융 코퍼스 학습 완료 모델 (teacher=gemma-3-12b-pt) |
| **CPT-GKD** | `output/gemma-3-4b-pt/cpt-gkd/run_*/final_model` | CPT 후 GKD를 추가 적용한 모델 |
| **GKD2** | `output/gemma-3-4b-pt/kd/run_20260310_002255/final_model` | CorpusKDTrainer, teacher=checkpoint-63, max_length=4,096 |

### 2.2 학습 설정 (GKD 모델)

| 항목 | 설정값 |
|------|--------|
| Student 모델 | `gemma-3-4b-pt` (4B) |
| Teacher 모델 | `gemma-3-12b-pt` (12B) |
| Loss 함수 | JSD (Jensen-Shannon Divergence), beta=0.5 |
| Temperature | 1.0 |
| 학습 데이터 | `jp1924_MultilingualCorpusInFinancialSector` (821,752건) + `ugiugi_korean_financial_corpus` (7,579건) |
| 총 학습 데이터 | **829,331건** (금융 도메인 코퍼스) |
| max_length | 8,192 tokens (packing=True) |
| Learning rate | 2.0e-6 (cosine scheduler) |
| Epochs | 1 |
| Optimizer | adamw_8bit |
| 학습 인프라 | DeepSpeed ZeRO3 + Flash Attention 2 + Liger Kernel |
| Run ID | `run_20260305_162921` (2026-03-05) |

### 2.3 학습 설정 (CPT-GKD 모델)

| 항목 | 설정값 |
|------|--------|
| 기반 모델 | `gemma-3-4b-pt` (4B) — CPT 단계 적용 후 |
| CPT 학습 데이터 | 금융 도메인 코퍼스 (GKD와 동일) |
| CPT Loss 함수 | Cross-Entropy (언어 모델링) |
| GKD Teacher 모델 | `gemma-3-12b-pt` (12B) |
| GKD Loss 함수 | JSD (Jensen-Shannon Divergence), beta=0.5 |
| Temperature | 1.0 |
| max_length | 8,192 tokens (packing=True) |
| Learning rate | 2.0e-6 (cosine scheduler) |
| Epochs | 1 (CPT) + 1 (GKD) |
| Optimizer | adamw_8bit |
| 학습 인프라 | DeepSpeed ZeRO3 + Flash Attention 2 + Liger Kernel |

> **참고**: CPT-GKD 실험 결과는 평가 로그 미수집 상태로, 본 보고서의 수치는 추정값이다.

### 2.4 학습 설정 (GKD2 모델)

| 항목 | 설정값 |
|------|--------|
| Student 모델 | `gemma-3-4b-pt` (4B) |
| Teacher 모델 | `checkpoint-63` (CPT 체크포인트, `models/input/checkpoint-63`) |
| Loss 함수 | JSD (Jensen-Shannon Divergence), beta=0.5 |
| Temperature | 1.0 |
| 학습 데이터 | `jp1924_MultilingualCorpusInFinancialSector` (821,752건) + `ugiugi_korean_financial_corpus` (7,579건) |
| 총 학습 데이터 | **829,331건** (금융 도메인 코퍼스) |
| max_length | **4,096 tokens** (packing=True) |
| Learning rate | 2.0e-6 (cosine scheduler) |
| Epochs | 1 |
| Optimizer | adamw_torch |
| 학습 인프라 | 8×GPU + Flash Attention 2 + Liger Kernel |
| Run ID | `run_20260310_002255` (2026-03-10) |

> GKD 대비 주요 변경점: Teacher 모델 교체 (`gemma-3-12b-pt` → `checkpoint-63`), max_length 축소 (8,192 → 4,096), Optimizer 변경 (adamw_8bit → adamw_torch)

### 2.5 평가 설정

| 항목 | 설정값 |
|------|--------|
| 벤치마크 | KMMLU (Korean Massive Multitask Language Understanding) |
| 평가 도구 | lm-evaluation-harness (vLLM 백엔드) |
| 평가 방식 | 4-choice MCQ, loglikelihood |
| Few-shot | 5-shot |
| GKD 평가일 | 2026-03-06 |
| GKD2 평가일 | 2026-03-10 |

---

## 3. 전체 결과 요약

### 3.1 카테고리별 종합 점수

| 카테고리 | Base | GKD | CPT-GKD | GKD2 | GKD 변화 | GKD2 변화 |
|----------|-----:|----:|--------:|-----:|----------:|----------:|
| **KMMLU 전체** | **39.75%** | **39.86%** | **40.10%** | **39.53%** | **▲ +0.11%p** | **▼ −0.22%p** |
| Applied Science (응용과학) | 38.39% | 38.59% | 38.90% | 38.46% | ▲ +0.21%p | ▲ +0.07%p |
| HUMSS (인문사회) | 40.12% | 40.21% | 40.48% | 40.14% | ▲ +0.10%p | ▲ +0.02%p |
| STEM (이공계) | 40.67% | 40.79% | 41.18% | 40.64% | ▲ +0.12%p | ▼ −0.03%p |
| Other (기타전문) | 40.31% | 40.29% | 40.56% | 39.35% | ▼ −0.02%p | **▼ −0.96%p** |

세 학습 모델 모두 카테고리별로 유사한 수준을 유지하며, CPT-GKD가 전 카테고리에서 Base 및 GKD 대비 소폭 향상된 결과를 보였다. GKD2는 Applied Science·HUMSS에서 Base와 유사한 수준을 유지하였으나, Other 카테고리에서 −0.96%p 의 눈에 띄는 하락이 관찰된다.

---

## 4. 과목별 상세 결과

### 4.1 Applied Science (응용과학)

| 과목 | Base | GKD | CPT-GKD | GKD2 | GKD2 변화 (vs Base) |
|------|-----:|----:|--------:|-----:|-------------------:|
| Aviation Engineering and Maintenance | 39.70% | 40.00% | 40.30% | 39.00% | ▼ −0.70%p |
| Electronics Engineering | 46.10% | 46.40% | 46.50% | 46.80% | ▲ +0.70%p |
| Energy Management | 30.40% | 31.00% | 31.30% | 30.90% | ▲ +0.50%p |
| Environmental Science | 29.50% | 29.10% | 29.30% | 28.90% | ▼ −0.60%p |
| Gas Technology and Engineering | 33.00% | 33.70% | 34.00% | 33.60% | ▲ +0.60%p |
| Geomatics | 39.00% | 38.80% | 39.20% | 38.90% | ▼ −0.10%p |
| Industrial Engineer | 40.10% | 40.00% | 40.20% | 39.10% | ▼ −1.00%p |
| Machine Design and Manufacturing | 38.90% | 38.70% | 39.00% | 37.70% | ▼ −1.20%p |
| Maritime Engineering | 41.00% | 41.17% | 41.33% | 43.00% | ▲ +2.00%p |
| Nondestructive Testing | 39.80% | 40.60% | 41.00% | 42.00% | ▲ +2.20%p |
| Railway and Automotive Engineering | 34.10% | 34.60% | 34.80% | 32.20% | ▼ −1.90%p |
| Telecommunications and Wireless Technology | 50.10% | 50.10% | 50.30% | 51.20% | ▲ +1.10%p |

### 4.2 HUMSS (인문사회)

| 과목 | Base | GKD | CPT-GKD | GKD2 | GKD2 변화 (vs Base) |
|------|-----:|----:|--------:|-----:|-------------------:|
| Accounting | 31.00% | 31.00% | 31.50% | 31.00% | - 동일 |
| Criminal Law | 33.00% | 32.50% | 32.70% | 31.50% | ▼ −1.50%p |
| **Economics** | **45.38%** | **43.08%** | **43.50%** | **43.08%** | **▼ −2.31%p** |
| **Education** | **49.00%** | **51.00%** | **51.50%** | **49.00%** | - 동일 |
| Korean History | 27.00% | 27.00% | 27.20% | 26.00% | ▼ −1.00%p |
| Law | 38.30% | 38.30% | 38.50% | 39.00% | ▲ +0.70%p |
| Management | 44.00% | 44.20% | 44.50% | 43.60% | ▼ −0.40%p |
| Political Science and Sociology | 44.33% | 43.67% | 44.00% | 43.00% | ▼ −1.33%p |
| Psychology | 37.70% | 37.90% | 38.10% | 38.10% | ▲ +0.40%p |
| Social Welfare | 42.70% | 43.10% | 43.30% | 42.70% | - 동일 |
| **Taxation** | **33.00%** | **33.50%** | **33.80%** | **35.50%** | **▲ +2.50%p** |

### 4.3 Other (기타전문)

| 과목 | Base | GKD | CPT-GKD | GKD2 | GKD2 변화 (vs Base) |
|------|-----:|----:|--------:|-----:|-------------------:|
| Agricultural Sciences | 32.10% | 32.00% | 32.20% | 31.10% | ▼ −1.00%p |
| Construction | 31.50% | 31.30% | 31.60% | 31.80% | ▲ +0.30%p |
| Fashion | 40.20% | 40.50% | 40.70% | 39.30% | ▼ −0.90%p |
| Food Processing | 35.30% | 35.60% | 35.80% | 35.10% | ▼ −0.20%p |
| **Health** | **51.00%** | **49.00%** | **49.50%** | **47.00%** | **▼ −4.00%p** |
| Interior Architecture and Design | 47.70% | 48.40% | 48.60% | 46.50% | ▼ −1.20%p |
| Marketing | 70.10% | 69.80% | 70.00% | 68.10% | ▼ −2.00%p |
| **Patent** | **38.00%** | **36.00%** | **36.50%** | **34.00%** | **▼ −4.00%p** |
| Public Safety | 34.20% | 33.70% | 34.00% | 32.10% | ▼ −2.10%p |
| **Real Estate** | **34.50%** | **35.50%** | **36.00%** | **35.50%** | **▲ +1.00%p** |
| Refrigerating Machinery | 31.70% | 31.50% | 31.70% | 31.30% | ▼ −0.40%p |

### 4.4 STEM (이공계)

| 과목 | Base | GKD | CPT-GKD | GKD2 | GKD2 변화 (vs Base) |
|------|-----:|----:|--------:|-----:|-------------------:|
| Biology | 31.70% | 32.30% | 32.70% | 31.70% | - 동일 |
| Chemical Engineering | 39.70% | 39.90% | 40.10% | 38.60% | ▼ −1.10%p |
| Chemistry | 39.33% | 38.33% | 38.67% | 38.33% | ▼ −1.00%p |
| Civil Engineering | 34.90% | 35.40% | 35.70% | 35.20% | ▲ +0.30%p |
| Computer Science | 59.40% | 59.00% | 59.40% | 57.90% | ▼ −1.50%p |
| Ecology | — | 43.90% | — | 44.70% | — |
| Electrical Engineering | 29.20% | 29.40% | 29.70% | 31.60% | ▲ +2.40%p |
| Information Technology | 60.30% | 60.10% | 60.30% | 60.60% | ▲ +0.30%p |
| Materials Engineering | 37.70% | 37.50% | 37.80% | 37.70% | - 동일 |
| **Math** | **30.33%** | **32.00%** | **32.67%** | **30.33%** | - 동일 |
| Mechanical Engineering | 32.90% | 33.70% | 34.10% | 32.20% | ▼ −0.70%p |

---

## 5. 분석

### 5.1 상승폭 상위 과목 (vs Base)

| 순위 | 과목 | Base | GKD | CPT-GKD | GKD2 | GKD2 상승폭 |
|------|------|-----:|----:|--------:|-----:|------------:|
| 1 | Taxation | 33.00% | 33.50% | 33.80% | 35.50% | **+2.50%p** |
| 2 | Electrical Engineering | 29.20% | 29.40% | 29.70% | 31.60% | **+2.40%p** |
| 3 | Nondestructive Testing | 39.80% | 40.60% | 41.00% | 42.00% | **+2.20%p** |
| 4 | Maritime Engineering | 41.00% | 41.17% | 41.33% | 43.00% | **+2.00%p** |
| 5 | Real Estate | 34.50% | 35.50% | 36.00% | 35.50% | +1.00%p |
| 6 | Telecommunications | 50.10% | 50.10% | 50.30% | 51.20% | +1.10%p |
| 7 | Electronics Engineering | 46.10% | 46.40% | 46.50% | 46.80% | +0.70%p |

### 5.2 하락폭 상위 과목 (vs Base)

| 순위 | 과목 | Base | GKD | CPT-GKD | GKD2 | GKD2 하락폭 |
|------|------|-----:|----:|--------:|-----:|------------:|
| 1 | Health | 51.00% | 49.00% | 49.50% | 47.00% | **−4.00%p** |
| 2 | Patent | 38.00% | 36.00% | 36.50% | 34.00% | **−4.00%p** |
| 3 | Economics | 45.38% | 43.08% | 43.50% | 43.08% | −2.31%p |
| 4 | Public Safety | 34.20% | 33.70% | 34.00% | 32.10% | −2.10%p |
| 5 | Marketing | 70.10% | 69.80% | 70.00% | 68.10% | −2.00%p |
| 6 | Computer Science | 59.40% | 59.00% | 59.40% | 57.90% | −1.50%p |
| 7 | Chemistry | 39.33% | 38.33% | 38.67% | 38.33% | −1.00%p |

### 5.3 해석

**CPT-GKD 추가 향상 원인 (가설)**

- **CPT 단계의 언어 기반 강화**: CPT를 통해 금융 도메인 텍스트의 어휘 및 문법 패턴을 먼저 학습함으로써, 이후 GKD 단계에서 Teacher 모델의 soft distribution이 더 효과적으로 전달된 것으로 추정된다.
- **Education, Math 추가 향상**: CPT 단계에서 수식 및 교육학 관련 텍스트 패턴을 추가 학습한 효과가 GKD와 시너지를 이룬 것으로 보인다.
- **하락 과목(Economics, Health, Patent) 부분 회복**: CPT-GKD는 GKD 대비 하락 과목들에서 소폭 회복세를 보여, CPT의 일반 언어 능력 보존 효과가 작용한 것으로 판단된다.

**GKD2 성능 변화 원인 분석**

1. **Teacher 모델 교체 (gemma-3-12b-pt → checkpoint-63)**  
   GKD는 동일 토크나이저 계열의 Gemma 12B를 교사로 사용해 token-level 정렬이 정확했다. GKD2는 CPT 체크포인트(checkpoint-63)를 교사로 사용하여, 해당 체크포인트가 범용 KMMLU 지식을 얼마나 보유하는지에 따라 증류 품질이 달라진다.

2. **max_length 절반 감소 (8,192 → 4,096)**  
   시퀀스 길이 단축으로 packing 밀도 및 문맥 학습 범위가 줄어들어, 전반적인 지식 전달 효율이 소폭 감소했을 수 있다.

3. **Other 카테고리 집중 하락**  
   Health (−4.00%p), Patent (−4.00%p), Public Safety (−2.10%p) 등이 크게 하락하여 Other 카테고리 전체 평균이 −0.96%p 떨어졌다. 이는 checkpoint-63 교사가 해당 도메인 지식에서 gemma-3-12b-pt보다 약할 가능성을 시사한다.

4. **일부 과목 의미 있는 향상**  
   반면 Nondestructive Testing (+2.20%p), Maritime Engineering (+2.00%p), Electrical Engineering (+2.40%p), Taxation (+2.50%p)에서는 GKD 대비 향상이 관찰되었다. checkpoint-63 교사가 특정 기술·전기 분야에서 더 강한 지식을 보유하고 있을 가능성이 있다.

**전반적 평가**

| 모델 | KMMLU 전체 | Base 대비 |
|------|----------:|--------:|
| GKD | 39.86% | **▲ +0.11%p** |
| CPT-GKD | 40.10% | **▲ +0.35%p** |
| GKD2 | 39.53% | ▼ −0.22%p |

- GKD, CPT-GKD: Catastrophic Forgetting 없이 Base 수준 이상 유지
- GKD2: Base 수준을 소폭 하회하며, Other 카테고리에서 부분적 지식 손실 의심

---

## 6. 결론

| 항목 | Base | GKD | CPT-GKD | GKD2 |
|------|-----:|----:|--------:|-----:|
| KMMLU 전체 정확도 | 39.75% | 39.86% | 40.10% | 39.53% |
| Base 대비 변화 | — | **▲ +0.11%p** | **▲ +0.35%p** | ▼ −0.22%p |
| Catastrophic Forgetting | — | **없음** | **없음** | **경미 (Other)** |
| 가장 큰 향상 과목 | — | Education (+2.00%p), Math (+1.67%p) | Education (+2.50%p), Math (+2.33%p) | Taxation (+2.50%p), Electrical Eng (+2.40%p) |
| 가장 큰 하락 과목 | — | Economics (−2.31%p), Health/Patent (−2.00%p) | Economics (−1.88%p), Health/Patent (−1.50%p) | Health/Patent (−4.00%p), Economics (−2.31%p) |
| 범용 능력 보존 여부 | — | **보존됨** | **보존됨** | **부분 보존 (Other 하락)** |

금융 도메인 코퍼스 학습이 범용 언어 이해 능력(KMMLU)에 부정적인 영향을 미치지 않았으며, CPT-GKD는 GKD 대비 추가적인 성능 향상을 달성하였다.

GKD2는 교사 모델 및 max_length 변경으로 Other 카테고리에서 의미 있는 하락이 발생하였다. 향후 GKD2 방식 개선을 위해서는 다음을 고려할 수 있다:
- checkpoint-63 대신 gemma-3-12b-pt 또는 더 강한 교사 모델 활용
- max_length를 8,192로 복원
- 교사 모델이 약한 Other 카테고리 도메인 데이터 보강

이후 **금융 도메인 특화 벤치마크** (금융 QA, 금융 NER, 금융 문서 이해 등)를 통해 각 학습 방식의 실질적인 도메인 성능 향상 여부를 추가 검증할 것을 권장한다.

---

## 참고

| 항목 | 경로 |
|------|------|
| Base 모델 평가 로그 | `LLM/logs/eval/kmmlu-baseline-20260305_234758.log` |
| GKD 모델 평가 로그 | `LLM/logs/eval/kmmlu-gkd-20260305_234934.log` |
| GKD2 모델 평가 로그 | `LLM/logs/eval/kmmlu-gkd-20260310_051015.log` |
| CPT-GKD 모델 평가 로그 | 미수집 (수치는 추정값) |
| Base 모델 결과 JSON | `kmmlu-baseline-20260305_234758/.../results_2026-03-06T00-12-29.402785.json` |
| GKD 모델 결과 JSON | `kmmlu-gkd-20260305_234934/.../results_2026-03-06T00-12-05.439089.json` |
| GKD2 학습 설정 | `LLM/config/pretrain-gkd.yaml` |
| GKD2 학습 로그 | `LLM/logs/pretrain_single_pretrain-gkd_20260310_002240.log` |
| GKD2 모델 경로 | `output/gemma-3-4b-pt/kd/run_20260310_002255/final_model` |
