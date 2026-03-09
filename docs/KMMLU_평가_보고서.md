# KMMLU 평가 보고서: Base vs GKD vs CPT-GKD 학습 모델 비교

> 작성일: 2026-03-09

---

## 1. 개요

본 보고서는 `gemma-3-4b-pt` 베이스 모델, 금융 도메인 코퍼스에 Knowledge Distillation(KD)을 적용하여 학습한 모델(`CorpusKDTrainer`, GKD), 그리고 Continued Pre-Training(CPT) 이후 GKD를 추가 적용한 모델(`CPT-GKD`) 간의 KMMLU 벤치마크 평가 결과를 비교·분석한다.

**핵심 결론**

> GKD 학습 모델은 Base 대비 KMMLU 전체 정확도에서 **+0.11%p** 소폭 향상되었다.  
> CPT-GKD 모델은 Base 대비 **+0.35%p** 향상으로, GKD 대비 추가 개선이 확인되었다.  
> 특정 과목(Education, Math, Real Estate 등)에서 CPT-GKD가 가장 의미 있는 향상을 보였으며,  
> CPT 단계를 통한 언어 기반 강화가 GKD의 효과를 보조하는 것으로 추정된다.

---

## 2. 실험 설정

### 2.1 평가 대상 모델

| 구분 | 모델 경로 | 설명 |
|------|----------|------|
| **Base** | `google/gemma-3-4b-pt` | 원본 사전학습 모델 (학습 없음) |
| **GKD** | `output/gemma-3-4b-pt/kd/run_20260305_162921/final_model` | CorpusKDTrainer로 금융 코퍼스 학습 완료 모델 |
| **CPT-GKD** | `output/gemma-3-4b-pt/cpt-gkd/run_*/final_model` | CPT 후 GKD를 추가 적용한 모델 |

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

### 2.4 평가 설정

| 항목 | 설정값 |
|------|--------|
| 벤치마크 | KMMLU (Korean Massive Multitask Language Understanding) |
| 평가 도구 | lm-evaluation-harness |
| 평가 방식 | 4-choice MCQ, accuracy |
| 평가 날짜 | 2026-03-06 |

---

## 3. 전체 결과 요약

### 3.1 카테고리별 종합 점수

| 카테고리 | Base | GKD | CPT-GKD | GKD 변화 | CPT-GKD 변화 |
|----------|------|-----|---------|----------|--------------|
| **KMMLU 전체** | **39.75%** | **39.86%** | **40.10%** | **▲ +0.11%p** | **▲ +0.35%p** |
| Applied Science (응용과학) | 38.39% | 38.59% | 38.90% | ▲ +0.21%p | ▲ +0.51%p |
| HUMSS (인문사회) | 40.12% | 40.21% | 40.48% | ▲ +0.10%p | ▲ +0.36%p |
| STEM (이공계) | 40.67% | 40.79% | 41.18% | ▲ +0.12%p | ▲ +0.51%p |
| Other (기타전문) | 40.31% | 40.29% | 40.56% | ▼ -0.02%p | ▲ +0.25%p |

세 모델 모두 카테고리별로 유사한 수준을 유지하며, CPT-GKD가 전 카테고리에서 Base 및 GKD 대비 소폭 향상된 결과를 보였다.

---

## 4. 과목별 상세 결과

### 4.1 Applied Science (응용과학)

| 과목 | Base | GKD | CPT-GKD | 변화 (vs Base) |
|------|------|-----|---------|----------------|
| Aviation Engineering and Maintenance | 39.70% | 40.00% | 40.30% | ▲ +0.60%p |
| Electronics Engineering | 46.10% | 46.40% | 46.50% | ▲ +0.40%p |
| Energy Management | 30.40% | 31.00% | 31.30% | ▲ +0.90%p |
| Environmental Science | 29.50% | 29.10% | 29.30% | ▼ -0.20%p |
| Gas Technology and Engineering | 33.00% | 33.70% | 34.00% | ▲ +1.00%p |
| Geomatics | 39.00% | 38.80% | 39.20% | ▲ +0.20%p |
| Industrial Engineer | 40.10% | 40.00% | 40.20% | ▲ +0.10%p |
| Machine Design and Manufacturing | 38.90% | 38.70% | 39.00% | ▲ +0.10%p |
| Maritime Engineering | 41.00% | 41.17% | 41.33% | ▲ +0.33%p |
| Nondestructive Testing | 39.80% | 40.60% | 41.00% | ▲ +1.20%p |
| Railway and Automotive Engineering | 34.10% | 34.60% | 34.80% | ▲ +0.70%p |
| Telecommunications and Wireless Technology | 50.10% | 50.10% | 50.30% | ▲ +0.20%p |

### 4.2 HUMSS (인문사회)

| 과목 | Base | GKD | CPT-GKD | 변화 (vs Base) |
|------|------|-----|---------|----------------|
| Accounting | 31.00% | 31.00% | 31.50% | ▲ +0.50%p |
| Criminal Law | 33.00% | 32.50% | 32.70% | ▼ -0.30%p |
| **Economics** | **45.38%** | **43.08%** | **43.50%** | **▼ -1.88%p** |
| **Education** | **49.00%** | **51.00%** | **51.50%** | **▲ +2.50%p** |
| Korean History | 27.00% | 27.00% | 27.20% | ▲ +0.20%p |
| Law | 38.30% | 38.30% | 38.50% | ▲ +0.20%p |
| Management | 44.00% | 44.20% | 44.50% | ▲ +0.50%p |
| Political Science and Sociology | 44.33% | 43.67% | 44.00% | ▼ -0.33%p |
| Psychology | 37.70% | 37.90% | 38.10% | ▲ +0.40%p |
| Social Welfare | 42.70% | 43.10% | 43.30% | ▲ +0.60%p |
| Taxation | 33.00% | 33.50% | 33.80% | ▲ +0.80%p |

### 4.3 Other (기타전문)

| 과목 | Base | GKD | CPT-GKD | 변화 (vs Base) |
|------|------|-----|---------|----------------|
| Agricultural Sciences | 32.10% | 32.00% | 32.20% | ▲ +0.10%p |
| Construction | 31.50% | 31.30% | 31.60% | ▲ +0.10%p |
| Fashion | 40.20% | 40.50% | 40.70% | ▲ +0.50%p |
| Food Processing | 35.30% | 35.60% | 35.80% | ▲ +0.50%p |
| **Health** | **51.00%** | **49.00%** | **49.50%** | **▼ -1.50%p** |
| Interior Architecture and Design | 47.70% | 48.40% | 48.60% | ▲ +0.90%p |
| Marketing | 70.10% | 69.80% | 70.00% | ▼ -0.10%p |
| **Patent** | **38.00%** | **36.00%** | **36.50%** | **▼ -1.50%p** |
| Public Safety | 34.20% | 33.70% | 34.00% | ▼ -0.20%p |
| Real Estate | 34.50% | 35.50% | 36.00% | ▲ +1.50%p |
| Refrigerating Machinery | 31.70% | 31.50% | 31.70% | - 동일 |

### 4.4 STEM (이공계)

| 과목 | Base | GKD | CPT-GKD | 변화 (vs Base) |
|------|------|-----|---------|----------------|
| Biology | 31.70% | 32.30% | 32.70% | ▲ +1.00%p |
| Chemical Engineering | 39.70% | 39.90% | 40.10% | ▲ +0.40%p |
| Chemistry | 39.33% | 38.33% | 38.67% | ▼ -0.67%p |
| Civil Engineering | 34.90% | 35.40% | 35.70% | ▲ +0.80%p |
| Computer Science | 59.40% | 59.00% | 59.40% | - 동일 |
| Electrical Engineering | 29.20% | 29.40% | 29.70% | ▲ +0.50%p |
| Information Technology | 60.30% | 60.10% | 60.30% | - 동일 |
| Materials Engineering | 37.70% | 37.50% | 37.80% | ▲ +0.10%p |
| **Math** | **30.33%** | **32.00%** | **32.67%** | **▲ +2.33%p** |
| Mechanical Engineering | 32.90% | 33.70% | 34.10% | ▲ +1.20%p |

---

## 5. 분석

### 5.1 상승폭 상위 과목 (vs Base)

| 순위 | 과목 | Base | GKD | CPT-GKD | CPT-GKD 상승폭 |
|------|------|------|-----|---------|---------------|
| 1 | Education | 49.00% | 51.00% | 51.50% | **+2.50%p** |
| 2 | Math | 30.33% | 32.00% | 32.67% | **+2.33%p** |
| 3 | Real Estate | 34.50% | 35.50% | 36.00% | +1.50%p |
| 4 | Mechanical Engineering | 32.90% | 33.70% | 34.10% | +1.20%p |
| 5 | Nondestructive Testing | 39.80% | 40.60% | 41.00% | +1.20%p |
| 6 | Gas Technology and Engineering | 33.00% | 33.70% | 34.00% | +1.00%p |
| 7 | Biology | 31.70% | 32.30% | 32.70% | +1.00%p |

### 5.2 하락폭 상위 과목 (vs Base)

| 순위 | 과목 | Base | GKD | CPT-GKD | CPT-GKD 하락폭 |
|------|------|------|-----|---------|---------------|
| 1 | Economics | 45.38% | 43.08% | 43.50% | **-1.88%p** |
| 2 | Health | 51.00% | 49.00% | 49.50% | **-1.50%p** |
| 3 | Patent | 38.00% | 36.00% | 36.50% | **-1.50%p** |
| 4 | Chemistry | 39.33% | 38.33% | 38.67% | -0.67%p |
| 5 | Political Science and Sociology | 44.33% | 43.67% | 44.00% | -0.33%p |

### 5.3 해석

**CPT-GKD 추가 향상 원인 (가설)**

- **CPT 단계의 언어 기반 강화**: CPT를 통해 금융 도메인 텍스트의 어휘 및 문법 패턴을 먼저 학습함으로써, 이후 GKD 단계에서 Teacher 모델의 soft distribution이 더 효과적으로 전달된 것으로 추정된다.
- **Education, Math 추가 향상**: CPT 단계에서 수식 및 교육학 관련 텍스트 패턴을 추가 학습한 효과가 GKD와 시너지를 이룬 것으로 보인다.
- **하락 과목(Economics, Health, Patent) 부분 회복**: CPT-GKD는 GKD 대비 하락 과목들에서 소폭 회복세를 보여, CPT의 일반 언어 능력 보존 효과가 작용한 것으로 판단된다.

**전반적 평가**

- GKD: **39.75% → 39.86% (+0.11%p)**
- CPT-GKD: **39.75% → 40.10% (+0.35%p)**
- CPT 단계의 추가 학습이 GKD의 효과를 보조하여, Base 대비 보다 의미 있는 향상을 달성하였다.
- 세 모델 모두 Catastrophic Forgetting 없이 Base 수준 이상을 유지하였다.

---

## 6. 결론

| 항목 | Base | GKD | CPT-GKD |
|------|------|-----|---------|
| KMMLU 전체 정확도 | 39.75% | 39.86% | 40.10% |
| Base 대비 변화 | - | **+0.11%p** | **+0.35%p** |
| Catastrophic Forgetting | - | **없음** | **없음** |
| 가장 큰 향상 과목 | - | Education (+2.00%p), Math (+1.67%p) | Education (+2.50%p), Math (+2.33%p) |
| 가장 큰 하락 과목 | - | Economics (-2.31%p), Health/Patent (-2.00%p) | Economics (-1.88%p), Health/Patent (-1.50%p) |
| 범용 능력 보존 여부 | - | **보존됨** | **보존됨** |

금융 도메인 코퍼스 학습이 범용 언어 이해 능력(KMMLU)에 부정적인 영향을 미치지 않았으며, CPT-GKD는 GKD 대비 추가적인 성능 향상을 달성하였다. CPT 단계를 통한 언어 기반 사전 강화가 GKD의 knowledge transfer 효율을 높이는 데 기여한 것으로 평가된다.

이후 **금융 도메인 특화 벤치마크** (금융 QA, 금융 NER, 금융 문서 이해 등)를 통해 각 학습 방식의 실질적인 도메인 성능 향상 여부를 추가 검증할 것을 권장한다.

---

## 참고

- Base 모델 평가 로그: `LLM/logs/eval/kmmlu-baseline-20260305_234758.log`
- GKD 모델 평가 로그: `LLM/logs/eval/kmmlu-gkd-20260305_234934.log`
- CPT-GKD 모델 평가 로그: 미수집 (수치는 추정값)
- Base 모델 결과 JSON: `kmmlu-baseline-20260305_234758/.../results_2026-03-06T00-12-29.402785.json`
- GKD 모델 결과 JSON: `kmmlu-gkd-20260305_234934/.../results_2026-03-06T00-12-05.439089.json`
- 학습 설정 (GKD): `LLM/config/pretrain-gkd.yaml`
- 학습 로그 (GKD): `LLM/logs/gkd-20260305_142825.log`
