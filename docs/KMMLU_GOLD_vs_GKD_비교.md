# KMMLU 평가 비교 보고서: GOLD vs GKD

> 작성일: 2026-03-09

---

## 1. 개요

본 보고서는 동일한 `gemma-3-4b-pt` 베이스 모델에서 출발한 두 가지 학습 방식 — **GOLD** (Cross-Tokenizer ULD Knowledge Distillation + NLL, LoRA PiSSA)와 **GKD** (Same-Tokenizer JSD Knowledge Distillation, Full Fine-tuning) — 의 KMMLU 벤치마크 결과를 비교·분석한다.

**핵심 결론**

> GKD 모델은 GOLD 모델 대비 KMMLU 전체 정확도에서 **+15.91%p** 높은 성능을 보였다.  
> GOLD 모델의 전체 정확도(23.95%)는 4지선다 랜덤 기준선(25.00%)에 근접하여,  
> 해당 학습 방식이 KMMLU 지식 습득에는 거의 기여하지 못한 것으로 판단된다.  
> 단, `korean_history` 과목에서는 GOLD가 GKD를 **+9.0%p** 앞서는 유일한 역전 사례가 확인되었다.

---

## 2. 실험 설정

### 2.1 평가 대상 모델

| 구분 | 모델 경로 | 설명 |
|------|----------|------|
| **Base** | `google/gemma-3-4b-pt` | 원본 사전학습 모델 (학습 없음) |
| **GKD** | `output/gemma-3-4b-pt/kd/run_20260305_162921/final_model` | 동일 토크나이저 JSD 증류, Full Fine-tuning |
| **GOLD** | `output/gemma-3-4b-pt/gold/run_20260306_141411/final_model` | 크로스-토크나이저 ULD 증류, LoRA PiSSA |

### 2.2 학습 설정 비교

| 항목 | GKD | GOLD |
|------|-----|------|
| Student 모델 | `gemma-3-4b-pt` (4B) | `gemma-3-4b-pt` (4B) |
| Teacher 모델 | `gemma-3-12b-pt` (12B, 동일 토크나이저) | `Qwen3.5-9B-Base` (9B, 크로스 토크나이저) |
| Loss 함수 | JSD (Jensen-Shannon Divergence, β=0.5) | ULD (Sorted L1) + NLL CE (weight=0.5) |
| 파인튜닝 방식 | Full Fine-tuning | LoRA PiSSA (r=128, α=256) |
| max_length | 8,192 tokens | 4,096 tokens |
| 학습 데이터 | 금융 코퍼스 829,331건 | 금융 코퍼스 829,331건 (동일) |
| Optimizer | adamw_8bit | paged_adamw_32bit |
| Learning rate | 2.0e-6 (cosine) | 2.0e-6 (cosine) |
| Epochs | 1 | 1 |
| Run ID | run_20260305_162921 | run_20260306_141411 |

### 2.3 평가 설정

| 항목 | 설정값 |
|------|--------|
| 벤치마크 | KMMLU (Korean Massive Multitask Language Understanding) |
| 평가 방식 | 4-choice MCQ, loglikelihood |
| Few-shot | 5-shot |
| 평가 도구 | lm-evaluation-harness (vLLM 백엔드) |
| GKD 평가일 | 2026-03-06 |
| GOLD 평가일 | 2026-03-09 |

---

## 3. 전체 결과 요약

### 3.1 카테고리별 종합 점수

| 카테고리 | Base | GKD | GOLD | GKD vs Base | GOLD vs Base | GOLD vs GKD |
|----------|-----:|----:|-----:|------------:|-------------:|------------:|
| **KMMLU 전체** | **39.75%** | **39.86%** | **23.95%** | **▲ +0.11%p** | **▼ −15.80%p** | **▼ −15.91%p** |
| Applied Science | 38.39% | 38.59% | 23.29% | ▲ +0.21%p | ▼ −15.10%p | ▼ −15.30%p |
| HUMSS | 40.12% | 40.21% | 23.20% | ▲ +0.10%p | ▼ −16.92%p | ▼ −17.01%p |
| Other | 40.31% | 40.29% | 24.74% | ▼ −0.02%p | ▼ −15.57%p | ▼ −15.55%p |
| STEM | 40.67% | 40.79% | 24.45% | ▲ +0.12%p | ▼ −16.22%p | ▼ −16.34%p |

> **랜덤 기준선**: 4지선다 무작위 선택 시 기대 정확도 = **25.00%**  
> GOLD의 전 카테고리 점수가 25% 미만이거나 25%에 근접하여, 실질적 학습 효과가 거의 없다.

---

## 4. 과목별 상세 결과

### 4.1 Applied Science (응용과학)

| 과목 | Base | GKD | GOLD | GOLD vs GKD |
|------|-----:|----:|-----:|------------:|
| Aviation Engineering and Maintenance | 39.70% | 40.00% | 31.50% | ▼ −8.50%p |
| Electronics Engineering | 46.10% | 46.40% | 27.50% | ▼ −18.90%p |
| Energy Management | 30.40% | 31.00% | 28.40% | ▼ −2.60%p |
| Environmental Science | 29.50% | 29.10% | 13.40% | ▼ −15.70%p |
| Gas Technology and Engineering | 33.00% | 33.70% | 26.60% | ▼ −7.10%p |
| Geomatics | 39.00% | 38.80% | 30.30% | ▼ −8.50%p |
| Industrial Engineer | 40.10% | 40.00% | 16.20% | ▼ −23.80%p |
| Machine Design and Manufacturing | 38.90% | 38.70% | 21.90% | ▼ −16.80%p |
| Maritime Engineering | 41.00% | 41.17% | 18.83% | ▼ −22.34%p |
| Nondestructive Testing | 39.80% | 40.60% | 19.80% | ▼ −20.80%p |
| Railway and Automotive Engineering | 34.10% | 34.60% | 18.30% | ▼ −16.30%p |
| Telecommunications and Wireless Technology | 50.10% | 50.10% | 25.00% | ▼ −25.10%p |

### 4.2 HUMSS (인문사회)

| 과목 | Base | GKD | GOLD | GOLD vs GKD |
|------|-----:|----:|-----:|------------:|
| Accounting | 31.00% | 31.00% | 29.00% | ▼ −2.00%p |
| Criminal Law | 33.00% | 32.50% | 22.00% | ▼ −10.50%p |
| Economics | 45.38% | 43.08% | 25.38% | ▼ −17.69%p |
| Education | 49.00% | 51.00% | 35.00% | ▼ −16.00%p |
| **Korean History** | **27.00%** | **27.00%** | **36.00%** | **▲ +9.00%p** ← GOLD 우세 |
| Law | 38.30% | 38.30% | 23.80% | ▼ −14.50%p |
| Management | 44.00% | 44.20% | 20.90% | ▼ −23.30%p |
| Political Science and Sociology | 44.33% | 43.67% | 26.67% | ▼ −17.00%p |
| Psychology | 37.70% | 37.90% | 22.90% | ▼ −15.00%p |
| Social Welfare | 42.70% | 43.10% | 21.10% | ▼ −22.00%p |
| Taxation | 33.00% | 33.50% | 23.00% | ▼ −10.50%p |

### 4.3 Other (기타전문)

| 과목 | Base | GKD | GOLD | GOLD vs GKD |
|------|-----:|----:|-----:|------------:|
| Agricultural Sciences | 32.10% | 32.00% | 19.00% | ▼ −13.00%p |
| **Construction** | **31.50%** | **31.30%** | **33.80%** | **▲ +2.50%p** ← GOLD 우세 |
| Fashion | 40.20% | 40.50% | 20.70% | ▼ −19.80%p |
| Food Processing | 35.30% | 35.60% | 29.00% | ▼ −6.60%p |
| Health | 51.00% | 49.00% | 28.00% | ▼ −21.00%p |
| Interior Architecture and Design | 47.70% | 48.40% | 25.90% | ▼ −22.50%p |
| Marketing | 70.10% | 69.80% | 16.30% | ▼ −53.50%p |
| Patent | 38.00% | 36.00% | 23.00% | ▼ −13.00%p |
| Public Safety | 34.20% | 33.70% | 27.40% | ▼ −6.30%p |
| Real Estate | 34.50% | 35.50% | 23.00% | ▼ −12.50%p |
| Refrigerating Machinery | 31.70% | 31.50% | 26.00% | ▼ −5.50%p |

### 4.4 STEM (이공계)

| 과목 | Base | GKD | GOLD | GOLD vs GKD |
|------|-----:|----:|-----:|------------:|
| Biology | 31.70% | 32.30% | 23.00% | ▼ −9.30%p |
| Chemical Engineering | 39.70% | 39.90% | 22.80% | ▼ −17.10%p |
| Chemistry | 39.33% | 38.33% | 25.67% | ▼ −12.67%p |
| Civil Engineering | 34.90% | 35.40% | 16.60% | ▼ −18.80%p |
| Computer Science | 59.40% | 59.00% | 33.00% | ▼ −26.00%p |
| Ecology | — | 43.90% | 20.70% | ▼ −23.20%p |
| Electrical Engineering | 29.20% | 29.40% | 20.60% | ▼ −8.80%p |
| Information Technology | 60.30% | 60.10% | 35.90% | ▼ −24.20%p |
| Materials Engineering | 37.70% | 37.50% | 29.10% | ▼ −8.40%p |
| **Math** | **30.33%** | **32.00%** | **31.00%** | ▼ −1.00%p (거의 동등) |
| Mechanical Engineering | 32.90% | 33.70% | 15.70% | ▼ −18.00%p |

---

## 5. 분석

### 5.1 GOLD가 GKD를 앞선 과목 (역전 사례)

| 과목 | GKD | GOLD | 차이 |
|------|----:|-----:|-----:|
| Korean History | 27.00% | **36.00%** | **▲ +9.00%p** |
| Construction | 31.30% | **33.80%** | **▲ +2.50%p** |
| Math | 32.00% | 31.00% | ▼ −1.00%p (통계적 동등) |

전체 45개 과목 중 GOLD가 GKD를 초과한 과목은 **2개**(Korean History, Construction)에 불과하며, Math는 오차 범위 내 동등 수준이다.

### 5.2 GOLD 열세 최대 과목 (Top 10)

| 순위 | 과목 | GKD | GOLD | 차이 |
|------|------|----:|-----:|-----:|
| 1 | Marketing | 69.80% | 16.30% | **▼ −53.50%p** |
| 2 | Telecommunications | 50.10% | 25.00% | ▼ −25.10%p |
| 3 | Computer Science | 59.00% | 33.00% | ▼ −26.00%p |
| 4 | Information Technology | 60.10% | 35.90% | ▼ −24.20%p |
| 5 | Ecology | 43.90% | 20.70% | ▼ −23.20%p |
| 6 | Industrial Engineer | 40.00% | 16.20% | ▼ −23.80%p |
| 7 | Management | 44.20% | 20.90% | ▼ −23.30%p |
| 8 | Interior Architecture | 48.40% | 25.90% | ▼ −22.50%p |
| 9 | Maritime Engineering | 41.17% | 18.83% | ▼ −22.34%p |
| 10 | Social Welfare | 43.10% | 21.10% | ▼ −22.00%p |

### 5.3 해석

#### GOLD 성능 저하 원인 분석

1. **LoRA PiSSA의 파라미터 효율 한계**  
   GKD는 Full Fine-tuning으로 전체 파라미터를 갱신한 반면, GOLD는 LoRA (r=128)로 제한된 파라미터만 학습했다. KMMLU처럼 폭넓은 지식을 요구하는 벤치마크에서는 Full Fine-tuning이 더 유리하다.

2. **크로스-토크나이저 ULD 정렬의 한계**  
   GOLD는 Student(Gemma, vocab 256K)와 Teacher(Qwen3.5-9B, vocab 151K)의 토크나이저가 달라 logit 정렬 시 정보 손실이 발생한다. GKD는 동일 토크나이저 계열(Gemma 4B/12B)을 사용해 정확한 token-level 증류가 가능하다.

3. **max_length 절반 (4096 vs 8192)**  
   GOLD는 4,096 tokens로 학습하여 longer context 지식 패턴 학습이 상대적으로 제한되었다.

4. **랜덤 기준선 근접 현상**  
   GOLD 전체 정확도 23.95%는 랜덤 기준선 25.00%보다 낮다. 이는 LoRA 학습 과정에서 베이스 모델의 기존 지식이 일부 손상(Catastrophic Forgetting)된 가능성을 시사한다.

#### Korean History 역전 현상

- GOLD 36.00% vs GKD 27.00% (**+9%p**)  
- 가설: Qwen3.5-9B-Base 교사 모델이 한국사 도메인에서 Gemma-3-12b-pt보다 더 강한 사전학습 지식을 보유하고 있어, ULD를 통해 효과적으로 전달된 것으로 추정된다.

#### Marketing 극단적 열세

- GKD 69.80% vs GOLD 16.30% (**−53.5%p**)  
- GKD의 Teacher인 Gemma-3-12b-pt가 마케팅/비즈니스 도메인에서 특히 강한 반면, GOLD는 해당 도메인 지식을 거의 전달받지 못한 것으로 보인다.

---

## 6. 결론

| 항목 | Base | GKD | GOLD |
|------|-----:|----:|-----:|
| **KMMLU 전체 정확도** | **39.75%** | **39.86%** | **23.95%** |
| Base 대비 변화 | — | **▲ +0.11%p** | **▼ −15.80%p** |
| 랜덤 기준선(25%) 대비 | +14.75%p | +14.86%p | **−1.05%p** |
| Catastrophic Forgetting | — | **없음** | **있음 (의심)** |
| GKD 대비 우세 과목 수 | — | — | 2 / 45개 (4.4%) |
| 가장 큰 역전 과목 | — | — | Korean History (+9.0%p) |
| 가장 큰 열세 과목 | — | — | Marketing (−53.5%p) |

**GOLD 방식의 현재 설정은 KMMLU 범용 지식 유지에 적합하지 않다.** GKD 방식(동일 토크나이저 JSD 증류 + Full Fine-tuning)이 범용 벤치마크에서 Base 수준을 보존하는 데 훨씬 효과적임이 확인되었다.

GOLD 방식의 개선을 위해서는 아래 방향을 고려할 수 있다:
- LoRA rank 증가 또는 Full Fine-tuning으로 전환
- max_length 8,192로 확장
- ULD top-k 조정 또는 hybrid matched/unmatched loss 적용
- Teacher 모델을 동일 Gemma 계열로 변경하거나 vocab overlap을 높이는 방향 탐색

---

## 참고

| 항목 | 경로 |
|------|------|
| Base 평가 로그 | `LLM/logs/eval/kmmlu-baseline-20260305_234758.log` |
| GKD 평가 로그 | `LLM/logs/eval/kmmlu-gkd-20260305_234934.log` |
| GOLD 평가 로그 | `LLM/logs/eval/kmmlu-gold-20260309_224336.log` |
| GKD 학습 설정 | `LLM/config/pretrain-gkd.yaml` |
| GOLD 학습 설정 | `LLM/config/pretrain-gold.yaml` |
| GOLD adapter | `output/gemma-3-4b-pt/gold/run_20260306_141411/final_model` |
| GOLD merged | `output/gemma-3-4b-pt/gold/run_20260306_141411/merged_model` |
| 이전 보고서 | `LLM/docs/KMMLU_평가_보고서.md` (Base vs GKD vs CPT-GKD) |
