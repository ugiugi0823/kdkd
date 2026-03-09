# CorpusKDTrainer 기술 타당성 보고서

> 작성일: 2026-03-05

---

## 1. 개요

본 보고서는 금융 도메인 코퍼스 지속 사전학습(Corpus Continual Pretraining)에 Knowledge Distillation(KD)을 적용하기 위해 설계된 `CorpusKDTrainer`의 기술적 타당성을 서술한다.

**핵심 결론**

> 코퍼스 사전학습 + KD 환경에서는 off-policy, forward-pass-only 방식이 유일하게 타당하다. `CorpusKDTrainer`는 이를 최소한의 변경으로 구현한다.

---

## 2. 배경: 왜 코퍼스 사전학습에 KD가 필요한가

### 2.1 NLL Loss의 한계

기존 표준 사전학습은 다음 NLL(Negative Log-Likelihood) loss를 사용한다.

```
L_NLL = -sum( log P_S(y_t | y_<t) )
```

이 방식은 정답 토큰 하나에만 gradient를 전달하므로 vocab 전체의 확률 분포 정보를 버린다. 정답 토큰 외 나머지 토큰들이 얼마나 그럴듯한지에 대한 신호가 없다.

### 2.2 KD가 제공하는 추가 정보

교사 모델이 각 위치에서 출력하는 soft probability distribution P_T는 vocab 전체에 걸친 의미적 관계를 내포한다.

```
예시) "내일 {주식/채권/펀드/...}을 매수할 예정이다"

NLL:  "주식"에만 gradient (정답 토큰)
KD:   P_T("주식")=0.45, P_T("채권")=0.23, P_T("펀드")=0.18, ...
      -> 학생이 금융 어휘 간 의미적 유사성까지 학습
```

이는 Hinton et al. (2015)이 지적한 "dark knowledge"로, 소용량 학생 모델이 대용량 교사의 내재 지식을 흡수하는 핵심 메커니즘이다.

---

## 3. 왜 TRL GKDTrainer를 사용하지 않는가

### 3.1 On-policy 방식과 PT 모델의 구조적 불일치

TRL의 `GKDTrainer`는 `lmbda` 파라미터로 on-policy 비율을 제어한다.

```
lmbda = 0.0  ->  off-policy: 사전 정의된 corpus 데이터만 사용
lmbda > 0.0  ->  on-policy:  student.generate() 로 텍스트 생성 후 사용
```

on-policy 학습의 전제 조건: 학생 모델이 `generate()`를 통해 의미 있는 텍스트를 생성할 수 있어야 한다.

그러나 pretrain-only(PT) 모델은 이 전제를 충족할 수 없다.

| 모델 종류 | generate() 동작 | 이유 |
|---------|----------------|------|
| IT 모델 | 정상 종료 | RLHF/SFT로 EOS 예측 학습됨 |
| PT 모델 | **무한 반복** | EOS 종료 신호 미학습, 토큰 패턴 루프 발생 |

PT 모델에서 `lmbda > 0`으로 설정하면 `student.generate()`가 무한 반복 텍스트를 생성하고, 그 위에서 교사의 distribution을 구하려 하므로 학습 신호 자체가 무효화된다.

### 3.2 코퍼스 사전학습 환경과의 구조적 충돌

`GKDTrainer`는 chat/SFT 스타일을 가정하여 설계되었으며, 다음의 제약이 코퍼스 사전학습과 충돌한다.

| GKDTrainer 제약 | 코퍼스 사전학습에서의 문제 |
|---------------|------------------------|
| `packing=False` 강제 | max_length=32768 윈도우에 문서 1개만 -> 패딩 낭비, 학습 효율 급감 |
| `messages` 형식 강제 | raw text를 chat 형식으로 래핑 -> 불필요한 전처리, 분포 왜곡 |
| on-policy 모드 | PT 모델 + 금융 코퍼스에서 구조적으로 불가 |

**결론: GKDTrainer는 코퍼스 사전학습에 적용 불가능한 도구다.**

---

## 4. CorpusKDTrainer 설계 원칙

### 4.1 핵심 설계: Off-policy Forward-pass Only

`CorpusKDTrainer`는 `compute_loss()` 한 메서드만 오버라이드한다.

```
학습 루프 한 스텝:

  corpus 배치 입력 x = [w_1, w_2, ..., w_T]
          |
          +-> Student(x) -> P_S(t)   (gradient O)
          |
          +-> Teacher(x) -> P_T(t)   (gradient X, torch.no_grad)
                   |
                   +-> JSD(P_S, P_T) -> Loss -> 역전파
```

`model.generate()` 호출 없음. 동일한 코퍼스 배치를 두 모델에 동시에 forward하여 분포 간 거리를 최소화하는 구조다.

### 4.2 기존 학습 인프라의 완전 보존

`SFTTrainer`를 상속함으로써 다음이 변경 없이 유지된다.

| 구성 요소 | 유지 여부 |
|---------|---------|
| packing=True (32768 토큰 윈도우) | 유지 |
| DeepSpeed ZeRO3 + CPU offload | 유지 |
| Liger Kernel (Gemma3 최적화) | 유지 |
| PiSSA LoRA 초기화 | 유지 |
| Flash Attention 2 | 유지 |
| gradient checkpointing | 유지 |

**변경된 것은 오직 loss 계산 함수 하나뿐이다.**

---

## 5. Loss 함수의 수학적 근거: JSD

### 5.1 JSD 정의

Jensen-Shannon Divergence는 두 확률 분포 간 대칭적 거리를 측정한다.

```
M_t = beta * P_S(t) + (1-beta) * P_T(t)          # 혼합 분포

L_JSD = beta * KL(P_S || M) + (1-beta) * KL(P_T || M)
```

### 5.2 beta에 따른 학습 특성

| beta 값 | 등가 Loss | 특성 |
|--------|---------|-----|
| 0.0 | Forward KL에 근접 | P_T의 커버리지 보존. 교사가 가능성 있다고 보는 토큰을 학생이 0에 수렴하지 않도록 강제 |
| 0.5 | Symmetric JSD | 양방향 균형. 기본값으로 권장 |
| 1.0 | Reverse KL에 근접 | 학생 분포의 최빈값 집중 |

### 5.3 NLL vs JSD 비교

```
NLL:  L = -log P_S(y_t)
      -> 정답 토큰 1개에만 신호

JSD:  L = beta * KL(P_S||M) + (1-beta) * KL(P_T||M)
      -> vocab V 전체 분포에 신호
```

JSD는 정답 토큰 외에도 교사가 부여한 확률값 전체를 학생에게 전달한다. 이를 통해 동일 문맥에서 의미적으로 유사한 토큰들이 유사한 확률을 갖도록 학습이 유도된다.

### 5.4 Temperature Scaling

```python
if self.temperature != 1.0:
    teacher_logits = teacher_logits / self.temperature
```

temperature T > 1.0으로 교사의 logit을 나누면 분포가 평탄(softer)해진다. 이는 교사 분포의 "dark knowledge"를 더 넓게 퍼뜨려 학생이 흡수할 정보량을 증가시킨다. Hinton et al. (2015)에서 원래 제안된 기법이다.

---

## 6. 구현 세부 사항의 기술적 타당성

### 6.1 Teacher requires_grad=False

```python
for param in teacher.parameters():
    param.requires_grad = False
teacher.eval()
```

교사 모델의 파라미터에 gradient를 계산하지 않으므로 optimizer state(momentum, variance 등)가 생성되지 않는다. 이로 인해 ZeRO3 CPU offload 대상에서 제외되어 메모리 효율이 극대화된다.

효과: 27B 교사 모델도 gradient 없이 ~54GB(BF16)만 차지. ZeRO3 분산으로 GPU당 ~3.4GB.

### 6.2 유효 토큰 마스킹

```python
mask = (labels != -100)   # packing 시 문서 경계 패딩 제외
loss = jsd_per_token[mask].mean()
```

packing 환경에서는 32768 토큰 윈도우 내에 여러 문서가 연결된다. 문서 경계의 패딩 위치(labels=-100)는 loss 계산에서 제외하여 문서 간 context leakage 방지 및 정확한 loss scale 유지를 보장한다.

### 6.3 Numerical Stability

```python
s_log_prob = F.log_softmax(student_logits, dim=-1)   # log-space 계산
t_log_prob = F.log_softmax(teacher_logits, dim=-1)
m_prob = (beta * s_prob + (1-beta) * t_prob).clamp(min=1e-8).log()
```

- `log_softmax`로 직접 계산하여 underflow 방지
- 혼합 분포 M에 `clamp(min=1e-8)` 적용으로 log(0) 방지
- `F.kl_div(reduction="none")`으로 토큰 단위 loss 계산 후 마스킹

---

## 7. 대안 방식과의 비교

### 7.1 TRL GKDTrainer (on-policy)

| 항목 | GKDTrainer (on-policy) | CorpusKDTrainer |
|-----|----------------------|----------------|
| PT 모델 호환성 | 불가 (무한 반복) | 가능 (forward만 사용) |
| packing 지원 | 강제 비활성화 | 완전 지원 |
| raw text 형식 | messages 강제 | 그대로 사용 |
| 학습 안정성 | PT 모델에서 불안정 | 안정 |

### 7.2 단순 NLL (KD 미적용)

| 항목 | NLL only | CorpusKDTrainer (JSD) |
|-----|---------|----------------------|
| 학습 신호 | 정답 토큰 1개 | vocab 전체 분포 |
| 의미적 관계 학습 | 간접적 | 직접적 |
| 수렴 속도 | 기준 | 동등 또는 빠름 |

### 7.3 Semi-on-policy (Teacher 생성 데이터로 Student 학습)

질문만 입력하고 IT 교사가 답변을 생성한 뒤 (질문+답변) 쌍으로 student를 학습하는 방식.

| 항목 | Semi-on-policy | CorpusKDTrainer |
|-----|--------------|----------------|
| 코퍼스 적용 가능성 | 불가 (Q&A 형식 필요) | 가능 (raw text 그대로) |
| 금융 코퍼스 전처리 | 대규모 변환 필요 | 불필요 |
| 교사 모델 요구사항 | IT 모델 필수 | PT 모델도 가능 |

---

## 8. 교사 모델 선택 기준

### 8.1 이상적 교사: gemma-3-27b-pt (실제 학습)

- PT 모델: raw text의 자연스러운 언어 분포 P_T를 그대로 보유
- 대용량(27B): 12B 학생보다 더 정확한 분포 -> 실질적 지식 전달
- 코퍼스 도메인 정합: 동일 유형 텍스트에 대해 최적화된 분포

### 8.2 테스트용 교사: gemma-3-12b-it (파이프라인 검증 전용)

- IT 모델이므로 raw text 분포가 PT 모델보다 편향됨
- 학생과 동일 사이즈(12B)로 용량 이점 없음
- 파이프라인 검증 목적으로만 허용, 실제 학습에는 부적합

---

## 9. 메모리 효율 분석

ZeRO3 + CPU offload 구성에서의 메모리 분석:

| 항목 | 크기 (BF16) | 위치 |
|-----|-----------|-----|
| Student 12B 파라미터 | ~24 GB | GPU (ZeRO3 분산) |
| Student optimizer state | ~48 GB | CPU offload |
| Teacher 27B 파라미터 (grad 없음) | ~54 GB | GPU (ZeRO3 분산) |
| GPU 합계 (16 GPU 기준) | **~5 GB/GPU** | 적합 |

Teacher의 `requires_grad=False`가 optimizer state를 완전 제거하기 때문에 27B 대형 교사도 메모리 부담 없이 사용 가능하다.

---

## 10. 결론

`CorpusKDTrainer`가 기술적으로 타당한 이유를 세 가지로 요약한다.

**① 구조적 필연성**  
PT 모델 + 코퍼스 학습 환경에서는 `model.generate()`를 사용하는 on-policy 방식이 물리적으로 불가능하다(무한 반복). Off-policy forward-only 방식만이 유효하다.

**② 정보량의 우월성**  
JSD loss는 NLL이 버리는 vocab 전체의 soft distribution을 학습 신호로 활용한다. 이는 동일 학습 스텝에서 더 풍부한 의미 정보를 학생에게 전달한다.

**③ 기반 인프라의 보존**  
`SFTTrainer` 상속 + `compute_loss()` 단일 오버라이드로, packing·ZeRO3·Liger Kernel·PiSSA 등 검증된 모든 학습 최적화를 변경 없이 유지한다.

---

## 참고 문헌

- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531
- Agarwal, R., et al. (2024). On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. ICLR 2024 (GKD 원논문)
- TRL GKDTrainer 소스코드 — on-policy lmbda 파라미터 구현 참조
- Rajbhandari et al. (2020). ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.
