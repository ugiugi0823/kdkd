# Plan 3: CorpusGOLDTrainer — 크로스-토크나이저 ULD 지식증류 계획

> 참고: [GOLD Trainer 공식 문서](https://huggingface.co/docs/trl/gold_trainer)

---

## 1. 왜 GOLD 방식이 필요한가

### 현재 CorpusKDTrainer (Plan 2)의 한계

| 제약 | 문제 |
|------|------|
| 동일 토크나이저 전제 | Teacher·Student가 같은 vocab을 공유해야 JSD 계산 가능 |
| 동일 모델 패밀리 전제 | Gemma ↔ Gemma는 가능, LLaMA ↔ Gemma는 불가 |
| JSD는 vocab 일치 필요 | 토큰 위치별 1:1 비교 → 어휘 불일치 시 의미 없음 |

### GOLD가 해결하는 것

```
현재:  Gemma-12B (student) ← Gemma-27B   (teacher)   ← 동일 vocab, JSD 가능
GOLD:  gemma-3-4b-pt       ← Qwen3.5-9B-Base (teacher)   ← 다른 vocab, ULD 필요
```

**핵심 아이디어 (ULD):** 토큰 ID가 아닌 **확률 순위(rank)** 를 기준으로 비교
→ 어휘가 달라도 "가장 확신하는 top-K 토큰" 분포를 정렬 기반 L1 거리로 비교

### 대상 모델 (이번 실험)

| 역할 | 모델 | 토크나이저 | Vocab 크기 |
|------|------|-----------|-----------|
| **Student** | `google/gemma-3-4b-pt` | Gemma3 tokenizer | 256,000 |
| **Teacher** | `Qwen/Qwen3.5-9B-Base` | Qwen3 tokenizer | 151,936 |

- vocab 크기 차이: **+104,064** (Gemma3 > Qwen3)
- 공통 어휘 거의 없음 → **순수 크로스-토크나이저** 상황
- ULD가 필수, Hybrid Loss는 불필요 (matched vocab이 사실상 0)

---

## 2. GOLD vs CorpusKDTrainer 비교

| 항목 | CorpusKDTrainer (Plan 2) | CorpusGOLDTrainer (Plan 3) |
|------|--------------------------|---------------------------|
| Loss | JSD (vocab 일치 전제) | ULD (vocab 무관) or Hybrid |
| 토크나이저 | Student = Teacher | Student ≠ Teacher 허용 |
| 크로스-토크나이저 | ✗ 불가 | ✅ 가능 (텍스트 span 정렬) |
| packing | ✅ 유지 | ✅ 유지 |
| raw text | ✅ 유지 | ✅ 유지 |
| DeepSpeed ZeRO3 | ✅ 유지 | ✅ 유지 |
| on-policy 샘플링 | ✗ 미사용 (프리트레인 무관) | ✗ 미사용 |
| Hybrid Loss | ✗ | ✅ 선택적 (matched/unmatched) |

---

## 3. Loss 수식

### 3-1. ULD Loss (기본, 크로스-토크나이저)

토큰 어휘가 달라도 적용 가능한 **정렬 기반 L1 거리**.

```
P_S_sorted = sort(softmax(logits_S), descending=True)   # (B, T, V_S)
P_T_sorted = sort(softmax(logits_T), descending=True)   # (B, T, V_T)

# V_S ≠ V_T 일 수 있으므로 min(V_S, V_T)까지만 비교
K = min(V_S, V_T)

L_ULD = mean( |P_S_sorted[:K] - P_T_sorted[:K]| )
```

- 토큰 ID를 보지 않고 **분포의 형태(shape)** 만 비교
- 교사가 어떤 언어모델이든 적용 가능

### 3-2. Hybrid ULD Loss (선택, 부분 vocab 중복 시)

동일 계열 모델처럼 vocab이 **일부 겹칠 때** 더 정확한 신호 제공.

```
# Matched 토큰 집합: 동일한 token_id가 양쪽에 존재하는 경우
L_matched   = CE( P_S[matched_ids], P_T[matched_ids] )   # 정확한 비교

# Unmatched 토큰 집합: 한쪽에만 있는 경우
P_S_sorted_unmatched = sort(P_S[unmatched_ids], descending=True)
P_T_sorted_unmatched = sort(P_T[unmatched_ids], descending=True)
L_unmatched = L1( P_S_sorted_unmatched, P_T_sorted_unmatched )   # ULD

L_hybrid = w_matched * L_matched + w_unmatched * L_unmatched
```

파라미터:
- `uld_hybrid_matched_weight` (기본 1.0)
- `uld_hybrid_unmatched_weight` (기본 1.0)

### 3-3. ULD + Cross-Entropy 혼합 (선택)

```
L_total = w_distill * L_ULD + w_ce * L_CE(student, labels)
```

- `uld_crossentropy_weight`: CE 비중 (0.0이면 pure ULD)
- `uld_distillation_weight`: ULD 비중 (기본 1.0)

---

## 4. 크로스-토크나이저 정렬 (Cross-Tokenizer Alignment)

Student와 Teacher의 토크나이저가 다를 경우 **같은 텍스트 구간을 기준으로 logit을 정렬**한다.

### 4-1. 문제 정의

Gemma3와 Qwen3는 완전히 독립적으로 학습된 토크나이저를 사용한다.

```
텍스트: "금융 시장"
Student (gemma-3-4b-pt, vocab=256k):  ["▁금융", "▁시장"]       → 2 tokens
Teacher (Qwen3.5-9B-Base, vocab=152k): ["金", "融", "▁市", "场"] → 4 tokens
```

같은 한국어 텍스트가 완전히 다른 방식으로 분절된다.

Student 1개 토큰 위치에 Teacher 여러 토큰의 확률을 **합산·병합**해야 한다.

### 4-2. 확률 병합 방식 (GOLD 방식)

Student 토큰 하나가 Teacher 토큰 k개에 대응할 때:

```
P_merged(y) = P_T(token_0 | ctx)
            × P_T(token_1 | token_0, ctx)
            × ...
            × P_T(token_{k-1} | ..., ctx)
```

- `P_T(token_0 | ctx)`: 첫 위치의 전체 vocabulary 분포 (벡터)
- 이후 조건부 확률은 **실제 생성된 토큰에 대한 스칼라값**만 추출
- 전체 분포에 스칼라를 곱해 합산 → unnormalized이지만 ULD에서는 OK

### 4-3. 동일 토크나이저일 경우

정렬 필요 없음. Teacher logit을 그대로 ULD loss에 입력.

---

## 5. CorpusGOLDTrainer 클래스 설계

```
SFTTrainer
    └── CorpusKDTrainer (Plan 2: JSD Loss)
    └── CorpusGOLDTrainer (Plan 3: ULD Loss + 크로스-토크나이저)
            ├── __init__: teacher_tokenizer 추가 저장
            ├── compute_loss(): ULD or Hybrid ULD loss
            ├── _merge_teacher_logits(): 크로스-토크나이저 정렬·병합
            └── _uld_loss(): 정렬 기반 L1 distance
```

### 5-1. `__init__` 추가 인자

```python
class CorpusGOLDTrainer(SFTTrainer):
    def __init__(
        self,
        teacher_model,
        teacher_tokenizer,          # 크로스-토크나이저 시 필수
        student_tokenizer,          # 텍스트 복원용
        # ULD 파라미터
        use_uld_loss: bool = True,
        uld_crossentropy_weight: float = 0.0,
        uld_distillation_weight: float = 1.0,
        uld_student_temperature: float = 1.0,
        uld_teacher_temperature: float = 1.0,
        # Hybrid 파라미터
        uld_use_hybrid_loss: bool = False,
        uld_hybrid_matched_weight: float = 1.0,
        uld_hybrid_unmatched_weight: float = 1.0,
        # 기타
        uld_skip_student_eos: bool = True,
        uld_skip_teacher_eos: bool = True,
        **kwargs,
    )
```

### 5-2. `compute_loss()` 흐름

```
1. Student forward → student_logits (B, T_S, V_S)
2. Teacher forward (no_grad) → teacher_logits (B, T_T, V_T)

3. 동일 토크나이저?
   ├── YES: 정렬 불필요, teacher_logits 그대로 사용
   └── NO:  _merge_teacher_logits() → 병합된 teacher_logits (B, T_S, V_T)

4. uld_use_hybrid_loss?
   ├── YES: _hybrid_uld_loss() → matched CE + unmatched ULD
   └── NO:  _uld_loss() → sorted L1

5. uld_crossentropy_weight > 0?
   └── YES: loss += w_ce * CE(student, labels)

6. return loss
```

### 5-3. `_uld_loss()` 구현 스케치

```python
def _uld_loss(self, student_logits, teacher_logits, mask):
    # Temperature scaling
    s_logits = student_logits / self.uld_student_temperature
    t_logits = teacher_logits / self.uld_teacher_temperature

    # Softmax → 확률 분포
    s_prob = F.softmax(s_logits, dim=-1)   # (B, T, V_S)
    t_prob = F.softmax(t_logits, dim=-1)   # (B, T, V_T)

    # 정렬 (내림차순)
    s_sorted, _ = s_prob.sort(dim=-1, descending=True)
    t_sorted, _ = t_prob.sort(dim=-1, descending=True)

    # vocab 크기 맞춤
    K = min(s_sorted.shape[-1], t_sorted.shape[-1])
    s_sorted = s_sorted[..., :K]
    t_sorted = t_sorted[..., :K]

    # L1 distance per token
    l1_per_token = (s_sorted - t_sorted).abs().sum(dim=-1)   # (B, T)

    # 유효 토큰 평균
    loss = (l1_per_token * mask).sum() / mask.sum().clamp(min=1)
    return loss
```

---

## 6. YAML 설정 — 추가 항목

기존 `use_gkd` 섹션과 **별도로** `use_gold` 섹션을 추가한다.  
`use_gkd`와 `use_gold`는 상호 배타적으로 동작한다.

**`config/pretrain-gold.yaml` (실전)**

```yaml
# ─── 모델 ──────────────────────────────────────────────────────────────────────
model_name_or_path: "google/gemma-3-4b-pt"
output_dir: "/PROJECT/0325120095_A/BASE/rex/LLM/output/gemma-3-4b-pt/gold"

# ─── GOLD Knowledge Distillation ──────────────────────────────────────────────
use_gold: true

gold:
  teacher_model_path: "Qwen/Qwen3.5-9B-Base"
  teacher_tokenizer_path: "Qwen/Qwen3.5-9B-Base"   # Gemma3와 완전히 다른 vocab

  # ULD Loss (Gemma3 ↔ Qwen3: vocab 불일치 → Hybrid 불필요)
  use_uld_loss: true
  uld_crossentropy_weight: 0.0      # pure ULD
  uld_distillation_weight: 1.0
  uld_student_temperature: 1.0
  uld_teacher_temperature: 1.0
  uld_skip_student_eos: true
  uld_skip_teacher_eos: true

  # Hybrid Loss: Gemma3↔Qwen3는 vocab 교집합 거의 없으므로 false
  uld_use_hybrid_loss: false
  uld_hybrid_matched_weight: 1.0
  uld_hybrid_unmatched_weight: 1.0
```

**`config/pretrain-gold-test.yaml` (테스트, 소규모 검증)**

```yaml
model_name_or_path: "google/gemma-3-4b-pt"
output_dir: "/PROJECT/0325120095_A/BASE/rex/LLM/output/gemma-3-4b-pt/gold-test"

use_gold: true

gold:
  teacher_model_path: "Qwen/Qwen3.5-9B-Base"
  teacher_tokenizer_path: "Qwen/Qwen3.5-9B-Base"
  use_uld_loss: true
  uld_crossentropy_weight: 0.0
  uld_distillation_weight: 1.0
  uld_student_temperature: 1.0
  uld_teacher_temperature: 1.0
  uld_skip_student_eos: true
  uld_skip_teacher_eos: true
  uld_use_hybrid_loss: false
```

### 이번 실험 설정

| 항목 | 값 |
|------|----|
| Student | `google/gemma-3-4b-pt` (vocab 256k) |
| Teacher | `Qwen/Qwen3.5-9B-Base` (vocab 152k) |
| Loss | Pure ULD (sorted L1) |
| Hybrid | false (vocab 교집합 거의 없음) |
| CE 혼합 | false (pure distillation) |

---

## 7. `train_pretrain.py` 수정 내용

### 7-1. 분기 로직 추가 (기존 `use_gkd` 분기 뒤에 추가)

```python
use_gold = cfg.get("use_gold", False)

if use_gold:
    gold_cfg = cfg["gold"]
    teacher_model = load_teacher_model(
        gold_cfg["teacher_model_path"],
        cfg.get("attn_implementation", "flash_attention_2"),
    )
    # 크로스-토크나이저: teacher_tokenizer_path가 student와 다를 수 있음
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        gold_cfg.get("teacher_tokenizer_path", gold_cfg["teacher_model_path"])
    )
    trainer = CorpusGOLDTrainer(
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=tokenizer,
        use_uld_loss=gold_cfg.get("use_uld_loss", True),
        uld_crossentropy_weight=gold_cfg.get("uld_crossentropy_weight", 0.0),
        uld_distillation_weight=gold_cfg.get("uld_distillation_weight", 1.0),
        uld_student_temperature=gold_cfg.get("uld_student_temperature", 1.0),
        uld_teacher_temperature=gold_cfg.get("uld_teacher_temperature", 1.0),
        uld_use_hybrid_loss=gold_cfg.get("uld_use_hybrid_loss", False),
        uld_hybrid_matched_weight=gold_cfg.get("uld_hybrid_matched_weight", 1.0),
        uld_hybrid_unmatched_weight=gold_cfg.get("uld_hybrid_unmatched_weight", 1.0),
        uld_skip_student_eos=gold_cfg.get("uld_skip_student_eos", True),
        uld_skip_teacher_eos=gold_cfg.get("uld_skip_teacher_eos", True),
        # 기존 SFTTrainer 인자
        model=model,
        args=sft_cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        data_collator=None,
    )
```

### 7-2. 우선순위

```
use_gold=true  → CorpusGOLDTrainer (Plan 3)
use_gkd=true   → CorpusKDTrainer   (Plan 2)
둘 다 false    → SFTTrainer        (기존)
```

---

## 8. 새 파일 구성

```
src/
├── kd_trainer.py       ← 기존 CorpusKDTrainer (Plan 2, 수정 없음)
└── gold_trainer.py     ← 신규 CorpusGOLDTrainer (Plan 3)
        ├── CorpusGOLDTrainer 클래스
        ├── _merge_teacher_logits()    ← 크로스-토크나이저 정렬·병합
        ├── _uld_loss()                ← ULD (sorted L1)
        └── _hybrid_uld_loss()         ← Hybrid ULD (선택적)

config/
├── pretrain-gold.yaml        ← 신규: 실전용
│     student: google/gemma-3-4b-pt
│     teacher: Qwen/Qwen3.5-9B-Base
└── pretrain-gold-test.yaml   ← 신규: 테스트용 (소규모 검증)
      student: google/gemma-3-4b-pt
      teacher: Qwen/Qwen3.5-9B-Base
```

---

## 9. 메모리 예상

### 이번 실험 조합: gemma-3-4b-pt (Student) + Qwen3.5-9B-Base (Teacher)

| 항목 | 파라미터 | 메모리 (BF16) |
|------|---------|-------------|
| Student `gemma-3-4b-pt` | ~4B | ~8 GB |
| Teacher `Qwen3.5-9B-Base` (grad 없음) | ~9B | ~18 GB |
| **합계 GPU** | | **~26 GB** |
| ZeRO3 + CPU offload, 4 GPU | | **GPU당 ~6.5 GB** ✅ |
| ZeRO3 + CPU offload, 2 GPU | | **GPU당 ~13 GB** ✅ |

- Teacher는 `requires_grad=False` → optimizer state 없음
- ZeRO3 덕분에 **GPU 2장**으로도 충분히 실험 가능
- Plan 2 (12B student + 27B teacher = ~78 GB) 대비 **메모리 3배 절약**

### Vocab 크기별 ULD 연산 비용

| 항목 | 값 |
|------|----|
| Student vocab (Gemma3) | 256,000 |
| Teacher vocab (Qwen3) | 151,936 |
| ULD 비교 크기 K | min(256k, 152k) = **151,936** |
| 정렬 연산 (`sort`) | 토큰당 1회, O(V log V) |

> 정렬 비용이 크므로 `top-K ULD` (상위 K개만 정렬·비교) 옵션도 추가 고려 가능:  
> `uld_top_k: 10000` → 상위 1만 개만 비교하면 연산량 93% 절감

---

## 10. Plan 2와의 차이점 요약

| 구분 | Plan 2 (CorpusKDTrainer) | Plan 3 (CorpusGOLDTrainer) |
|------|--------------------------|---------------------------|
| **Loss** | JSD (vocab 매칭 기반) | ULD (정렬 기반 L1) |
| **토크나이저** | Student = Teacher 필수 | Student ≠ Teacher 허용 |
| **vocab 크기** | 동일해야 함 | 달라도 됨 |
| **Hybrid** | 없음 | matched CE + unmatched ULD |
| **CE 혼합** | 없음 | uld_crossentropy_weight로 제어 |
| **추가 파일** | kd_trainer.py | gold_trainer.py |
| **기존 호환성** | 유지 | 유지 (use_gkd 분기 그대로) |

---

## 11. 구현 순서

```
Step 1. src/gold_trainer.py 신규 작성
         ├── CorpusGOLDTrainer 클래스 (SFTTrainer 상속)
         ├── _is_same_tokenizer() 판별 함수
         │     → vocab_size 비교 + tokenizer class 비교
         ├── _merge_teacher_logits()
         │     → Gemma3 토큰 시퀀스 ↔ Qwen3 토큰 시퀀스 텍스트 span 정렬
         │     → Qwen3 다수 토큰 → Gemma3 단일 토큰 위치로 확률 병합
         ├── _uld_loss()
         │     → sort(P_S)[:K], sort(P_T)[:K], L1 distance
         │     → (이번 실험) K = min(256000, 151936) = 151936
         └── _hybrid_uld_loss() ← 이번 실험에서 미사용, 구현만

Step 2. src/train_pretrain.py 수정
         ├── from src.gold_trainer import CorpusGOLDTrainer 추가
         ├── use_gold 분기 추가 (use_gkd 분기 뒤)
         └── teacher_tokenizer 로드: AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B-Base")

Step 3. config/pretrain-gold-test.yaml 신규 작성
         ├── model_name_or_path: "google/gemma-3-4b-pt"
         ├── use_gold: true
         └── gold.teacher_model_path: "Qwen/Qwen3.5-9B-Base"

Step 4. config/pretrain-gold.yaml 신규 작성
         ├── model_name_or_path: "google/gemma-3-4b-pt"
         ├── use_gold: true
         └── gold.teacher_model_path: "Qwen/Qwen3.5-9B-Base"

Step 5. 크로스-토크나이저 정렬 단위 테스트
         └── 한국어 샘플 문장으로 Gemma3↔Qwen3 span 정렬 결과 육안 확인

Step 6. 소규모 학습 검증 (pretrain-gold-test.yaml)
         └── loss가 수렴하는지, NaN 없는지 확인
```
