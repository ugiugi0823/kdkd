# Plan 2: Corpus KD (지식 증류) 통합 계획

---

## 1. 왜 GKDTrainer를 쓰지 않는가

TRL의 `GKDTrainer`는 chat/SFT 스타일 KD 전용으로 설계되어 있다.
코퍼스 프리트레인에 적용하면 다음 문제가 발생한다.

| GKDTrainer 제약 | 코퍼스 프리트레인에서의 문제 |
|----------------|--------------------------|
| `packing=False` 강제 | 32768 토큰 윈도우에 문서 1개 → 대부분 패딩 낭비, 학습 속도 급감 |
| `messages` 형식 강제 | raw text를 억지로 감싸야 함 — 부자연스럽고 불필요한 전처리 |
| on-policy 모드 (`lmbda>0`) | 학생이 코퍼스 텍스트를 생성? → 프리트레인 목적과 무관 |

**결론: GKDTrainer 미사용. `SFTTrainer`를 상속해 `compute_loss()`만 오버라이드한다.**

---

## 2. 설계 방향

기존 `train_pretrain.py`의 모든 구조(packing, raw text, DeepSpeed ZeRO3)를 그대로 유지하고,  
`SFTTrainer`를 상속한 **`CorpusKDTrainer`** 클래스를 추가한다.

```
기존 SFTTrainer
    └── CorpusKDTrainer (상속)
            ├── __init__: teacher_model, beta, temperature 추가 저장
            └── compute_loss(): NLL → JSD(student, teacher) 대체
```

`use_gkd: false` → 기존 `SFTTrainer` 그대로  
`use_gkd: true`  → `CorpusKDTrainer` 사용 (나머지 모든 것 동일)

---

## 3. Loss 수식

### 기존 NLL Loss (SFTTrainer)

```
L = -sum( y_t * log P_S(y_t | x) )
```

### GKD JSD Loss (CorpusKDTrainer)

```
M_t = beta * P_S(t) + (1 - beta) * P_T(t)   # 혼합 분포

L_JSD = beta     * KL( P_S ‖ M )   # student → 혼합
      + (1-beta) * KL( P_T ‖ M )   # teacher → 혼합
```

- `beta=0.0` → Forward KL에 근접 (P_T 커버리지 보존)
- `beta=1.0` → Reverse KL에 근접 (P_S 모드 집중)
- `beta=0.5` → 대칭 JSD (균형)
- `temperature` → teacher softmax 온도 (높을수록 분포가 부드러워짐)

---

## 4. `CorpusKDTrainer` 클래스 설계

```python
import torch
import torch.nn.functional as F
from trl import SFTTrainer


class CorpusKDTrainer(SFTTrainer):
    def __init__(self, teacher_model, beta: float, temperature: float, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.beta = beta
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # ── Student forward ───────────────────────────────────────────────
        outputs = model(**inputs)
        student_logits = outputs.logits          # (B, T, V)

        labels = inputs.get("labels")

        # ── Teacher forward (gradient 없음) ───────────────────────────────
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits  # (B, T, V)

        # ── Temperature scaling ───────────────────────────────────────────
        if self.temperature != 1.0:
            teacher_logits = teacher_logits / self.temperature

        # ── JSD Loss 계산 (유효 토큰만) ───────────────────────────────────
        # labels == -100 인 위치(패딩/마스크)는 제외
        if labels is not None:
            mask = labels != -100                # (B, T)
        else:
            mask = torch.ones(
                student_logits.shape[:2],
                dtype=torch.bool,
                device=student_logits.device,
            )

        s_log_prob = F.log_softmax(student_logits, dim=-1)  # (B, T, V)
        t_prob     = F.softmax(teacher_logits,     dim=-1)  # (B, T, V)
        s_prob     = s_log_prob.exp()

        # 혼합 분포
        m_prob = self.beta * s_prob + (1 - self.beta) * t_prob
        m_log_prob = m_prob.clamp(min=1e-8).log()

        # KL(P_S ‖ M) + KL(P_T ‖ M)  → JSD
        kl_s = F.kl_div(m_log_prob, s_prob, reduction="none").sum(-1)   # (B, T)
        kl_t = F.kl_div(m_log_prob, t_prob, reduction="none").sum(-1)   # (B, T)

        jsd = self.beta * kl_s + (1 - self.beta) * kl_t                 # (B, T)

        # 유효 토큰 평균
        loss = jsd[mask].mean()

        return (loss, outputs) if return_outputs else loss
```

---

## 5. `pretrain.yaml` 추가 내용

기존 항목은 **전혀 변경하지 않는다.**  
아래 섹션만 추가한다.

```yaml
# ─── GKD (Knowledge Distillation) ────────────────────────────────────────────
use_gkd: false   # false: 기존 SFTTrainer / true: CorpusKDTrainer

gkd:
  teacher_model_path: "google/gemma-3-27b-pt"  # 교사 모델 경로
  beta: 0.5        # 0.0=forward KL, 1.0=reverse KL, 0.5=symmetric JSD
  temperature: 1.0 # 교사 softmax 온도
```

`packing`, `max_length`, `loss_type` 등 기존 설정은 그대로 유지된다.

---

## 6. `train_pretrain.py` 수정 내용

### 6-1. import 추가

```python
# 추가 (파일 상단)
from src.kd_trainer import CorpusKDTrainer   # 또는 같은 파일 내 클래스 정의
```

### 6-2. teacher 모델 로드 함수 추가

```python
def load_teacher_model(
    teacher_path: str,
    attn_impl: str,
) -> AutoModelForCausalLM:
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,        # DeepSpeed가 관리
        low_cpu_mem_usage=True,
    )
    teacher.config.use_cache = False
    for param in teacher.parameters():
        param.requires_grad = False   # teacher는 추론 전용
    teacher.eval()
    return teacher
```

### 6-3. main() 분기 (최소 변경)

```python
use_gkd = cfg.get("use_gkd", False)

if use_gkd:
    gkd_cfg = cfg["gkd"]
    teacher_model = load_teacher_model(
        gkd_cfg["teacher_model_path"],
        cfg.get("attn_implementation", "flash_attention_2"),
    )
    trainer = CorpusKDTrainer(
        teacher_model=teacher_model,
        beta=gkd_cfg.get("beta", 0.5),
        temperature=gkd_cfg.get("temperature", 1.0),
        # 아래는 기존 SFTTrainer 인자 그대로
        model=model,
        args=sft_cfg,           # SFTConfig 변경 없음
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        data_collator=None,
    )
else:
    trainer = SFTTrainer(       # 기존 코드 그대로
        model=model,
        args=sft_cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        data_collator=None,
    )
```

---

## 7. 메모리 예상

Teacher는 `requires_grad=False`이므로 optimizer state가 없다.

| 항목 | 메모리 (BF16) |
|------|-------------|
| Student 12B (파라미터) | ~24 GB |
| Student optimizer state (ZeRO3 CPU offload) | CPU |
| Teacher 27B (파라미터, grad 없음) | ~54 GB |
| **합계 GPU** | ~78 GB |
| ZeRO3 + CPU offload, 16 GPU | **GPU당 ~5 GB** ✅ |

메모리 절약 옵션: teacher를 동일 크기 IT 버전으로 대체

```yaml
gkd:
  teacher_model_path: "google/gemma-3-12b-it"  # 12B → GPU당 절반
```

---

## 8. 유지되는 것 / 바뀌는 것

| 항목 | 유지 | 변경 |
|------|------|------|
| packing=True | ✅ | |
| raw text 형식 | ✅ | |
| DeepSpeed ZeRO3 | ✅ | |
| Liger Kernel | ✅ | |
| PiSSA LoRA | ✅ | |
| SFTConfig | ✅ | |
| Loss 함수 | | NLL → JSD |
| Trainer 클래스 | | SFTTrainer → CorpusKDTrainer |
| 모델 수 | | 1개 → 2개 (student + teacher) |

---

## 9. 구현 순서

```
Step 1. src/kd_trainer.py 신규 작성
         └── CorpusKDTrainer 클래스 (SFTTrainer 상속, compute_loss 오버라이드)

Step 2. config/pretrain.yaml 수정
         └── use_gkd / gkd 섹션 추가

Step 3. src/train_pretrain.py 수정
         ├── load_teacher_model() 함수 추가
         └── main() 분기 (use_gkd true/false)

Step 4. config/pretrain-gkd.yaml 신규 작성
         └── use_gkd: true, teacher: gemma-3-12b-it (소규모 테스트용)
```
