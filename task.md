# GOLD Trainer 구현 태스크

## 목표
Cross-tokenizer Knowledge Distillation (KD) 학습 성공
- **Student**: `google/gemma-3-4b-pt` (Gemma3 tokenizer, vocab 262,208)
- **Teacher**: `Qwen/Qwen3.5-9B-Base` (Qwen3 tokenizer, vocab 151,936)
- **방법**: GOLD (General Online Logit Distillation) + ULD (Universal Logit Distillation) loss
- **환경**: DeepSpeed ZeRO3, 4 GPU (0,1,2,7), packing 지원

---

## 구현 파일

| 파일 | 설명 |
|---|---|
| `src/gold_trainer.py` | `CorpusGOLDTrainer` 구현 (576 lines) |
| `src/train_pretrain.py` | 학습 진입점. `use_gold: true`이면 GOLD 분기 |
| `config/pretrain-gold.yaml` | 전체 학습 설정 |
| `config/pretrain-gold-test.yaml` | 테스트 설정 (top_k=10,000, LoRA) |
| `plan3.md` | 구현 계획서 |

---

## 핵심 구조 (`gold_trainer.py`)

```
CorpusGOLDTrainer (SFTTrainer 상속)
├── _teacher_forward_same_tok()    # 동일 tokenizer teacher forward
├── _teacher_forward_cross_tok()   # 크로스 tokenizer forward + 정렬
├── _align_teacher_logits()        # character-level span 정렬
├── _uld_loss()                    # ULD loss 계산
└── compute_loss()                 # student forward → teacher forward → loss
```

---

## 실행 이력 & 에러 해결

### Run 1 — `gold-test-20260305_172146.log`
- **에러**: `RuntimeError: 'weight' must be 2-D`
- **원인**: DeepSpeed ZeRO3가 teacher 파라미터를 1D shard로 분할. `nn.Embedding` 호출 시 2D weight 필요
- **조치**: `gold_trainer.py`에 `deepspeed.zero.GatheredParameters` 적용

### Run 2 — `gold-test-20260305_174222.log`
- **에러**: `CUDA error: an illegal memory access` (in `torch_chunk_gated_delta_rule`)
- **원인**: Qwen3.5의 linear attention fallback 커널(`torch_chunk_gated_delta_rule`)이 gathered parameter와 충돌. `flash-linear-attention` 미설치로 인한 PyTorch fallback 사용
- **조치**: `causal-conv1d` + `flash-linear-attention` 설치 (`CUDA_HOME=/usr/local/cuda-12.8`, `TMPDIR=/PROJECT/.../tmp_build`)

### Run 3 — `gold-test-20260305_175400.log`
- **에러**: `RuntimeError: 'weight' must be 2-D` (재발)
- **원인**: `GatheredParameters` 적용 범위 미흡
- **조치**: `_teacher_forward_same_tok` / `_teacher_forward_cross_tok` 양쪽 모두 `GatheredParameters` 적용

### Run 4 — `gold-test-20260305_180330.log`
- **에러**: `CUDA error: an illegal memory access` (in `torch_chunk_gated_delta_rule`)
- **원인**: `causal_conv1d_fn` CUDA 커널의 async 에러가 다음 동기화 지점에서 보고됨
- **조치**:
  - `GatheredParameters` 컨텍스트 안에서 `torch.cuda.synchronize()` 추가
  - logits를 `.cpu()`로 복사 후 context 종료 → use-after-free 방지

### Run 5 — `gold-test-20260305_183451.log`
- **에러**: `CUDA error: an illegal memory access` (in `flash_attention_forward`)
- **원인**: `eager` attention 없이 flash_attention_2 + ZeRO3 GatheredParams → async CUDA 에러
- **조치**: `train_pretrain.py` teacher → `attn_implementation="eager"`

### Run 6 — `gold-test-20260305_223207.log`
- **에러**: `--config` 플래그 누락 (명령어 포맷 오류)

### Run 7 — `gold-test-20260305_233117.log`
- **에러**: `DeepSpeed CUDAMismatchException: 12.8 ≠ torch 13.0`
- **원인**: `CUDA_HOME=/usr/local/cuda-12.8` 설정 → DeepSpeed JIT 컴파일 불일치
- **조치**: 런타임에서 `CUDA_HOME` 제거 (causal-conv1d는 이미 설치됨, 재빌드 불필요)

### Run 8 — `gold-test-20260305_233807.log`
- **에러**: `CUDA out of memory — tried to allocate 104.67 GiB` (in `eager_attention_forward`)
- **원인**: `eager` attention은 O(T²) attention matrix → 32K 토큰 시 OOM
- **조치**: teacher → `attn_implementation="sdpa"` (PyTorch fused, FlashAttention 백엔드)

### Run 9 — `gold-test-20260305_234415.log`
- **에러**: `SyntaxError: unmatched ')'` (`train_pretrain.py` line 552)
- **원인**: 이전 편집 실패로 파일 끝에 잔여 코드 삽입
- **조치**: 중복 라인 제거

### Run 10 — `gold-test-20260305_234858.log`
- **에러**: `CUDA out of memory — tried to allocate 32.01 GiB` (in `_uld_loss` → `F.softmax`)
- **원인**: `softmax(student_logits)` = (B=2, T=32768, V=262208) × bf16 = **34 GiB**
- **조치**:
  - `gold_trainer.py` `_uld_loss`: full vocab softmax 제거 → `topk(K, logits)` 먼저 추출 후 softmax (메모리 O(B·T·K))
  - `pretrain-gold-test.yaml`: `max_length: 32768 → 4096`, `per_device_train_batch_size: 2 → 1`

### Run 11 — `gold-test-20260305_235522.log` ✅ **성공**
- **loss 출력 확인**: `loss: 1.17~1.22`, `grad_norm: ~2.8` — 정상 수렴
- **처리 속도**: ~3,500 tok/s (4 GPU)

---

## 현재 상태 (`2026-03-05 기준`) — ✅ 학습 성공

### 적용된 최신 패치 (누적)
1. **`gold_trainer.py`**: labels 제거 후 student forward → Liger Kernel logits=None 문제 해결 ✅
2. **`gold_trainer.py`**: `GatheredParameters` 양쪽 teacher forward 적용 ✅
3. **`gold_trainer.py`**: gathered params `.contiguous()` 강제 + `torch.cuda.synchronize()` ✅
4. **`gold_trainer.py`**: logits `.cpu()` 복사로 GatheredParams 해제 후 use-after-free 방지 ✅
5. **`gold_trainer.py`** `_uld_loss`: top-K logits 추출 후 softmax → 메모리 O(B·T·K) ✅
6. **`train_pretrain.py`**: teacher → `attn_implementation="sdpa"` ✅
7. **`config/pretrain-gold-test.yaml`**: `max_length=4096`, `batch_size=1` (테스트용) ✅

### 남은 과제
- [ ] Run 11 전체 epoch 완료 확인
- [ ] `pretrain-gold.yaml` (본 학습) 설정 검토 — max_length/batch_size 프로덕션 값으로 조정

---

## 재실행 명령어

```bash
cd /PROJECT/0325120095_A/BASE/rex/LLM
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG=logs/gold-test-${TIMESTAMP}.log
CUDA_HOME=/usr/local/cuda-12.8 \
TMPDIR=/PROJECT/0325120095_A/BASE/rex/LLM/tmp_build \
CUDA_VISIBLE_DEVICES=0,1,2,7 \
venv/bin/accelerate launch \
  --config_file config/zero3-4gpu.yaml \
  src/train_pretrain.py \
  --config config/pretrain-gold-test.yaml \
  > $LOG 2>&1 &
```

---

## 핵심 기술 이슈 요약

### ZeRO3 + Qwen3.5 Teacher 추론 문제
- ZeRO3는 모든 파라미터를 1D shard로 분할 → 직접 forward 불가
- `GatheredParameters(modifier_rank=None)` 로 임시 full 파라미터 복원
- Qwen3.5는 **하이브리드 아키텍처** (linear attention + full attention):
  - Linear attention: `causal_conv1d_fn` 커널 (별도 라이브러리 필요)
  - Full attention: `flash_attention_2` 커널
  - 두 커널 모두 gathered parameter의 메모리 레이아웃에 민감

### 설치된 추가 라이브러리
```bash
# CUDA 12.8 환경에서 빌드
CUDA_HOME=/usr/local/cuda-12.8 TMPDIR=.../tmp_build \
pip install causal-conv1d flash-linear-attention
```

### Liger Kernel 이슈
- Student(Gemma3)에 Liger Kernel 적용 시 `labels` 전달하면 `outputs.logits = None`
- **해결**: `compute_loss`에서 `labels` 제거 후 student forward → logits 직접 취득

---

## 참고 링크
- [TRL GOLD Trainer 문서](https://huggingface.co/docs/trl/gold_trainer)
- [Plan3 설계 문서](plan3.md)
