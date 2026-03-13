# KDKd — Corpus Pretrain & GKD (JSD 기반 Knowledge Distillation)

코퍼스 프리트레인 및 **GKD(Generalized Knowledge Distillation)** JSD 손실 기반 학습을 위한 프로젝트입니다.

---

## 가상환경 설치

### 1. 가상환경 만들기

Python 3.10 이상을 권장합니다. 프로젝트 루트에서:

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# Windows: venv\Scripts\activate
```

실행 스크립트(`sh/corpus/run-single.sh`)는 프로젝트 루트의 `venv` 를 자동으로 활성화합니다.

### 2. 패키지 설치

**방법 A — 요구 목록 파일이 있는 경우**

```bash
pip install -r requirements.txt
```

(`requirements.txt` 가 없으면 아래 방법 B로 핵심 패키지를 설치하면 됩니다.)

**방법 B — 핵심 패키지만 설치**

```bash
pip install torch transformers accelerate deepspeed datasets peft trl PyYAML wandb
pip install flash-attn --no-build-isolation   # Flash Attention 2 (CUDA 환경)
```

- **accelerate**: 분산 학습 래퍼
- **deepspeed**: ZeRO3 등 대규모 학습
- **transformers**, **trl**: 모델·SFTTrainer
- **peft**: LoRA 등 PEFT
- **datasets**: Arrow 데이터셋 로드
- **flash-attn**: CUDA 환경에서 권장(설정에서 `attn_implementation: flash_attention_2` 사용 시)

이미 설치된 환경의 전체 목록은 `kd_train_req.txt` 를 참고할 수 있습니다.

### 3. Accelerate 설정

DeepSpeed ZeRO3로 실행하려면 설정이 필요합니다. 이 저장소에 포함된 설정을 쓰면 됩니다.

```bash
# 단일 노드 8 GPU (run-single.sh 기본)
# config/zero3-single.yaml 사용
```

직접 `accelerate config` 로 할 경우, DeepSpeed를 선택하고 ZeRO Stage 3 등을 설정하면 됩니다.

---

## GKD(JSD)로 학습하려면

### 1. 사전 요구사항

- **환경**: 위처럼 가상환경 생성 후 필요한 패키지 설치
- **데이터**: 로컬 Arrow 데이터셋 디렉터리. 각 데이터셋은 `sentence_ls` 필드(문장 리스트)를 가진 형태여야 함.
- **교사 모델**: GKD 사용 시 **교사(teacher) 모델 경로**가 필요함 (HF 모델명 또는 로컬 체크포인트)

### 2. 설정 파일에서 GKD 활성화

`config/pretrain-gkd.yaml` 을 사용하거나, 기존 pretrain 설정을 복사한 뒤 아래를 추가/수정합니다.

```yaml
# ─── Loss / KD ─────────────────────────────────────────────────────────────
loss_type: "nll"   # use_gkd=true 이면 무시되고 JSD 사용

# ─── Knowledge Distillation ───────────────────────────────────────────────────
use_gkd: true

gkd:
  teacher_model_path: "/path/to/teacher-checkpoint"   # 또는 "HF_MODEL_NAME"
  beta: 0.5        # 0.0=forward KL, 0.5=대칭 JSD, 1.0=reverse KL
  temperature: 1.0  # 교사 softmax 온도
```

| 항목 | 설명 |
|------|------|
| `teacher_model_path` | 교사 모델 경로 (필수). HuggingFace 모델 ID 또는 로컬 디렉터리 |
| `beta` | JSD 계수. `0.5` = 대칭 JSD |
| `temperature` | 교사 로짓에 적용하는 softmax 온도 |

그 외 `model_name_or_path`, `output_dir`, `data_paths` 등은 일반 pretrain과 동일하게 설정합니다.

### 3. 실행

단일 노드(예: GPU 8장) 기준:

```bash
# GKD(JSD) 학습 — pretrain-gkd.yaml 사용
bash sh/corpus/run-single.sh pretrain-gkd.yaml
```

- **일반 NLL 프리트레인만** 할 때는 config 인자를 생략하면 `pretrain.yaml` 이 사용됩니다.

```bash
# 기본 NLL 프리트레인
bash sh/corpus/run-single.sh
```

실행 전에 `config/*.yaml` 의 다음 경로들이 실제 환경에 맞는지 확인하세요.

- `model_name_or_path` — 학생 모델
- `output_dir` — 체크포인트 저장 경로
- `data_paths` — 코퍼스 Arrow 데이터셋 디렉터리 목록
- `gkd.teacher_model_path` — GKD 사용 시 교사 모델 경로

### 4. 요약

| 목적 | Config | 명령 |
|------|--------|------|
| GKD(JSD) 코퍼스 학습 | `pretrain-gkd.yaml` | `bash sh/corpus/run-single.sh pretrain-gkd.yaml` |
| 일반 NLL 프리트레인 | `pretrain.yaml` | `bash sh/corpus/run-single.sh` |

로그는 `./logs/pretrain_single_<config_stem>_<timestamp>.log` 에 남으며, 실시간 확인은 `tail -f <로그파일>` 로 할 수 있습니다.
