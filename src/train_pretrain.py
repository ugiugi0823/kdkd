#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corpus Continual Pretrain — Accelerate + DeepSpeed ZeRO3

실행 방법:
    # 서버1 (Node 0)
    accelerate launch --config_file config/zero3-node0.yaml \
        src/train_pretrain.py --config config/pretrain.yaml

    # 서버2 (Node 1) — 서버1과 동시 실행
    accelerate launch --config_file config/zero3-node1.yaml \
        src/train_pretrain.py --config config/pretrain.yaml

주요 특징:
- DeepSpeed ZeRO3 (파라미터 / 옵티마이저 상태 CPU 오프로드)
- PiSSA LoRA 초기화 (pissa_niter_4)
- Liger Kernel (Gemma3 최적화)
- Flash Attention 2
- Packing (여러 문서를 max_length 윈도우로 패킹)
- 데이터: /xtmp/ 로컬 Arrow 포맷 (sentence_ls 필드 → join → text)
"""

import os
import argparse
import warnings
from datetime import datetime
from typing import Any

import yaml
import torch
import wandb
from datasets import load_from_disk, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import Gemma3ForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType
from kd_trainer import CorpusKDTrainer
from gold_trainer import CorpusGOLDTrainer

warnings.filterwarnings("ignore")


# ─── Config ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config 파일을 찾을 수 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_env(env_dict: dict[str, Any]) -> None:
    for key, val in env_dict.items():
        os.environ[key] = str(val)


# ─── Gemma3 텍스트 전용 로더 ─────────────────────────────────────────────────

class _Gemma3CausalLMFromVLM(Gemma3ForCausalLM):
    """VLM 체크포인트(language_model.* 키)에서 Gemma3ForCausalLM을 로드하는 전용 클래스.

    Gemma3 4B/12B/27B PT 체크포인트는 VLM 형식(language_model.* prefix)이다.
    base_model_prefix = "language_model" 로 설정하면 transformers의 prefix 제거
    로직이 자동으로 처리한다. 파일 변환 없이 직접 로드 가능.
    비전 타워 키(vision_tower.*, multi_modal_projector.*)는 UNEXPECTED로 무시된다.
    """
    base_model_prefix = "language_model"


def _load_gemma3_causal_lm(
    model_path: str,
    attn_impl: str,
) -> Gemma3ForCausalLM:
    """VLM 체크포인트를 Gemma3ForCausalLM으로 직접 로드 (파일 변환 없음)."""
    vlm_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_cfg = vlm_cfg.text_config   # Gemma3TextConfig

    return _Gemma3CausalLMFromVLM.from_pretrained(
        model_path,
        config=text_cfg,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,
        low_cpu_mem_usage=True,
    )


# ─── Teacher Model ───────────────────────────────────────────────────────────

def _is_gemma3_checkpoint(model_path: str) -> bool:
    """모델 경로의 config를 확인해 Gemma3 VLM 체크포인트인지 판별."""
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return (
            type(cfg).__name__ in ("Gemma3Config",)
            and hasattr(cfg, "text_config")
        )
    except Exception:
        return False


def load_teacher_model(
    teacher_path: str,
    attn_impl: str,
) -> torch.nn.Module:
    """
    교사 모델 로드. gradient 비활성화 후 eval 모드로 고정.

    Gemma3 VLM 체크포인트(google/gemma-3-*-pt 등): _load_gemma3_causal_lm 사용
    그 외 모델(Qwen, LLaMA 등): AutoModelForCausalLM 범용 로더 사용

    ZeRO3 CPU 오프로드 환경에서 티처 모델을 CPU에 두면 8 프로세스 × 모델 크기만큼
    시스템 RAM을 소비해 OOM이 발생한다. 각 rank의 전용 CUDA 장치에 로드하면
    VRAM을 사용하므로 시스템 RAM 압박 없이 안정적으로 동작한다.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"

    if _is_gemma3_checkpoint(teacher_path):
        teacher = _load_gemma3_causal_lm(teacher_path, attn_impl)
    else:
        # Qwen3.5 같은 하이브리드 모델(linear + full attention):
        # - eager: O(T²) attention matrix → 32K 토큰 OOM
        # - flash_attention_2: ZeRO3 GatheredParams + async CUDA 에러
        # - sdpa: PyTorch 내장 fused SDPA (FlashAttention 백엔드) → 메모리 효율적
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_path,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=None,
            low_cpu_mem_usage=True,
        )

    teacher.config.use_cache = False
    for param in teacher.parameters():
        param.requires_grad = False   # 추론 전용, optimizer state 없음
    teacher.eval()
    teacher = teacher.to(device)
    return teacher


# ─── Data ────────────────────────────────────────────────────────────────────

def extract_text(example: dict[str, Any], text_field: str, sep: str) -> dict[str, str]:
    """sentence_ls(List[str]) → 하나의 text 문자열로 병합"""
    raw = example.get(text_field)
    if isinstance(raw, list):
        text = sep.join(s for s in raw if isinstance(s, str) and s.strip())
    elif isinstance(raw, str):
        text = raw
    else:
        text = ""
    return {"text": text}


def load_datasets(
    data_paths: list[str],
    text_field: str,
    text_join_sep: str,
    validation_truncate: int,
    preprocessing_num_workers: int,
    preprocessing_batch_size: int,
    is_main_process: bool,
) -> tuple[Any, Any]:
    """로컬 Arrow 데이터셋 로드 → 텍스트 추출 → 병합"""

    train_datasets, eval_datasets = [], []

    for path in data_paths:
        ds_name = os.path.basename(path)
        if is_main_process:
            print(f"  로드 중: {path}")

        ds_dict: DatasetDict = load_from_disk(path)

        for split_name in ["train", "validation"]:
            if split_name not in ds_dict:
                continue

            split_ds = ds_dict[split_name]

            # text 추출
            split_ds = split_ds.map(
                extract_text,
                fn_kwargs={"text_field": text_field, "sep": text_join_sep},
                num_proc=preprocessing_num_workers,
                batch_size=preprocessing_batch_size,
                remove_columns=split_ds.column_names,
                desc=f"{ds_name}/{split_name} 전처리",
            )

            # 빈 text 제거
            split_ds = split_ds.filter(
                lambda x: len(x["text"].strip()) > 0,
                num_proc=preprocessing_num_workers,
            )

            if split_name == "train":
                train_datasets.append(split_ds)
            else:
                # validation은 각 데이터셋별 500개로 제한
                split_ds = split_ds.select(range(min(validation_truncate, len(split_ds))))
                eval_datasets.append(split_ds)

            if is_main_process:
                print(f"    {split_name}: {len(split_ds):,} 건")

    train_dataset = concatenate_datasets(train_datasets).shuffle(seed=42)
    eval_dataset = concatenate_datasets(eval_datasets) if eval_datasets else None

    return train_dataset, eval_dataset


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Corpus Continual Pretrain")
    parser.add_argument("--config", type=str, required=True, help="YAML 설정 파일 경로")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 환경 변수 설정
    if "env" in cfg:
        set_env(cfg["env"])

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank <= 0

    # 분산 환경에서 timestamp 동기화
    master_port = os.environ.get("MASTER_PORT", "29500")
    timestamp_file = f"/tmp/pretrain_ts_{master_port}.txt"

    if world_size > 1:
        import time
        if is_main:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(timestamp_file, "w") as f:
                f.write(ts)
        else:
            waited = 0
            while not os.path.exists(timestamp_file) and waited < 30:
                time.sleep(0.1)
                waited += 0.1
            ts = open(timestamp_file).read().strip() if os.path.exists(timestamp_file) else datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if is_main:
        print("=" * 60)
        print(f"  Corpus Pretrain: {cfg['model_name_or_path']}")
        print(f"  timestamp : {ts}")
        print(f"  world_size: {world_size}")
        print("=" * 60)

    # ── 1. 토크나이저 ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name_or_path"],
        trust_remote_code=True,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if is_main:
        print(f"✅ Tokenizer: {type(tokenizer).__name__}  pad_token_id={tokenizer.pad_token_id}")

    # ── 2. 데이터 로드 ────────────────────────────────────────────────────────
    if is_main:
        print("\n데이터셋 로드 중...")

    train_dataset, eval_dataset = load_datasets(
        data_paths=cfg["data_paths"],
        text_field=cfg.get("text_field", "sentence_ls"),
        text_join_sep=cfg.get("text_join_sep", "\n"),
        validation_truncate=cfg.get("validation_truncate", 500),
        preprocessing_num_workers=cfg.get("preprocessing_num_workers", 8),
        preprocessing_batch_size=cfg.get("preprocessing_batch_size", 1000),
        is_main_process=is_main,
    )

    if is_main:
        print(f"\n✅ 학습 데이터: {len(train_dataset):,}건")
        print(f"✅ 검증 데이터: {len(eval_dataset):,}건" if eval_dataset else "✅ 검증 데이터: 없음")

    # ── 3. Liger Kernel ──────────────────────────────────────────────────────
    if cfg.get("use_liger_kernel", False):
        from liger_kernel.transformers import apply_liger_kernel_to_gemma3_text
        apply_liger_kernel_to_gemma3_text()
        if is_main:
            print("✅ Liger Kernel (Gemma3 text) 적용")

    # ── 4. 모델 로드 ─────────────────────────────────────────────────────────
    # VLM 체크포인트(language_model.* 키)를 Gemma3ForCausalLM으로 로드한다.
    model = _load_gemma3_causal_lm(
        cfg["model_name_or_path"],
        cfg.get("attn_implementation", "flash_attention_2"),
    )
    model.config.use_cache = False

    if is_main:
        print("✅ 모델 로드 완료")

    # ── 5. PiSSA LoRA 설정 ───────────────────────────────────────────────────
    peft_config = None
    if cfg.get("use_peft", True):
        target_modules = cfg.get("lora_target_modules", "all-linear")
        # PEFT 0.15+ 에서는 "all-linear" 문자열을 그대로 전달 (None 불가)
        if not isinstance(target_modules, str):
            target_modules = list(target_modules) if target_modules else "all-linear"

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.get("lora_r", 128),
            lora_alpha=cfg.get("lora_alpha", 256),
            lora_dropout=cfg.get("lora_dropout", 0.0),
            target_modules=target_modules,
            init_lora_weights=cfg.get("lora_init_weights", "pissa_niter_4"),
            use_rslora=cfg.get("use_rslora", False),
        )
        if is_main:
            print(f"✅ LoRA 설정 (PiSSA init={cfg.get('lora_init_weights')}  r={cfg.get('lora_r')})")

    # ── 6. WandB ─────────────────────────────────────────────────────────────
    if cfg.get("report_to") == "wandb" and is_main:
        run_name = f"{ts}_{cfg.get('run_name', 'pretrain')}"
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "LLM"),
            group=os.environ.get("WANDB_RUN_GROUP", "corpus-pretrain"),
            name=run_name,
            config=cfg,
        )

    # ── 7. SFTConfig ─────────────────────────────────────────────────────────
    output_dir = os.path.join(cfg["output_dir"], f"run_{ts}")

    # packing=True 일 때 warmup_steps를 config에서 직접 받음
    # packing=False 일 때는 자동 계산 (여기서는 packing=True 기본)
    warmup_steps = max(1, int(
        # 대략적인 global_step 계산 불가 (packing 때문에) → warmup_ratio 직접 사용
        # SFTConfig의 warmup_ratio 인자를 사용
        0
    ))

    sft_cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.get("num_train_epochs", 1),
        max_steps=cfg.get("max_steps", -1),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 2),
        eval_accumulation_steps=cfg.get("eval_accumulation_steps", 32),
        learning_rate=cfg.get("learning_rate", 2e-6),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        weight_decay=cfg.get("weight_decay", 0.0),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        optim=cfg.get("optim", "paged_adamw_32bit"),
        # eval / save
        do_eval=cfg.get("do_eval", True) and eval_dataset is not None,
        eval_on_start=cfg.get("eval_on_start", True) and eval_dataset is not None,
        eval_strategy="steps" if (eval_dataset and cfg.get("do_eval", True)) else "no",
        eval_steps=cfg.get("eval_steps", 74) if eval_dataset else None,
        save_strategy="steps",
        save_steps=cfg.get("save_steps", 74),
        save_only_model=cfg.get("save_only_model", True),
        torch_empty_cache_steps=cfg.get("torch_empty_cache_steps", 74),
        # 정밀도
        bf16=cfg.get("bf16", True),
        tf32=cfg.get("tf32", True),
        # 최적화
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs=cfg.get("gradient_checkpointing_kwargs", {"use_reentrant": False}),
        torch_compile=cfg.get("torch_compile", False),
        use_liger_kernel=cfg.get("use_liger_kernel", True),
        # 데이터
        packing=cfg.get("packing", True),
        padding_free=cfg.get("padding_free", False),
        max_length=cfg.get("max_length", 32768),
        remove_unused_columns=cfg.get("remove_unused_columns", False),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 10),
        dataloader_prefetch_factor=cfg.get("dataloader_prefetch_factor", 10),
        # Loss
        loss_type=cfg.get("loss_type", "nll"),
        completion_only_loss=False,  # pretrain: 전체 시퀀스 loss
        # logging
        report_to=cfg.get("report_to", "wandb"),
        logging_strategy=cfg.get("logging_strategy", "steps"),
        logging_steps=cfg.get("logging_steps", 1),
        run_name=f"{ts}_{cfg.get('run_name', 'pretrain')}",
        include_num_input_tokens_seen=cfg.get("include_num_input_tokens_seen", True),
        ddp_timeout=cfg.get("ddp_timeout", 18000000),
        seed=cfg.get("seed", 42),
        # metric — None으로 두면 _determine_best_metric이 eval_loss 조회를 건너뜀
        # (GOLD/KD trainer는 eval_loss를 반환하지 않을 수 있어 None이 안전)
        metric_for_best_model=None,
        greater_is_better=False,
        load_best_model_at_end=False,  # ZeRO3에서 OOM 방지
        save_total_limit=3,
        # 분산: Accelerate가 DeepSpeed 설정을 자동 적용
        dataloader_drop_last=True,
    )

    # ── 8. Trainer 생성 (우선순위: use_gold > use_gkd > SFTTrainer) ──────────
    use_gold = cfg.get("use_gold", False)
    use_gkd = cfg.get("use_gkd", False)

    if use_gold:
        # ── Plan 3: CorpusGOLDTrainer (Cross-tokenizer ULD KD) ───────────────
        gold_cfg = cfg.get("gold", {})
        teacher_path = gold_cfg.get("teacher_model_path")
        if not teacher_path:
            raise ValueError("use_gold=true 이지만 gold.teacher_model_path가 설정되지 않았습니다.")

        teacher_tok_path = gold_cfg.get("teacher_tokenizer_path", teacher_path)

        if is_main:
            print(f"\n[GOLD] 교사 모델 로드 중: {teacher_path}")

        teacher_model = load_teacher_model(
            teacher_path=teacher_path,
            attn_impl=cfg.get("attn_implementation", "flash_attention_2"),
        )

        teacher_tokenizer = AutoTokenizer.from_pretrained(
            teacher_tok_path,
            trust_remote_code=True,
            use_fast=True,
        )

        if is_main:
            print(
                f"✅ [GOLD] 교사 로드 완료\n"
                f"   teacher: {teacher_path}  (vocab={getattr(teacher_tokenizer, 'vocab_size', '?')})\n"
                f"   student: {cfg['model_name_or_path']}  (vocab={getattr(tokenizer, 'vocab_size', '?')})\n"
                f"   mode: {'same-tok' if type(tokenizer).__name__ == type(teacher_tokenizer).__name__ and getattr(tokenizer,'vocab_size',None) == getattr(teacher_tokenizer,'vocab_size',None) else 'cross-tok'}\n"
                f"   use_uld_loss={gold_cfg.get('use_uld_loss', True)}, "
                f"hybrid={gold_cfg.get('uld_use_hybrid_loss', False)}, "
                f"top_k={gold_cfg.get('uld_top_k', None)}"
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
            uld_top_k=gold_cfg.get("uld_top_k", None),
            model=model,
            args=sft_cfg,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            data_collator=None,
        )

    elif use_gkd:
        # ── Plan 2: CorpusKDTrainer (JSD KD, 동일 토크나이저) ────────────────
        gkd_cfg = cfg.get("gkd", {})
        teacher_path = gkd_cfg.get("teacher_model_path")
        if not teacher_path:
            raise ValueError("use_gkd=true 이지만 gkd.teacher_model_path가 설정되지 않았습니다.")

        if is_main:
            print(f"\n[GKD] 교사 모델 로드 중: {teacher_path}")

        teacher_model = load_teacher_model(
            teacher_path=teacher_path,
            attn_impl=cfg.get("attn_implementation", "flash_attention_2"),
        )

        if is_main:
            print(f"✅ [GKD] 교사 모델 로드 완료 (beta={gkd_cfg.get('beta', 0.5)}, temperature={gkd_cfg.get('temperature', 1.0)})")

        trainer = CorpusKDTrainer(
            teacher_model=teacher_model,
            beta=gkd_cfg.get("beta", 0.5),
            temperature=gkd_cfg.get("temperature", 1.0),
            model=model,
            args=sft_cfg,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            data_collator=None,
        )

    else:
        # ── 기본: SFTTrainer (NLL Loss) ───────────────────────────────────────
        trainer = SFTTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            data_collator=None,
        )

    if is_main:
        if use_gold:
            mode_str = (
                f"CorpusGOLDTrainer (ULD cross-tok, "
                f"teacher={cfg.get('gold', {}).get('teacher_model_path', '')})"
            )
        elif use_gkd:
            mode_str = (
                f"CorpusKDTrainer (JSD, "
                f"teacher={cfg.get('gkd', {}).get('teacher_model_path', '')})"
            )
        else:
            mode_str = "SFTTrainer (NLL)"
        print(f"✅ Trainer: {mode_str}")
        sample = train_dataset[0]
        preview = sample["text"][:300].replace("\n", "↵")
        print(f"\n학습 샘플 미리보기:\n  {preview}...\n")

    # ── 9. 학습 ──────────────────────────────────────────────────────────────
    trainer.train()

    # ── 10. 저장 ─────────────────────────────────────────────────────────────
    final_dir = os.path.join(output_dir, "final_model")

    if hasattr(trainer, "accelerator"):
        trainer.accelerator.wait_for_everyone()

    # ZeRO3: 모든 rank가 save_model 호출 (내부적으로 gather 후 rank0만 저장)
    trainer.save_model(final_dir)

    if is_main:
        tokenizer.save_pretrained(final_dir)
        print(f"\n✅ 저장 완료: {final_dir}")

    if hasattr(trainer, "accelerator"):
        trainer.accelerator.wait_for_everyone()

    if cfg.get("report_to") == "wandb" and is_main:
        wandb.finish()

    # timestamp 임시 파일 정리
    if is_main and world_size > 1 and os.path.exists(timestamp_file):
        os.remove(timestamp_file)


if __name__ == "__main__":
    main()
