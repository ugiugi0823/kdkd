#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CorpusKDTrainer — SFTTrainer 기반 Corpus Knowledge Distillation Trainer

기존 SFTTrainer의 모든 기능(packing, DeepSpeed ZeRO3, Liger Kernel 등)을
그대로 유지하면서, NLL loss 대신 JSD (Jensen-Shannon Divergence) loss를 사용한다.

Loss 수식:
    M_t   = beta * P_S(t) + (1 - beta) * P_T(t)   # 혼합 분포
    L_JSD = beta * KL(P_S ‖ M) + (1-beta) * KL(P_T ‖ M)

    beta = 0.0 → Forward KL  (교사 분포 커버리지 보존)
    beta = 0.5 → Symmetric JSD (균형, 기본값)
    beta = 1.0 → Reverse KL  (학생 분포 모드 집중)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from trl import SFTTrainer

# DeepSpeed ZeRO3 parameter gathering helper (optional import)
try:
    import deepspeed as _deepspeed
    _HAS_DEEPSPEED = True
except ImportError:
    _HAS_DEEPSPEED = False


class CorpusKDTrainer(SFTTrainer):
    """
    SFTTrainer를 상속하여 compute_loss()만 JSD 기반으로 교체한 KD Trainer.

    SFTConfig, packing, padding_free, Liger Kernel, DeepSpeed ZeRO3 등
    기존 모든 설정은 변경 없이 그대로 동작한다.

    Args:
        teacher_model: 추론 전용 교사 모델 (requires_grad=False 로 전달)
        beta: JSD 보간 계수. 0.0=forward KL, 0.5=symmetric JSD, 1.0=reverse KL
        temperature: 교사 logit에 적용할 softmax 온도. 1.0=변경 없음
        **kwargs: SFTTrainer에 전달되는 모든 인자
    """

    def __init__(
        self,
        teacher_model: torch.nn.Module,
        beta: float = 0.5,
        temperature: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.beta = beta
        self.temperature = temperature

    @staticmethod
    def _make_model_inputs(inputs: dict) -> dict:
        """labels 제거: Liger Kernel의 fused CE가 logits=None을 반환하는 것을 방지.

        KD는 전체 logits가 필요하므로 labels 없이 forward 후 mask는 별도 보관.
        Gemma3ForCausalLM(text-only)은 token_type_ids 불필요.
        """
        return {k: v for k, v in inputs.items() if k != "labels"}

    @staticmethod
    def _get_teacher_gather_params(teacher_model: torch.nn.Module) -> list:
        """ZeRO3 GatheredParameters 대상: Gemma3ForCausalLM 전체 파라미터."""
        return list(teacher_model.parameters())

    # ── eval: teacher 없이 표준 CE loss 사용 ────────────────────────────────

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Eval 단계에서는 teacher forward를 생략하고 표준 CE loss만 계산한다.
        - teacher GatheredParameters는 메모리 비용이 크므로 eval에서 제외
        - Eval 지표는 student 단독 perplexity로 측정
        """
        inputs = self._prepare_inputs(inputs)
        eval_inputs = self._make_model_inputs(inputs)
        # eval은 Liger fused CE로 loss 계산해도 되므로 labels 다시 추가
        if "labels" in inputs:
            eval_inputs["labels"] = inputs["labels"]
        with torch.no_grad():
            outputs = model(**eval_inputs)
            loss = outputs.loss if outputs.loss is not None else None

        if prediction_loss_only:
            return (loss, None, None)

        labels = inputs.get("labels")
        return (loss, None, labels)

    # ── loss ──────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        model,
        inputs: dict,
        return_outputs: bool = False,
        **kwargs,
    ):
        # ── Student forward ───────────────────────────────────────────────────
        labels = inputs.get("labels")     # (B, T)  -100 = 마스크
        model_inputs = self._make_model_inputs(inputs)   # labels 제거 + token_type_ids 주입
        outputs = model(**model_inputs)
        student_logits = outputs.logits   # (B, T, V)

        # ── Teacher forward (gradient 없음) ───────────────────────────────────
        # zero3_init_flag: true 환경에서는 teacher 파라미터도 ZeRO-3에 의해
        # 1D shard로 파티셔닝된다. GatheredParameters로 full 텐서를 복원한 뒤
        # forward를 수행하고, 완료 후 다시 파티션 상태로 반환한다.
        teacher_device = next(self.teacher_model.parameters()).device
        cuda_inputs = {k: v.to(teacher_device) if isinstance(v, torch.Tensor) else v
                       for k, v in model_inputs.items()}
        with torch.no_grad():
            if _HAS_DEEPSPEED:
                with _deepspeed.zero.GatheredParameters(
                    list(self.teacher_model.parameters()),
                    modifier_rank=None,
                ):
                    teacher_outputs = self.teacher_model(**cuda_inputs)
            else:
                teacher_outputs = self.teacher_model(**cuda_inputs)
            teacher_logits = teacher_outputs.logits   # (B, T, V)

        # ── Temperature scaling (teacher logit) ───────────────────────────────
        if self.temperature != 1.0:
            teacher_logits = teacher_logits / self.temperature

        # ── 유효 토큰 마스크 ──────────────────────────────────────────────────
        if labels is not None:
            mask = (labels != -100)   # (B, T)
        else:
            mask = torch.ones(
                student_logits.shape[:2],
                dtype=torch.bool,
                device=student_logits.device,
            )

        n_valid = int(mask.sum())
        if n_valid == 0:
            loss = student_logits.sum() * 0.0
            return (loss, outputs) if return_outputs else loss

        # ── 유효 토큰만 추출: (B,T,V) → (N,V) ──────────────────────────────
        # (B,T,V) fp32 텐서를 동시에 여러 개 유지하면 ~42 GB 이상 소요됨.
        # 유효 토큰(N)만 뽑아 (N,V)로 줄이고, 청크 단위로 JSD를 계산한다.
        s_valid = student_logits[mask].float()          # (N, V)
        t_valid = teacher_logits[mask].to(s_valid.device).float()  # (N, V)
        del student_logits, teacher_logits

        # ── 청크 단위 JSD 계산 (메모리 절약) ──────────────────────────────────
        # 청크 크기 512: 512 * 256000 * 4 byte ≈ 500 MB (텐서 3개 동시 → ~1.5 GB)
        CHUNK = 512
        log_beta = torch.log(torch.tensor(self.beta, device=s_valid.device, dtype=torch.float32))
        log_1mb = torch.log(torch.tensor(1.0 - self.beta, device=s_valid.device, dtype=torch.float32))

        jsd_chunks: list[torch.Tensor] = []
        debug_kl_s = debug_kl_t = 0.0

        for start in range(0, n_valid, CHUNK):
            s_c = F.log_softmax(s_valid[start:start + CHUNK], dim=-1)   # (C, V)
            t_c = F.log_softmax(t_valid[start:start + CHUNK], dim=-1)   # (C, V)

            # log_M = logsumexp(log_beta + log_S, log_(1-beta) + log_T)
            m_c = torch.logaddexp(s_c + log_beta, t_c + log_1mb)         # (C, V)

            kl_s_c = (s_c.exp() * (s_c - m_c)).sum(dim=-1)              # (C,)
            kl_t_c = (t_c.exp() * (t_c - m_c)).sum(dim=-1)              # (C,)
            jsd_chunks.append(self.beta * kl_s_c + (1.0 - self.beta) * kl_t_c)

            if self.state.global_step % 10 == 0:
                debug_kl_s += kl_s_c.sum().item()
                debug_kl_t += kl_t_c.sum().item()

            del s_c, t_c, m_c, kl_s_c, kl_t_c

        del s_valid, t_valid
        loss = torch.cat(jsd_chunks).mean()

        # ── 디버그 로깅 ────────────────────────────────────────────────────
        if self.state.global_step % 10 == 0:
            import os
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(
                    f"[JSD DEBUG] step={self.state.global_step}  "
                    f"loss_raw={loss.item():.8f}  "
                    f"mask_n={n_valid}  "
                    f"kl_s_mean={debug_kl_s / n_valid:.8f}  "
                    f"kl_t_mean={debug_kl_t / n_valid:.8f}",
                    flush=True,
                )

        return (loss, outputs) if return_outputs else loss
