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
        # 티처는 각 rank의 전용 CUDA 장치에 완전히 로드되어 있으므로
        # ZeRO3 GatheredParameters 불필요 — 직접 forward만 수행한다.
        teacher_device = next(self.teacher_model.parameters()).device
        cuda_inputs = {k: v.to(teacher_device) if isinstance(v, torch.Tensor) else v
                       for k, v in model_inputs.items()}
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**cuda_inputs)
            teacher_logits = teacher_outputs.logits   # (B, T, V)

        # ── Temperature scaling (teacher logit) ───────────────────────────────
        if self.temperature != 1.0:
            teacher_logits = teacher_logits / self.temperature

        # ── 유효 토큰 마스크 ──────────────────────────────────────────────────
        # packing 환경에서는 labels != -100 인 위치만 loss 대상
        if labels is not None:
            mask = (labels != -100)   # (B, T)
        else:
            mask = torch.ones(
                student_logits.shape[:2],
                dtype=torch.bool,
                device=student_logits.device,
            )

        if mask.sum() == 0:
            # 유효 토큰이 전혀 없는 경우 (안전 처리)
            loss = student_logits.sum() * 0.0
            return (loss, outputs) if return_outputs else loss

        # ── 확률 분포 계산 (log-space, 수치 안정성 확보) ─────────────────────
        # BF16 환경에서 exp() 후 log()를 거치면 0 * (-inf) = NaN 발생.
        # log-space KL: KL(P||M) = sum_v exp(log_P) * (log_P - log_M)
        #   = sum_v P * log_P/M  (log-space에서 직접 계산)
        # log_M = log(beta*exp(log_S) + (1-beta)*exp(log_T))
        #       = log_S + log(beta + (1-beta)*exp(log_T - log_S))  (수치 안정)
        s_log_prob = F.log_softmax(student_logits.float(), dim=-1)   # fp32로 계산
        t_log_prob = F.log_softmax(teacher_logits.float(), dim=-1)
        del student_logits, teacher_logits

        # log_M = logsumexp trick으로 수치 안정하게 계산
        # log(beta*S + (1-beta)*T) = log(sum_i w_i * exp(log_x_i))
        log_weights = torch.stack(
            [s_log_prob + torch.log(torch.tensor(self.beta, device=s_log_prob.device)),
             t_log_prob + torch.log(torch.tensor(1.0 - self.beta, device=t_log_prob.device))],
            dim=0,
        )
        m_log_prob = torch.logsumexp(log_weights, dim=0)   # (B, T, V)
        del log_weights

        # KL(S||M) = sum_v S*(log_S - log_M), KL(T||M) = sum_v T*(log_T - log_M)
        kl_s = (s_log_prob.exp() * (s_log_prob - m_log_prob)).sum(dim=-1)   # (B, T)
        kl_t = (t_log_prob.exp() * (t_log_prob - m_log_prob)).sum(dim=-1)   # (B, T)
        del s_log_prob, t_log_prob, m_log_prob

        jsd_per_token = self.beta * kl_s + (1.0 - self.beta) * kl_t   # (B, T)

        # ── 유효 토큰 평균 ────────────────────────────────────────────────────
        loss = jsd_per_token[mask].mean()

        # ── 디버그: 실제 JSD 값 로깅 (round 전) ────────────────────────────
        if self.state.global_step % 10 == 0:
            import os
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(
                    f"[JSD DEBUG] step={self.state.global_step}  "
                    f"loss_raw={loss.item():.8f}  "
                    f"mask_n={int(mask.sum())}  "
                    f"kl_s_mean={kl_s[mask].mean().item():.8f}  "
                    f"kl_t_mean={kl_t[mask].mean().item():.8f}",
                    flush=True,
                )

        return (loss, outputs) if return_outputs else loss
