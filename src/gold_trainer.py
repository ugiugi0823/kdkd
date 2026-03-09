#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CorpusGOLDTrainer — Cross-Tokenizer ULD Knowledge Distillation Trainer
Plan 3 구현

대상:
  Student: google/gemma-3-4b-pt  (Gemma3 tokenizer, vocab 256,000)
  Teacher: Qwen/Qwen3.5-9B-Base  (Qwen3 tokenizer, vocab 151,936)

핵심 기능:
  1. ULD (Universal Logit Distillation): 정렬 기반 L1 거리 — vocab 무관
  2. 크로스-토크나이저 정렬: text span 기반으로 teacher logit을 student 위치에 매핑
  3. Hybrid ULD: matched vocab CE + unmatched ULD (부분 vocab 중복 시)
  4. CE 혼합: uld_crossentropy_weight > 0 이면 NLL loss 추가

ULD Loss 수식:
    P_S_sorted = sort(softmax(s_logits / T_s), descending=True)
    P_T_sorted = sort(softmax(t_logits / T_t), descending=True)
    K = min(V_S, V_T)
    L_ULD = mean( |P_S_sorted[:K] - P_T_sorted[:K]| )  [유효 토큰만]

크로스-토크나이저 정렬:
  Student token s_i → text span [char_a, char_b]
  Teacher tokens  t_j ... t_{j+k-1} covering same span
  → merged teacher logit at s_i = t_logits[t_{j+k-1}]
     (teacher 분포 "이 구간 이후 무엇이 오는가?" — 학생과 같은 예측 과제)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, Trainer
from trl import SFTTrainer

# DeepSpeed ZeRO3 parameter gathering helper (optional import)
try:
    import deepspeed as _deepspeed
    _HAS_DEEPSPEED = True
except ImportError:
    _HAS_DEEPSPEED = False

logger = logging.getLogger(__name__)


class CorpusGOLDTrainer(SFTTrainer):
    """
    SFTTrainer를 상속한 크로스-토크나이저 ULD Corpus KD Trainer.

    packing=True, raw text, DeepSpeed ZeRO3 환경을 그대로 유지하면서
    서로 다른 tokenizer를 쓰는 student/teacher 간의 지식증류를 지원한다.

    Args:
        teacher_model: 추론 전용 교사 모델 (requires_grad=False)
        teacher_tokenizer: 교사 모델 토크나이저
        student_tokenizer: 학생 모델 토크나이저 (processing_class와 동일)
        use_uld_loss: ULD Loss 사용 (기본 True)
        uld_crossentropy_weight: NLL CE 혼합 비중 (0.0=pure ULD)
        uld_distillation_weight: ULD 비중 (기본 1.0)
        uld_student_temperature: 학생 logit softmax 온도
        uld_teacher_temperature: 교사 logit softmax 온도
        uld_use_hybrid_loss: Hybrid ULD (matched CE + unmatched ULD)
        uld_hybrid_matched_weight: Hybrid matched 가중치
        uld_hybrid_unmatched_weight: Hybrid unmatched 가중치
        uld_skip_student_eos: Student EOS 위치 loss 제외
        uld_skip_teacher_eos: Teacher EOS 위치 loss 제외 (동일 토크나이저 시)
        uld_top_k: None=전체 vocab 정렬, int=상위 K개만 정렬 (속도 절감)
        **kwargs: SFTTrainer에 그대로 전달
    """

    def __init__(
        self,
        teacher_model: torch.nn.Module,
        teacher_tokenizer: PreTrainedTokenizerBase,
        student_tokenizer: PreTrainedTokenizerBase,
        use_uld_loss: bool = True,
        uld_crossentropy_weight: float = 0.0,
        uld_distillation_weight: float = 1.0,
        uld_student_temperature: float = 1.0,
        uld_teacher_temperature: float = 1.0,
        uld_use_hybrid_loss: bool = False,
        uld_hybrid_matched_weight: float = 1.0,
        uld_hybrid_unmatched_weight: float = 1.0,
        uld_skip_student_eos: bool = True,
        uld_skip_teacher_eos: bool = True,
        uld_top_k: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.use_uld_loss = use_uld_loss
        self.uld_crossentropy_weight = uld_crossentropy_weight
        self.uld_distillation_weight = uld_distillation_weight
        self.uld_student_temperature = uld_student_temperature
        self.uld_teacher_temperature = uld_teacher_temperature
        self.uld_use_hybrid_loss = uld_use_hybrid_loss
        self.uld_hybrid_matched_weight = uld_hybrid_matched_weight
        self.uld_hybrid_unmatched_weight = uld_hybrid_unmatched_weight
        self.uld_skip_student_eos = uld_skip_student_eos
        self.uld_skip_teacher_eos = uld_skip_teacher_eos
        self.uld_top_k = uld_top_k

        self._same_tokenizer: bool = self._is_same_tokenizer()
        if self._same_tokenizer:
            logger.info("CorpusGOLDTrainer: 동일 토크나이저 감지 — re-encoding 생략")
        else:
            logger.info(
                "CorpusGOLDTrainer: 크로스-토크나이저 모드 "
                f"(student vocab={getattr(student_tokenizer, 'vocab_size', '?')}, "
                f"teacher vocab={getattr(teacher_tokenizer, 'vocab_size', '?')})"
            )
            # fast tokenizer 여부 사전 확인
            self._student_is_fast = getattr(student_tokenizer, "is_fast", False)
            self._teacher_is_fast = getattr(teacher_tokenizer, "is_fast", False)
            if not self._student_is_fast or not self._teacher_is_fast:
                logger.warning(
                    "Slow tokenizer 감지 — offset_mapping 미지원. "
                    "token-string fallback 사용 (정확도 저하 가능)"
                )

    # ── Tokenizer 판별 ─────────────────────────────────────────────────────────

    def _is_same_tokenizer(self) -> bool:
        """vocab 크기 + tokenizer 클래스 비교로 동일 토크나이저 판별."""
        if type(self.student_tokenizer) is not type(self.teacher_tokenizer):
            return False
        sv = getattr(self.student_tokenizer, "vocab_size", None)
        tv = getattr(self.teacher_tokenizer, "vocab_size", None)
        return (sv is not None) and (tv is not None) and (sv == tv)

    # ── Teacher Forward: 동일 토크나이저 ──────────────────────────────────────

    def _teacher_forward_same_tok(
        self, inputs: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Student inputs를 teacher에 전달 (same tokenizer). ZeRO3 GatheredParameters safe.

        Returns:
            teacher_logits: (B, T, V_T)
            aligned_mask:   (B, T) bool — same-tokenizer는 모두 valid
        """
        device = next(
            (v.device for v in inputs.values() if isinstance(v, torch.Tensor)), None
        )
        logits_cpu: list[torch.Tensor] = []

        def _run():
            out = self.teacher_model(**inputs)
            if device is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
            logits_cpu.append(out.logits.cpu())

        with torch.no_grad():
            if _HAS_DEEPSPEED and hasattr(_deepspeed.zero, "GatheredParameters"):
                with _deepspeed.zero.GatheredParameters(
                    list(self.teacher_model.parameters()), modifier_rank=None
                ):
                    for p in self.teacher_model.parameters():
                        if p.data is not None and not p.data.is_contiguous():
                            p.data = p.data.contiguous()
                    _run()
            else:
                _run()

        logits = logits_cpu[0]
        if device is not None:
            logits = logits.to(device)

        # same-tokenizer: 모든 위치가 유효
        aligned_mask = torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device)
        return logits, aligned_mask  # (B, T, V_T), (B, T)

    # ── Teacher Forward: 크로스-토크나이저 ────────────────────────────────────

    def _teacher_forward_cross_tok(
        self,
        inputs: dict,
        T_S: int,
    ) -> torch.Tensor:
        """
        크로스-토크나이저 teacher forward.

        각 batch item에 대해:
          1. student input_ids → 텍스트 디코딩
          2. teacher tokenizer로 재인코딩 + offset_mapping 취득
          3. student offset_mapping 취득
          4. Teacher forward pass (one at a time)
          5. Character-level 정렬로 teacher logits를 student 위치에 매핑

        Returns:
            (B, T_S, V_T) — student 각 위치에 대응하는 teacher 분포
        """
        student_input_ids = inputs["input_ids"]  # (B, T_S)
        B = student_input_ids.shape[0]
        device = student_input_ids.device
        V_T = self.teacher_model.config.vocab_size

        # per-item 인코딩 (CPU)
        per_item_data: list[dict] = []
        for b in range(B):
            s_ids = student_input_ids[b].tolist()
            text = self.student_tokenizer.decode(
                s_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            t_enc = self._encode_with_offsets(
                self.teacher_tokenizer, text, is_fast=self._teacher_is_fast
            )
            s_enc = self._encode_with_offsets(
                self.student_tokenizer, text, is_fast=self._student_is_fast, max_length=T_S
            )
            per_item_data.append({
                "t_input_ids": t_enc["input_ids"],   # (T_T,) CPU
                "t_offset_map": t_enc["offset_mapping"],
                "s_offset_map": s_enc["offset_mapping"],
            })

        # Teacher forward: GatheredParameters로 ZeRO3 shard를 gather 후 실행
        # CUDA async 완료 전 gather 해제 방지를 위해 logits를 CPU로 복사 후 sync
        raw_logits_cpu: list[torch.Tensor] = []  # per-item (T_T, V_T) CPU

        def _do_teacher_forward():
            for item in per_item_data:
                t_ids = item["t_input_ids"].unsqueeze(0).to(device)   # (1, T_T) GPU
                t_mask = torch.ones_like(t_ids)
                t_out = self.teacher_model(
                    input_ids=t_ids, attention_mask=t_mask, use_cache=False
                )
                # GPU→CPU로 복사해 gathered param 해제 후에도 안전하게 참조
                raw_logits_cpu.append(t_out.logits[0].cpu())

        with torch.no_grad():
            if _HAS_DEEPSPEED and hasattr(_deepspeed.zero, "GatheredParameters"):
                with _deepspeed.zero.GatheredParameters(
                    list(self.teacher_model.parameters()), modifier_rank=None
                ):
                    # ZeRO3 gather 후 비연속 텐서 → contiguous 강제 (causal_conv1d 등 커널 안정성)
                    for p in self.teacher_model.parameters():
                        if p.data is not None and not p.data.is_contiguous():
                            p.data = p.data.contiguous()
                    _do_teacher_forward()
                    # CUDA async 커널 완료 전 gather 해제 방지
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
            else:
                _do_teacher_forward()

        # Character-level 정렬
        merged_list: list[torch.Tensor] = []
        mask_list: list[torch.Tensor] = []
        for b, item in enumerate(per_item_data):
            t_logits_gpu = raw_logits_cpu[b].to(device)  # (T_T, V_T) GPU
            merged, aligned_mask = self._align_teacher_logits(
                s_offset_map=item["s_offset_map"],
                t_offset_map=item["t_offset_map"],
                t_logits=t_logits_gpu,
                T_S=T_S,
                V_T=V_T,
            )  # (T_S, V_T), (T_S,)
            merged_list.append(merged)
            mask_list.append(aligned_mask)

        teacher_logits = torch.stack(merged_list, dim=0)   # (B, T_S, V_T)
        aligned_mask   = torch.stack(mask_list,   dim=0)   # (B, T_S)
        return teacher_logits, aligned_mask

    def _encode_with_offsets(
        self,
        tokenizer: PreTrainedTokenizerBase,
        text: str,
        is_fast: bool,
        max_length: Optional[int] = None,
    ) -> dict:
        """
        텍스트를 인코딩하고 character offset_mapping을 반환한다.

        Fast tokenizer: return_offsets_mapping=True 사용
        Slow tokenizer: token 문자열로부터 offset 추정 (fallback)

        Returns:
            {"input_ids": LongTensor(T,), "offset_mapping": list[(int,int)]}
        """
        kwargs = dict(
            add_special_tokens=False,
            truncation=(max_length is not None),
        )
        if max_length is not None:
            kwargs["max_length"] = max_length

        if is_fast:
            enc = tokenizer(text, return_offsets_mapping=True, **kwargs)
            input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
            offset_mapping = enc["offset_mapping"]
        else:
            enc = tokenizer(text, **kwargs)
            input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
            offset_mapping = self._estimate_offsets(tokenizer, enc["input_ids"], text)

        return {"input_ids": input_ids, "offset_mapping": offset_mapping}

    @staticmethod
    def _estimate_offsets(
        tokenizer: PreTrainedTokenizerBase,
        token_ids: list[int],
        text: str,
    ) -> list[tuple[int, int]]:
        """
        Slow tokenizer용 character offset 추정.
        각 token을 하나씩 디코딩하여 text 내 위치를 탐색한다.
        """
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for tid in token_ids:
            tok_str = tokenizer.decode([tid], skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
            if not tok_str:
                offsets.append((cursor, cursor))
                continue
            pos = text.find(tok_str, cursor)
            if pos == -1:
                # 찾지 못한 경우 (정규화 차이): 현재 위치에서 길이만큼 넘김
                end = cursor + len(tok_str)
                offsets.append((cursor, min(end, len(text))))
                cursor = min(end, len(text))
            else:
                end = pos + len(tok_str)
                offsets.append((pos, end))
                cursor = end
        return offsets

    def _align_teacher_logits(
        self,
        s_offset_map: list[tuple[int, int]],
        t_offset_map: list[tuple[int, int]],
        t_logits: torch.Tensor,
        T_S: int,
        V_T: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Character offset 기반으로 teacher logit을 student 위치에 매핑.

        Student 토큰 s_i가 text 구간 [s_a, s_b)를 커버할 때,
        같은 구간을 커버하는 teacher 토큰들 t_j..t_{j+k-1}을 찾는다.

        merged[s_i] = t_logits[t_{j+k-1}]
            → "이 구간 이후 무엇이 오는가?" teacher 분포
            → student s_logits[i]와 같은 예측 과제 (다음 토큰 예측)

        1:1 매핑 (k=1): t_logits[t_j] 그대로 사용
        1:N 매핑 (k>1): 마지막 teacher 위치 t_logits[t_{j+k-1}] 사용

        Returns:
            merged:        (T_S, V_T) — 정렬된 teacher logits (실패 위치는 zeros)
            aligned_mask:  (T_S,) bool — True = 정렬 성공, False = 실패 (ULD 제외 대상)
        """
        device = t_logits.device
        T_T = t_logits.shape[0]
        merged = torch.zeros(T_S, V_T, device=device, dtype=t_logits.dtype)
        aligned_mask = torch.zeros(T_S, dtype=torch.bool, device=device)

        t_ptr = 0  # teacher 포인터 (단조 증가)

        for s_i in range(min(T_S, len(s_offset_map))):
            s_a, s_b = s_offset_map[s_i]

            # ── 특수 토큰: character span 없음 ───────────────────────────────
            if s_a == s_b:
                while t_ptr < T_T:
                    t_a, t_b = t_offset_map[t_ptr]
                    if t_a == t_b:
                        merged[s_i] = t_logits[t_ptr]
                        aligned_mask[s_i] = True
                        t_ptr += 1
                        break
                    break
                continue

            # ── 일반 토큰: [s_a, s_b) 구간을 커버하는 teacher 토큰들 수집 ───
            t_start = t_ptr

            while t_ptr < T_T:
                t_a, t_b = t_offset_map[t_ptr]
                if t_b <= s_a:
                    t_ptr += 1
                    t_start = t_ptr
                    continue
                if t_a >= s_b:
                    break
                t_ptr += 1

            t_end = t_ptr  # teacher span: [t_start, t_end)

            if t_start >= t_end:
                # 정렬 실패 → zeros 유지, mask=False
                continue

            if t_end > T_T:
                t_end = T_T

            last_t = t_end - 1
            merged[s_i] = t_logits[last_t]
            aligned_mask[s_i] = True

        return merged, aligned_mask

    # ── ULD Loss ───────────────────────────────────────────────────────────────

    def _uld_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Universal Logit Distillation (ULD) Loss — 정렬 기반 L1 거리.

        두 분포를 각각 내림차순 정렬 후 min(V_S, V_T)개까지 L1 비교.
        vocab 크기가 달라도 적용 가능 (Gemma3 256k ↔ Qwen3 152k).

        Args:
            student_logits: (B, T, V_S)
            teacher_logits: (B, T, V_T)  — 이미 student 위치로 정렬된 상태
            mask: (B, T) bool — True 위치만 loss 계산

        Returns:
            scalar loss
        """
        s_logits = student_logits / self.uld_student_temperature
        t_logits = teacher_logits / self.uld_teacher_temperature

        # ── 메모리 효율적 ULD: full vocab softmax 대신 top-K logits만 추출 ─────────
        # softmax(logits) 정렬 ≡ logits 정렬 (단조성). topk(logits)로 top-K를 먼저
        # 뽑은 뒤 해당 부분만 softmax → (B, T, V) 텐서 생성을 피함.
        # (예: V_S=262K, T=32K, B=2 → full softmax = 34 GiB → OOM)
        V_S = s_logits.shape[-1]
        V_T = t_logits.shape[-1]

        if self.uld_top_k is not None:
            K_s = min(self.uld_top_k, V_S)
            K_t = min(self.uld_top_k, V_T)
        else:
            K_s, K_t = V_S, V_T

        if K_s < V_S:
            # top-K 로짓 추출 → 해당 부분만 softmax (근사: tail mass 무시)
            s_topk = s_logits.topk(K_s, dim=-1, sorted=True).values   # (B, T, K_s)
            s_sorted = F.softmax(s_topk, dim=-1)                       # (B, T, K_s)
        else:
            s_sorted = F.softmax(s_logits, dim=-1).sort(dim=-1, descending=True).values

        if K_t < V_T:
            t_topk = t_logits.topk(K_t, dim=-1, sorted=True).values   # (B, T, K_t)
            t_sorted = F.softmax(t_topk, dim=-1)                       # (B, T, K_t)
        else:
            t_sorted = F.softmax(t_logits, dim=-1).sort(dim=-1, descending=True).values

        K = min(s_sorted.shape[-1], t_sorted.shape[-1])
        s_sorted = s_sorted[..., :K]   # (B, T, K)
        t_sorted = t_sorted[..., :K]   # (B, T, K)

        l1_per_token = (s_sorted - t_sorted).abs().sum(dim=-1)  # (B, T)

        mask_f = mask.float()
        loss = (l1_per_token * mask_f).sum() / mask_f.sum().clamp(min=1.0)
        return loss

    def _hybrid_uld_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hybrid ULD Loss: 공통 vocab 범위는 CE, 초과 범위는 ULD.

        vocab ID가 양쪽에 모두 존재하는 범위 [0, K) 에서는 CE로 직접 비교하고,
        그 외 범위는 정렬 기반 L1으로 비교한다.

        Gemma3 ↔ Qwen3 조합에서는 공통 vocab 교집합이 거의 없으므로
        실질적으로 pure ULD에 수렴한다. 부분 vocab 중복 모델 쌍에 유용.

        Note: 이 구현은 token ID 범위 [0, min(V_S, V_T)) 를 matched로 간주.
              완전한 string-level vocab 교집합 계산은 별도 사전 구축 필요.
        """
        V_S = student_logits.shape[-1]
        V_T = teacher_logits.shape[-1]
        K = min(V_S, V_T)  # 공통 범위 상한

        mask_f = mask.float()
        denom = mask_f.sum().clamp(min=1.0)

        s_prob = F.softmax(student_logits / self.uld_student_temperature, dim=-1)
        t_prob = F.softmax(teacher_logits / self.uld_teacher_temperature, dim=-1)

        # Matched CE: student log-prob vs teacher prob (공통 범위 내)
        s_log = F.log_softmax(student_logits[..., :K] / self.uld_student_temperature, dim=-1)
        t_matched = t_prob[..., :K]
        ce_per_token = -(t_matched * s_log).sum(dim=-1)  # (B, T)
        l_matched = (ce_per_token * mask_f).sum() / denom

        # Unmatched ULD: 양쪽의 공통 범위 초과 부분
        l_unmatched = torch.tensor(0.0, device=student_logits.device)
        s_excess = s_prob[..., K:]   # (B, T, V_S - K)
        t_excess = t_prob[..., K:]   # (B, T, V_T - K)
        if s_excess.shape[-1] > 0 and t_excess.shape[-1] > 0:
            K2 = min(s_excess.shape[-1], t_excess.shape[-1])
            s_ex_sorted = s_excess.sort(dim=-1, descending=True).values[..., :K2]
            t_ex_sorted = t_excess.sort(dim=-1, descending=True).values[..., :K2]
            l1_ex = (s_ex_sorted - t_ex_sorted).abs().sum(dim=-1)
            l_unmatched = (l1_ex * mask_f).sum() / denom

        return (self.uld_hybrid_matched_weight * l_matched
                + self.uld_hybrid_unmatched_weight * l_unmatched)

    # ── prediction_step + eval 우회 ───────────────────────────────────────────

    def _determine_best_metric(self, metrics, trial):
        """eval_loss 없어도 KeyError 없이 통과."""
        return False

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Eval: Trainer.prediction_step의 has_labels 체크를 완전히 우회.

        use_liger_kernel=True 시 SFTTrainer가 eval dataset에서 labels를 제거하므로
        has_labels=False → Trainer.prediction_step이 compute_loss를 호출하지 않음.
        여기서 직접 compute_loss를 호출해 CE loss를 계산한다.
        """
        inputs = self._prepare_inputs(inputs)
        inputs["_prediction_loss_only"] = prediction_loss_only

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if loss is not None:
            loss = loss.detach().mean()

        return (loss, None, None)

    # ── compute_loss ───────────────────────────────────────────────────────────

    def compute_loss(
        self,
        model,
        inputs: dict,
        return_outputs: bool = False,
        **kwargs,
    ):
        # ── Eval 모드: ULD + CE 합산 eval_loss 계산 ──────────────────────────
        # SFTTrainer가 prediction_step에서 inputs["_prediction_loss_only"] 를 주입.
        # model.training은 ZeRO3 wrapper에서 eval 중에도 True일 수 있으므로
        # 이 키를 eval 신호로 사용한다.
        is_eval = "_prediction_loss_only" in inputs
        if is_eval or not model.training:
            eval_inputs = {
                k: v for k, v in inputs.items() if not k.startswith("_")
            }
            eval_inputs["use_cache"] = False

            # ── Student: logits 취득 (labels 제외 → Liger fused CE 우회) ─────
            student_forward_inputs = {
                k: v for k, v in eval_inputs.items() if k != "labels"
            }
            with torch.no_grad():
                outputs = model(**student_forward_inputs)
            student_logits = outputs.logits  # (B, T_S, V_S)

            if student_logits is None:
                # Liger가 labels를 받아 fused CE 처리 — labels를 붙여서 재시도
                if "input_ids" in eval_inputs:
                    eval_inputs["labels"] = eval_inputs["input_ids"]
                with torch.no_grad():
                    outputs = model(**eval_inputs)
                loss = outputs.loss
                return (loss, outputs) if return_outputs else loss

            B, T_S = student_logits.shape[:2]
            device  = student_logits.device
            mask    = torch.ones(B, T_S, dtype=torch.bool, device=device)

            # ── Teacher forward (no_grad, GatheredParameters) ────────────────
            with torch.no_grad():
                if self._same_tokenizer:
                    teacher_logits, aligned_mask = self._teacher_forward_same_tok(eval_inputs)
                else:
                    teacher_logits, aligned_mask = self._teacher_forward_cross_tok(eval_inputs, T_S)

            # ── Alignment 실패 위치 제외 ──────────────────────────────────────
            uld_mask = mask & aligned_mask   # (B, T_S)

            # ── ULD loss ──────────────────────────────────────────────────────
            if self.uld_use_hybrid_loss:
                distill_loss = self._hybrid_uld_loss(student_logits, teacher_logits, uld_mask)
            else:
                distill_loss = self._uld_loss(student_logits, teacher_logits, uld_mask)

            eval_loss = self.uld_distillation_weight * distill_loss

            # ── CE loss (학습과 동일 가중치) ───────────────────────────────────
            if self.uld_crossentropy_weight > 0.0:
                shift_logits = student_logits[:, :-1, :].contiguous()
                shift_labels = eval_inputs.get(
                    "labels",
                    eval_inputs.get("input_ids", student_logits.new_zeros(B, T_S, dtype=torch.long))
                )[:, 1:].contiguous()
                ce_loss  = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                eval_loss = eval_loss + self.uld_crossentropy_weight * ce_loss

            return (eval_loss, outputs) if return_outputs else eval_loss

        # ── Student forward (train) ─────────────────────────────────────────────
        # labels를 제거하고 forward → 항상 raw logits 취득
        # (Liger Kernel fused CE는 labels가 있으면 logits=None으로 반환)
        labels = inputs.get("labels")     # (B, T_S)
        forward_inputs = {
            k: v for k, v in inputs.items()
            if k not in ("labels", "_prediction_loss_only")
        }
        outputs = model(**forward_inputs)
        student_logits = outputs.logits   # (B, T_S, V_S)

        B, T_S = student_logits.shape[:2]

        # ── 유효 토큰 마스크 ──────────────────────────────────────────────────
        if labels is not None:
            mask = (labels != -100)
        else:
            mask = torch.ones(B, T_S, dtype=torch.bool, device=student_logits.device)

        # Student EOS 위치 제외
        if self.uld_skip_student_eos and labels is not None:
            eos_id = getattr(self.student_tokenizer, "eos_token_id", None)
            if eos_id is not None:
                mask = mask & (labels != eos_id)

        if mask.sum() == 0:
            loss = student_logits.sum() * 0.0
            return (loss, outputs) if return_outputs else loss

        # ── Teacher forward ────────────────────────────────────────────────────
        if self._same_tokenizer:
            teacher_logits, aligned_mask = self._teacher_forward_same_tok(inputs)
        else:
            teacher_logits, aligned_mask = self._teacher_forward_cross_tok(inputs, T_S)

        # ── Alignment 실패 위치 제외: ULD mask = valid label & 정렬 성공 ──────
        uld_mask = mask & aligned_mask   # (B, T_S)

        # ── Alignment 실패율 진단 로깅 (매 50 step마다) ───────────────────────
        if self.state.global_step % 50 == 0:
            total_valid   = mask.sum().item()
            total_aligned = uld_mask.sum().item()
            fail_rate     = 1.0 - (total_aligned / max(total_valid, 1))
            logger.warning(
                f"[Step {self.state.global_step}] "
                f"Alignment: {int(total_aligned)}/{int(total_valid)} valid "
                f"(fail_rate={fail_rate:.1%})"
            )

        # ── Distillation Loss ─────────────────────────────────────────────────
        if self.uld_use_hybrid_loss:
            distill_loss = self._hybrid_uld_loss(student_logits, teacher_logits, uld_mask)
        else:
            distill_loss = self._uld_loss(student_logits, teacher_logits, uld_mask)

        loss = self.uld_distillation_weight * distill_loss

        # ── CE 혼합 (선택) ────────────────────────────────────────────────────
        if self.uld_crossentropy_weight > 0.0 and labels is not None:
            # SFTTrainer 방식: shift는 model 내부에서 처리하지 않으므로
            # labels이 이미 input_ids와 동일한 포맷 (같은 위치, 모델이 내부에서 shift)
            # HF CausalLM loss 방식: shift_logits[:-1] vs shift_labels[1:]
            shift_logits = student_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = loss + self.uld_crossentropy_weight * ce_loss

        return (loss, outputs) if return_outputs else loss
