#!/usr/bin/env python3
"""LoRA 어댑터를 Gemma3ForCausalLM 베이스 모델에 머지하는 스크립트.
학습 시 _Gemma3CausalLMFromVLM 방식과 동일하게 VLM 체크포인트를
text-only Gemma3ForCausalLM으로 로드 후 어댑터를 머지한다.
"""
import sys
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer, Gemma3ForCausalLM


class _Gemma3CausalLMFromVLM(Gemma3ForCausalLM):
    """VLM 체크포인트(language_model.* 키)에서 Gemma3ForCausalLM을 로드.
    base_model_prefix = "language_model" 으로 설정하면 transformers의
    prefix 제거 로직이 자동으로 처리한다.
    비전 타워 키는 UNEXPECTED로 무시된다.
    """
    base_model_prefix = "language_model"


def main():
    adapter_path = "/PROJECT/0325120095_A/BASE/rex/LLM/output/gemma-3-4b-pt/gold/run_20260306_141411/final_model"
    merged_path  = "/PROJECT/0325120095_A/BASE/rex/LLM/output/gemma-3-4b-pt/gold/run_20260306_141411/merged_model"
    base_model_id = "/PROJECT/0325120095_A/BASE/rex/LLM/models/input/google/gemma-3-4b-pt"

    print(f"Loading base model as Gemma3ForCausalLM from VLM checkpoint...")
    vlm_cfg = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
    text_cfg = vlm_cfg.text_config
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    model = _Gemma3CausalLMFromVLM.from_pretrained(
        base_model_id,
        config=text_cfg,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    print(f"Base model loaded. Architecture: {model.__class__.__name__}")

    print("Loading PEFT adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging and unloading adapter...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {merged_path} ...")
    model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    # save_pretrained 이 내부 클래스명(_Gemma3CausalLMFromVLM)을 저장하므로 수정
    import json
    cfg_path = f"{merged_path}/config.json"
    cfg = json.load(open(cfg_path))
    cfg["architectures"] = ["Gemma3ForCausalLM"]
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("config.json architectures fixed -> Gemma3ForCausalLM")
    print("Merge complete.")


if __name__ == "__main__":
    main()
