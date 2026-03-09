from transformers import Gemma3ForCausalLM, AutoConfig
import torch

class _Gemma3CausalLMFromVLM(Gemma3ForCausalLM):
    base_model_prefix = "language_model"

path = "models/input/google/gemma-3-4b-pt"
vlm_cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
text_cfg = vlm_cfg.text_config
print("text_config:", type(text_cfg).__name__)

print("로드 테스트...")
model = _Gemma3CausalLMFromVLM.from_pretrained(
    path,
    config=text_cfg,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="cpu",
)
print("모델 타입:", type(model).__name__)
print("파라미터 수:", sum(p.numel() for p in model.parameters()) / 1e9, "B")

ids = torch.randint(0, 100, (1, 10))
with torch.no_grad():
    out = model(input_ids=ids)
print("logits shape:", out.logits.shape)
print("✅ 성공")
