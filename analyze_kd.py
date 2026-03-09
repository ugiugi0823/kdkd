"""
GKD 코퍼스 학습 효과 분석
- vllm을 이용한 퍼플렉시티(PPL) 비교: GKD 모델 vs Baseline
- 한국어 금융 텍스트 생성 비교
"""
import os, sys, json, math, random
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

BASE_DIR = Path("/PROJECT/0325120095_A/BASE/rex/LLM")
GKD_MODEL  = str(BASE_DIR / "output/gemma-3-4b-pt/kd/run_20260305_162921/final_model")
BASE_MODEL = str(BASE_DIR / "models/input/google/gemma-3-4b-pt")
DATA_DIR1  = Path("/xtmp/jp1924_MultilingualCorpusInFinancialSector")
DATA_DIR2  = Path("/xtmp/ugiugi_korean_financial_corpus")

# ── 1. 평가 데이터 샘플링 ──────────────────────────────────────────────────────
print("=" * 60)
print("1. 평가 데이터 로드")
print("=" * 60)

from datasets import load_from_disk
import numpy as np

random.seed(42)
np.random.seed(42)

ds1 = load_from_disk(str(DATA_DIR1))
ds2 = load_from_disk(str(DATA_DIR2))

# validation 50건씩 합쳐서 100건
val1 = ds1["validation"].select(range(min(50, len(ds1["validation"]))))
val2 = ds2["validation"].select(range(min(50, len(ds2["validation"]))))

def get_text(sample):
    """sentence_ls 필드에서 텍스트 합치기"""
    parts = sample.get("sentence_ls", [])
    if isinstance(parts, list):
        return "\n".join(str(p) for p in parts if p)
    return str(parts)

val_texts = [get_text(s) for s in val1] + [get_text(s) for s in val2]
val_texts = [t for t in val_texts if len(t.strip()) > 50]  # 너무 짧은 것 제거
print(f"  평가 샘플: {len(val_texts)}건")
print(f"  평균 길이: {np.mean([len(t) for t in val_texts]):.0f} 문자")

# PPL 계산용: 각 텍스트를 앞 512 토큰으로 자르기
# 생성 테스트용: 앞 100자로 프롬프트 → 뒤 생성
ppl_texts = val_texts  # 전부 사용
gen_prompts = [t[:150] for t in val_texts[:10]]  # 10개만 생성 테스트

# ── 2. 프롬프트 미리보기 ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. 평가 샘플 미리보기")
print("=" * 60)
for i, t in enumerate(val_texts[:3]):
    print(f"\n[샘플 {i+1}] {t[:200]}...")

# ── 3. vllm으로 두 모델 PPL 계산 ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. 퍼플렉시티(PPL) 계산")
print("=" * 60)

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def compute_ppl(model_path, texts, max_tokens=512, batch_size=50, block_size=32):
    """
    vllm을 사용해 텍스트 리스트의 평균 PPL 계산
    logprobs 합산 → PPL = exp(-mean_logprob)
    """
    print(f"\n  모델 로드: {model_path.split('/')[-2]}/{model_path.split('/')[-1]}")
    
    extra_args = {}
    # gemma3_text 모델은 block_size=32 필요
    model_cfg_path = Path(model_path) / "config.json"
    if model_cfg_path.exists():
        cfg = json.loads(model_cfg_path.read_text())
        if cfg.get("model_type") == "gemma3_text":
            extra_args["block_size"] = block_size

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=1024,
        **extra_args,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 텍스트를 토큰화하여 max_tokens로 자르기
    truncated = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=True)[:max_tokens]
        truncated.append(tokenizer.decode(ids, skip_special_tokens=False))

    # logprobs 계산: prompt 전체를 생성 없이 점수만
    params = SamplingParams(
        max_tokens=1,          # 최소 1토큰 생성 필요
        prompt_logprobs=1,     # prompt 각 토큰의 logprob
        temperature=0,
    )

    results = llm.generate(truncated, params)

    total_logprob = 0.0
    total_tokens  = 0
    per_text_ppls = []

    for r in results:
        # prompt_logprobs는 RequestOutput 직속 속성
        prompt_lps = r.prompt_logprobs  # list[dict|None] or None
        if prompt_lps is None:
            continue
        lp_sum = 0.0
        count  = 0
        for lp_entry in prompt_lps:
            if lp_entry is None:
                continue
            # lp_entry: dict[token_id → Logprob]
            for tok_lp in lp_entry.values():
                lp_sum += tok_lp.logprob
                count  += 1
                break  # 첫 번째 (greedy) 토큰만
        if count > 0:
            ppl = math.exp(-lp_sum / count)
            per_text_ppls.append(ppl)
            total_logprob += lp_sum
            total_tokens  += count

    overall_ppl = math.exp(-total_logprob / total_tokens) if total_tokens > 0 else float("inf")
    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    return overall_ppl, per_text_ppls


# Baseline PPL
ppl_base, per_base = compute_ppl(BASE_MODEL, ppl_texts)
print(f"\n  Baseline PPL: {ppl_base:.4f}")

# GKD PPL
ppl_gkd, per_gkd = compute_ppl(GKD_MODEL, ppl_texts)
print(f"  GKD     PPL: {ppl_gkd:.4f}")

ppl_delta = ppl_base - ppl_gkd
ppl_rel   = ppl_delta / ppl_base * 100
print(f"\n  PPL 개선: {ppl_delta:+.4f} ({ppl_rel:+.2f}%)")
print(f"  {'✅ GKD가 낮은 PPL (학습 효과 확인)' if ppl_gkd < ppl_base else '⚠️ Baseline이 낮은 PPL'}")

# ── 4. 텍스트 생성 비교 ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. 텍스트 생성 비교 (top-5 프롬프트)")
print("=" * 60)

def generate_texts(model_path, prompts, max_new_tokens=200, block_size=32):
    extra_args = {}
    model_cfg_path = Path(model_path) / "config.json"
    if model_cfg_path.exists():
        cfg = json.loads(model_cfg_path.read_text())
        if cfg.get("model_type") == "gemma3_text":
            extra_args["block_size"] = block_size

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=1024,
        **extra_args,
    )
    params = SamplingParams(max_tokens=max_new_tokens, temperature=0.1, top_p=0.9)
    outputs = llm.generate(prompts, params)
    texts   = [o.outputs[0].text for o in outputs]
    del llm
    import gc, torch; gc.collect(); torch.cuda.empty_cache()
    return texts


gen_base = generate_texts(BASE_MODEL, gen_prompts)
gen_gkd  = generate_texts(GKD_MODEL,  gen_prompts)

for i, (prompt, base_out, gkd_out) in enumerate(zip(gen_prompts, gen_base, gen_gkd)):
    print(f"\n{'━'*60}")
    print(f"[프롬프트 {i+1}]")
    print(f"  {prompt[:100]}...")
    print(f"\n[Baseline 생성]")
    print(f"  {base_out[:300]}")
    print(f"\n[GKD 생성]")
    print(f"  {gkd_out[:300]}")

# ── 5. Per-sample PPL 분포 통계 ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Per-sample PPL 통계")
print("=" * 60)

import numpy as np
ppls_b = np.array(per_base)
ppls_g = np.array(per_gkd)

print(f"\n{'지표':<20} {'Baseline':>12} {'GKD':>12} {'차이':>12}")
print("-" * 60)
print(f"{'Mean PPL':<20} {ppls_b.mean():>12.3f} {ppls_g.mean():>12.3f} {ppls_g.mean()-ppls_b.mean():>+12.3f}")
print(f"{'Median PPL':<20} {np.median(ppls_b):>12.3f} {np.median(ppls_g):>12.3f} {np.median(ppls_g)-np.median(ppls_b):>+12.3f}")
print(f"{'Std PPL':<20} {ppls_b.std():>12.3f} {ppls_g.std():>12.3f} {ppls_g.std()-ppls_b.std():>+12.3f}")
print(f"{'Min PPL':<20} {ppls_b.min():>12.3f} {ppls_g.min():>12.3f} {ppls_g.min()-ppls_b.min():>+12.3f}")
print(f"{'Max PPL':<20} {ppls_b.max():>12.3f} {ppls_g.max():>12.3f} {ppls_g.max()-ppls_b.max():>+12.3f}")

# GKD가 Baseline보다 낮은 샘플 비율
n_improved = (ppls_g < ppls_b).sum()
print(f"\n  GKD PPL < Baseline 샘플: {n_improved}/{len(ppls_g)} ({n_improved/len(ppls_g)*100:.1f}%)")

# ── 6. 결과 저장 ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. 결과 저장")
print("=" * 60)

out_dir = BASE_DIR / "logs/analysis"
out_dir.mkdir(exist_ok=True)

result = {
    "overall": {
        "baseline_ppl": ppl_base,
        "gkd_ppl":      ppl_gkd,
        "ppl_delta":    ppl_delta,
        "ppl_rel_pct":  ppl_rel,
    },
    "stats": {
        "baseline": {"mean": float(ppls_b.mean()), "median": float(np.median(ppls_b)), "std": float(ppls_b.std())},
        "gkd":      {"mean": float(ppls_g.mean()), "median": float(np.median(ppls_g)), "std": float(ppls_g.std())},
        "n_improved": int(n_improved),
        "n_total":    len(ppls_g),
    },
    "generation": [
        {"prompt": p, "baseline": b, "gkd": g}
        for p, b, g in zip(gen_prompts, gen_base, gen_gkd)
    ],
    "per_sample_ppl": {
        "baseline": ppls_b.tolist(),
        "gkd":      ppls_g.tolist(),
    },
}

out_path = out_dir / "kd_analysis.json"
out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(f"  저장 완료: {out_path}")

print("\n" + "=" * 60)
print("분석 완료!")
print("=" * 60)
print(f"\n  📊 전체 PPL:  Baseline={ppl_base:.3f}  GKD={ppl_gkd:.3f}  (개선={ppl_rel:+.2f}%)")
print(f"  📈 개선 샘플: {n_improved}/{len(ppls_g)} ({n_improved/len(ppls_g)*100:.1f}%)")
