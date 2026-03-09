#!/usr/bin/env python3
"""
Teacher forward eval context 단독 실험 (단일 GPU, ZeRO3 없이).
실행: python src/test_teacher_eval_forward.py
"""
import time, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

STUDENT_PATH = "/PROJECT/0325120095_A/BASE/rex/LLM/models/input/google/gemma-3-4b-pt"
TEACHER_PATH = "/PROJECT/0325120095_A/BASE/rex/LLM/models/input/Qwen/Qwen3.5-9B-Base"
ULD_TOP_K = 10000
SEQ_LEN   = 256
DEVICE    = "cuda:0" if torch.cuda.is_available() else "cpu"


def uld_loss(s_logits, t_logits, top_k=10000):
    K_s = min(top_k, s_logits.shape[-1])
    K_t = min(top_k, t_logits.shape[-1])
    s_p = F.softmax(s_logits.topk(K_s, dim=-1, sorted=True).values, dim=-1)
    t_p = F.softmax(t_logits.topk(K_t, dim=-1, sorted=True).values, dim=-1)
    K   = min(s_p.shape[-1], t_p.shape[-1])
    return (s_p[..., :K] - t_p[..., :K]).abs().sum(dim=-1).mean()


def run():
    print("=" * 60)
    print("Teacher eval forward 단독 실험")
    print(f"device={DEVICE}  seq_len={SEQ_LEN}  top_k={ULD_TOP_K}")
    print("=" * 60)

    s_tok = AutoTokenizer.from_pretrained(STUDENT_PATH)
    t_tok = AutoTokenizer.from_pretrained(TEACHER_PATH)
    text  = ("금융 리스크 관리는 포트폴리오 안정성의 핵심입니다. " * 15)

    s_ids  = s_tok(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)["input_ids"].to(DEVICE)
    t_ids  = t_tok(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)["input_ids"].to(DEVICE)
    s_mask = torch.ones_like(s_ids)
    t_mask = torch.ones_like(t_ids)
    print(f"student seq={s_ids.shape}  teacher seq={t_ids.shape}")

    print("\n[load] student...")
    t0      = time.time()
    student = AutoModelForCausalLM.from_pretrained(STUDENT_PATH, torch_dtype=torch.bfloat16).to(DEVICE)
    student.eval()
    print(f"  done {time.time()-t0:.1f}s")

    print("\n[load] teacher...")
    t0      = time.time()
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"  done {time.time()-t0:.1f}s")

    res = {}
    t_logits = None
    s_logits = None
    ce_loss  = None
    uld      = None

    print("\n[EXP1] teacher forward (no_grad)")
    try:
        t0 = time.time()
        with torch.no_grad():
            t_out = teacher(input_ids=t_ids, attention_mask=t_mask, use_cache=False)
        if "cuda" in DEVICE: torch.cuda.synchronize()
        t_logits = t_out.logits
        print(f"  OK  shape={t_logits.shape} {time.time()-t0:.2f}s")
        print(f"      min={t_logits.min():.3f} max={t_logits.max():.3f} mean={t_logits.mean():.3f}")
        res["exp1"] = True
    except Exception as e:
        print(f"  FAIL: {e}")
        res["exp1"] = False

    print("\n[EXP2] student CE loss (labels=input_ids)")
    try:
        t0 = time.time()
        with torch.no_grad():
            s_out = student(input_ids=s_ids, attention_mask=s_mask, labels=s_ids, use_cache=False)
        if "cuda" in DEVICE: torch.cuda.synchronize()
        ce_loss = s_out.loss
        print(f"  OK  ce_loss={ce_loss.item():.4f}  logits={s_out.logits}  {time.time()-t0:.2f}s")
        res["exp2"] = True
        res["ce"]   = ce_loss.item()
    except Exception as e:
        print(f"  FAIL: {e}")
        res["exp2"] = False

    print("\n[EXP3] student logits (no labels)")
    try:
        t0 = time.time()
        with torch.no_grad():
            s_out2 = student(input_ids=s_ids, attention_mask=s_mask, use_cache=False)
        if "cuda" in DEVICE: torch.cuda.synchronize()
        s_logits = s_out2.logits
        if s_logits is None:
            raise RuntimeError("logits=None")
        print(f"  OK  shape={s_logits.shape}  {time.time()-t0:.2f}s")
        res["exp3"] = True
    except Exception as e:
        print(f"  FAIL: {e}")
        res["exp3"] = False

    print("\n[EXP4] ULD loss")
    if res.get("exp1") and res.get("exp3") and t_logits is not None and s_logits is not None:
        try:
            min_T = min(s_logits.shape[1], t_logits.shape[1])
            t0 = time.time()
            with torch.no_grad():
                uld = uld_loss(s_logits[:, :min_T], t_logits[:, :min_T], ULD_TOP_K)
            if "cuda" in DEVICE: torch.cuda.synchronize()
            print(f"  OK  uld_loss={uld.item():.4f}  {time.time()-t0:.2f}s")
            res["exp4"] = True
            res["uld"]  = uld.item()
        except Exception as e:
            print(f"  FAIL: {e}")
            res["exp4"] = False
    else:
        print("  SKIP")
        res["exp4"] = False

    print("\n[EXP5] eval_loss = ULD x1.0 + CE x0.5")
    if res.get("exp2") and res.get("exp4") and ce_loss is not None and uld is not None:
        total = uld.item() + ce_loss.item() * 0.5
        print(f"  OK  {uld.item():.4f} + {ce_loss.item():.4f}x0.5 = {total:.4f}")
        res["exp5"] = True
    else:
        print("  SKIP")
        res["exp5"] = False

    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    descs = [
        ("exp1", "teacher forward (no_grad)"),
        ("exp2", "student CE loss (labels=input_ids)"),
        ("exp3", "student logits (no labels)"),
        ("exp4", "ULD loss"),
        ("exp5", "eval_loss = ULD + CE"),
    ]
    all_ok = True
    for k, d in descs:
        ok = res.get(k, False)
        print(f"  {'[OK  ]' if ok else '[FAIL]'}  {d}")
        if not ok:
            all_ok = False

    if all_ok and "uld" in res and "ce" in res:
        total = res["uld"] + res["ce"] * 0.5
        print(f"\n  CE={res['ce']:.4f}  ULD={res['uld']:.4f}  합산={total:.4f}")
        print("  -> eval에서 ULD+CE 합산 가능, gold_trainer.py 수정 가능")
    elif not all_ok:
        print("\n  -> 실패 항목 존재")


if __name__ == "__main__":
    run()
