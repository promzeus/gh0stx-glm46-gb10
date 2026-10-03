#!/usr/bin/env python3
"""
vllm_hs_extractor_mtp_calib.py — снять hidden-states под калибровку keep-set MTP-головы.

Отличие от vllm_hs_extractor.py: вместо build_eagle3_dataset/specforge
(генерик-пайплайн под тренировку с loss_mask) — напрямую токенизирует calib_corpus.py's
доменные тексты (нужны ВСЕ токены для подсчёта routing, не только ответные), и сохраняет
domain вместе с input_ids/hidden_states. Хук на model.norm (POST-final-norm) и общая
механика запуска — БЕЗ ИЗМЕНЕНИЙ, уже верно сверены с живым vLLM (см. архивный файл).

ВЫРАВНИВАНИЕ (как в архивном экстракторе): hidden_states[j] <-> input_ids[j] — сдвиги
(+1 для embed-входа, +2 для label) делает калибровочный скрипт (calibrate_mtp_keepset.py),
не экстрактор.

ENV: MODEL_DIR (= /home/gh0stx/nvfp4_v4, тот же чекпойнт что serve), OUT_DIR,
     CALIB_TARGET_TOTAL=800, CALIB_USE_HF=1, MAX_LEN=2048.
Квантизация/dtype/флаги ДОЛЖНЫ совпадать с serve. Запускать из образа с установленным
vLLM, VLLM_ENABLE_V1_MULTIPROCESSING=0 (движок in-process -> хук видит модуль).
СЕРВ ДОЛЖЕН БЫТЬ ОСТАНОВЛЕН перед запуском (см. процедуру в плане: stop -> verify-free ->
cooldown -> этот скрипт -> stop -> verify-free -> cooldown -> restart серва).
"""
import os
import hashlib
import sys

import torch
import torch.nn as nn


def log(m):
    print(f"[extract-mtp-calib] {m}", flush=True)


def find_layers(root):
    """Longest ModuleList = target's decoder layers."""
    best = None
    for name, mod in root.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 40:
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    return best


def find_final_norm(root, layers_name):
    """Final RMSNorm = sibling of '.layers' in the same module (parent.norm)."""
    parent_name = layers_name.rsplit(".layers", 1)[0]
    parent = root.get_submodule(parent_name) if parent_name else root
    norm = getattr(parent, "norm", None)
    assert norm is not None, f"no .norm on '{parent_name}' (modules: {[n for n, _ in parent.named_children()]})"
    return f"{parent_name}.norm", norm


def main():
    MODEL_DIR = os.environ.get("MODEL_DIR", "/model")
    OUT_DIR = os.environ["OUT_DIR"]
    os.makedirs(OUT_DIR, exist_ok=True)
    CALIB_TARGET_TOTAL = int(os.environ.get("CALIB_TARGET_TOTAL", "800"))
    CALIB_USE_HF = os.environ.get("CALIB_USE_HF", "1") == "1"
    MAX_LEN = int(os.environ.get("MAX_LEN", "2048"))
    GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.70"))  # см. план: НЕ дефолт 0.90
    MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0")) or None  # канареечный прогон

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from calib_corpus import assemble

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    corpus = assemble(target_total=CALIB_TARGET_TOTAL, use_hf=CALIB_USE_HF, max_chars=MAX_LEN * 3)
    if MAX_SAMPLES:
        corpus = corpus[:MAX_SAMPLES]
    from collections import Counter
    by_domain = Counter(c["domain"] for c in corpus)
    log(f"calib corpus: {len(corpus)} sequences across domains {dict(by_domain)}")
    for d, n in by_domain.items():
        if n < 20:
            log(f"  !! WARNING: domain '{d}' has only {n} sequences (thin — see plan's floor check in Step A)")

    log(f"init vLLM {MODEL_DIR} (nvfp4/compressed-tensors, eager, language_model_only, "
        f"gpu_mem_util={GPU_MEM_UTIL})...")
    llm = LLM(
        model=MODEL_DIR, quantization="compressed-tensors", dtype="bfloat16",
        enforce_eager=True, gpu_memory_utilization=GPU_MEM_UTIL, max_model_len=MAX_LEN + 128,
        trust_remote_code=True, enable_prefix_caching=False, language_model_only=True,
        load_format="runai_streamer",
    )
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    found = find_layers(runner.model)
    assert found, "decoder layers not found"
    lname, layers = found
    nname, final_norm = find_final_norm(runner.model, lname)
    log(f"decoder layers: '{lname}' ({len(layers)}); final norm: '{nname}' -> hook on POST-norm output")

    cap = {}

    def norm_hook(mod, inp, out):
        # GemmaRMSNorm.forward(x, residual) -> (normed, residual); serve returns normed
        # (out[0]). Do NOT add out[0]+out[1] (that's the residual) — we want exactly the
        # POST-norm normed output, same as the archived MTP extractor (already verified
        # against live vLLM internals).
        h = out[0] if isinstance(out, tuple) else out
        cap["h"] = h.detach().clone()

    handle = final_norm.register_forward_hook(norm_hook)

    sp = SamplingParams(max_tokens=1, temperature=0)
    n = 0
    for item in corpus:
        try:
            text, domain = item["text"], item["domain"]
            ids = tok(text, truncation=True, max_length=MAX_LEN, add_special_tokens=True)["input_ids"]
            if len(ids) < 8:
                continue
            key = hashlib.md5((domain + ":" + ",".join(map(str, ids))).encode()).hexdigest()
            fpath = os.path.join(OUT_DIR, f"{domain}_{key}.pt")
            if os.path.exists(fpath):  # resume
                n += 1
                continue
            cap.clear()
            llm.generate(TokensPrompt(prompt_token_ids=ids), sp, use_tqdm=False)
            hs = cap["h"][: len(ids)].float().cpu()
            if hs.shape[0] != len(ids):
                log(f"skip (len {hs.shape[0]}!={len(ids)})")
                continue
            torch.save(
                {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "domain": domain,
                    "hidden_states": hs.to(torch.bfloat16),
                },
                fpath,
            )
            n += 1
            if n <= 3 or n % 100 == 0:
                log(f"{n} ok, hs {tuple(hs.shape)}, domain={domain}")
        except Exception as e:  # noqa: BLE001 — one bad sample must not kill the whole run
            log(f"skip sample ({type(e).__name__}: {str(e)[:80]})")
            continue
    handle.remove()
    log(f"DONE: {n} samples -> {OUT_DIR} (hidden_states = POST-final-norm, index-aligned with input_ids)")


if __name__ == "__main__":
    main()
