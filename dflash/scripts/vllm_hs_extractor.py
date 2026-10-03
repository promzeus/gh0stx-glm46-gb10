#!/usr/bin/env python3
"""
vllm_hs_extractor.py — снять hidden-states с NVFP4 v4 (GB10) для offline DFlash-трейна.
КЛЮЧ: input_ids строятся ТЕМ ЖЕ build_eagle3_dataset, что и train_dflash (chat_template/max_length),
чтобы хэш offline-backend совпал. Для каждого сэмпла: vLLM prefill -> forward-хуки на TARGET_LAYERS ->
конкат [seq, H*ncap] -> {out}/{md5(input_ids)}.pt = {input_ids, loss_mask, hidden_states}.
Запускать из specforge-ready (нужен specforge) с VLLM_ENABLE_V1_MULTIPROCESSING=0 (движок in-process -> хуки).

ENV: MODEL_DIR, OUT_DIR, DATA_JSONL, CHAT_TEMPLATE=qwen3.5, TARGET_LAYERS=1,15,29,43,57,
     MAX_LEN=2048, MAX_SAMPLES=optional
"""
import os, hashlib, torch
import torch.nn as nn


def log(m): print(f"[extract] {m}", flush=True)


def find_layers(root):
    best = None
    for name, mod in root.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 40:
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    return best


def main():
    MODEL_DIR = os.environ.get("MODEL_DIR", "/model")
    OUT_DIR = os.environ["OUT_DIR"]; os.makedirs(OUT_DIR, exist_ok=True)
    DATA = os.environ["DATA_JSONL"]
    CHAT = os.environ.get("CHAT_TEMPLATE", "qwen3.5")
    LAYERS = [int(x) for x in os.environ.get("TARGET_LAYERS", "1,15,29,43,57").split(",")]
    MAX_LEN = int(os.environ.get("MAX_LEN", "2048"))
    MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0")) or None

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from specforge.data import build_eagle3_dataset

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    ds = load_dataset("json", data_files=DATA)["train"]
    if MAX_SAMPLES: ds = ds.select(range(min(MAX_SAMPLES, len(ds))))
    eds = build_eagle3_dataset(dataset=ds, tokenizer=tok, chat_template=CHAT,
                               max_length=MAX_LEN, is_preformatted=False)
    log(f"датасет (build_eagle3): {len(eds)} сэмплов, колонки {eds.column_names}")

    log(f"init vLLM {MODEL_DIR} (nvfp4, eager, language_model_only)...")
    llm = LLM(model=MODEL_DIR, quantization="compressed-tensors", dtype="bfloat16",
              enforce_eager=True, gpu_memory_utilization=0.90, max_model_len=MAX_LEN + 128,
              trust_remote_code=True, enable_prefix_caching=False, language_model_only=True,
              load_format="runai_streamer")  # стрим в GPU (RAM 31G < 88G -> дефолтный загрузчик трэшит)
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    found = find_layers(runner.model)
    assert found, "не нашёл декодер-слои"
    lname, layers = found
    log(f"декодер-слои: '{lname}' ({len(layers)}); хуки на {LAYERS}")

    cap = {}
    def mk_hook(i):
        def hook(mod, inp, out):
            # vLLM Qwen3_5DecoderLayer.forward -> (hidden, residual): out[0]=MLP-дельта,
            # истинный residual stream (== HF outputs.hidden_states[i+1], == что vLLM-serve
            # кормит драфтеру) = out[0]+out[1]. Раньше брали только out[0] -> content-free дельта.
            if isinstance(out, tuple):
                h = out[0]
                if len(out) > 1 and out[1] is not None:
                    h = h + out[1]
            else:
                h = out
            cap[i] = h.detach().clone()   # clone: защита от in-place fused_add_rms_norm
        return hook
    handles = [layers[i].register_forward_hook(mk_hook(i)) for i in LAYERS]

    sp = SamplingParams(max_tokens=1, temperature=0)
    n = 0
    def _flat(t):
        if hasattr(t, "dim") and t.dim() == 2:
            t = t.squeeze(0)          # build_eagle3 хранит [1, seq] -> [seq]
        return t.tolist() if hasattr(t, "tolist") else list(t)
    for ex in eds:
        try:
            ids = _flat(ex["input_ids"])
            lm = _flat(ex["loss_mask"])
            if len(ids) < 8: continue
            key = hashlib.md5(",".join(map(str, ids)).encode()).hexdigest()
            fpath = os.path.join(OUT_DIR, key + ".pt")
            if os.path.exists(fpath):  # resume: не пересчитывать
                n += 1; continue
            cap.clear()
            llm.generate(TokensPrompt(prompt_token_ids=ids), sp, use_tqdm=False)
            hs = torch.cat([cap[i][:len(ids)].float().cpu() for i in LAYERS], dim=-1)
            if hs.shape[0] != len(ids):
                log(f"skip (len {hs.shape[0]}!={len(ids)})"); continue
            torch.save({"input_ids": torch.tensor(ids, dtype=torch.long),
                        "loss_mask": torch.tensor(lm, dtype=torch.long),
                        "hidden_states": hs.to(torch.bfloat16)}, fpath)
            n += 1
            if n <= 3 or n % 200 == 0: log(f"{n} ok, hs {tuple(hs.shape)}")
        except Exception as e:
            log(f"skip sample ({type(e).__name__}: {str(e)[:80]})"); continue
    for h in handles: h.remove()
    log(f"DONE: {n} сэмплов -> {OUT_DIR}")


if __name__ == "__main__":
    main()
