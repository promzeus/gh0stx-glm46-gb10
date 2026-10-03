#!/usr/bin/env python3
"""Generate self-distilled calibration traces from the bf16 pruned model.

The NVFP4 W4A4 checkpoint degenerates in long <think> reasoning because the
stock ultrachat_200k calibration never sees the model's own reasoning-mode
activation distribution (QuantLRM, arXiv 2602.02581: standard PTQ calibration
misses reasoning intermediate states -> repetition/incoherence). Here we run the
coherent bf16 model on diverse instructions with thinking enabled and save the
full inference token stream (prompt + generated <think>+answer) as calibration
input_ids. Feeding these into oneshot matches calibration activations to the real
inference distribution.
"""
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.environ.get("GEN_IN", "/data/pruned")
OUT = os.environ.get("GEN_OUT", "/data/calib.jsonl")
N = int(os.environ.get("GEN_SAMPLES", "128"))
MAX_NEW = int(os.environ.get("GEN_MAXNEW", "512"))

print(f"[gen] loading tokenizer from {MODEL_DIR}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

print(f"[gen] loading bf16 model (device_map=auto) from {MODEL_DIR}", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
model.eval()

print(f"[gen] seeds: ultrachat_200k train_sft[:{N}]", flush=True)
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{N}]").shuffle(seed=42)

written = 0
with open(OUT, "w") as f:
    for i, ex in enumerate(ds):
        user = ex["messages"][0]["content"]
        enc = tok.apply_chat_template(
            [{"role": "user", "content": user}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(enc, "input_ids"):
            input_ids = enc.input_ids
        elif isinstance(enc, dict):
            input_ids = enc["input_ids"]
        else:
            input_ids = enc
        input_ids = input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=20,
            )
        full = out[0].tolist()
        f.write(json.dumps({"input_ids": full}) + "\n")
        f.flush()
        written += 1
        print(f"[gen] {written}/{N} len={len(full)}", flush=True)

print(f"[gen] GEN_DONE wrote={written} -> {OUT}", flush=True)
