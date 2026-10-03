import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.environ.get("BF16_IN", "/data/pruned")
OUT_PATH = os.environ.get("BF16_OUT", "/data/results.jsonl")

PROMPTS = [
    "Скажи одним словом: работаешь?",
    "Explain in a few sentences what a hash table is and why it is useful.",
    "What is 15 * 23?",
    "Write a Python function that reverses a string.",
]

print(f"[test] loading tokenizer from {MODEL_DIR}", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

print(f"[test] loading model from {MODEL_DIR} (bf16, device_map=auto)", flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print(f"[test] model loaded in {time.time() - t0:.1f}s", flush=True)

with open(OUT_PATH, "w") as f:
    for idx, prompt in enumerate(PROMPTS):
        for temp in (0.0, 1.0):
            print(f"[test] prompt={idx} temp={temp}", flush=True)
            messages = [{"role": "user", "content": prompt}]
            enc = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            if hasattr(enc, "input_ids"):
                input_ids = enc.input_ids.to(model.device)
            elif isinstance(enc, dict):
                input_ids = enc["input_ids"].to(model.device)
            else:
                input_ids = enc.to(model.device)
            gen_kwargs = dict(
                max_new_tokens=150,
                do_sample=temp > 0,
            )
            if temp > 0:
                gen_kwargs.update(temperature=temp, top_p=0.95, top_k=20)
            t1 = time.time()
            with torch.no_grad():
                out = model.generate(input_ids, **gen_kwargs)
            dt = time.time() - t1
            text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
            print(f"[test] prompt={idx} temp={temp} took {dt:.1f}s", flush=True)
            f.write(json.dumps({"prompt_idx": idx, "temp": temp, "text": text}) + "\n")
            f.flush()

print("[test] BF16_TEST_DONE", flush=True)
