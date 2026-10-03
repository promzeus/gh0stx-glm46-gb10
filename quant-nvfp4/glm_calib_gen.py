#!/usr/bin/env python3
"""Calibration rows for GLM-4.6 GPTQ, pre-tokenized JSONL ({"input_ids": [...]} per line) for
nvfp4_quant.py (NVFP4_CALIB_JSONL). Self-distillation from the pruned bf16 is not possible on the
quant node (282 GB CPU-resident, one GPU), so the rows come from text.

CALIB_SOURCE
  offline    bundled seeds (math, code, ops prose) wrapped as User/Assistant text, no network. This is
             what glm46-nvfp4-gptq-full was calibrated with: no <think> content.
  reasoning  public <think>-trace datasets rendered through the GLM chat template, so the rows carry
             the thinking-mode token stream (`<|assistant|>\\n<think>...</think>\\n...`):
               glaiveai/reasoning-v1-20m      prompt/response, general-domain reasoning   (share 0.5)
               open-r1/OpenR1-Math-220k       problem/generations[0], math                (share 0.25)
               open-r1/codeforces-cots        prompt/generation, config solutions_py, code (share 0.25)
             Traces are DeepSeek-R1 style, a proxy for GLM's own; the model's own traces would be
             better ("Think Before You Prune", arXiv 2511.18864) but need a serving GPU.
Env: CALIB_MODEL (tokenizer dir), CALIB_OUT, CALIB_N, CALIB_SEQ, CALIB_SOURCE, CALIB_SEED.
"""
import os, json, sys, random
from transformers import AutoTokenizer

MODEL  = os.environ.get("CALIB_MODEL", "/data/pruned")
OUT    = os.environ.get("CALIB_OUT", "/data/calib.jsonl")
N      = int(os.environ.get("CALIB_N", "64"))
SEQ    = int(os.environ.get("CALIB_SEQ", "1024"))
SOURCE = os.environ.get("CALIB_SOURCE", "offline")
SEED   = int(os.environ.get("CALIB_SEED", "42"))

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

def log(msg): print(f"[calib] {msg}", flush=True)

# ---------------------------------------------------------------- offline
SEEDS = [
    "A snail climbs a 10 m well, +3 m by day and -2 m by night. Which day does it reach the top? Reason step by step.",
    "What is 47 * 89? Show the multiplication steps.",
    "Explain how a hash table resolves collisions with chaining vs open addressing.",
    "Write a Python function that returns the nth Fibonacci number iteratively, then explain its complexity.",
    "A TCP handshake fails only on port 443 but succeeds on 80. List the ordered hypotheses to check.",
    "A service returns 504 only under load but 200 when idle. List the ordered hypotheses to check.",
    "Prove that the sum of the first n odd numbers equals n squared.",
    "Explain the difference between a race condition and a deadlock, with one concrete example each.",
    "Reverse-engineer what this does: for i in range(len(a)): a[i] ^= key[i % len(key)]. What is it and how to break it?",
    "A pod is Pending with UnfulfillableCapacity on spot. Walk through the diagnosis and the fixes in order.",
    "Compute the GCD of 1071 and 462 using the Euclidean algorithm, showing each step.",
    "Explain how NVFP4 group quantization stores a weight: group size, scale dtype, and the global scale.",
    "Write a Go token-bucket rate limiter and explain the refill math.",
    "Given an HTTP 301 redirect loop between two hosts, enumerate the causes and how to confirm each.",
    "Derive the closed form for 1 + 2 + ... + n and verify it for n = 10.",
    "What does 'ORDER BY 1--' in a URL parameter suggest, and how would you enumerate columns next?",
]

def rows_offline():
    rng = random.Random(SEED)
    rows, attempts = [], 0
    while len(rows) < N and attempts < N * 40:
        attempts += 1
        picks = rng.sample(SEEDS, k=rng.randint(3, 6))
        parts = [f"User: {p}\nAssistant: Let me reason step by step.\n{p}\n"
                 f"Step 1: identify what is asked. Step 2: work it out carefully. "
                 f"Step 3: verify the result and state the final answer.\n" for p in picks]
        ids = tok("\n".join(parts)).input_ids[:SEQ]
        if len(ids) >= 64:
            rows.append(ids)
    return rows

# ---------------------------------------------------------------- reasoning traces
def render(prompt, answer):
    """GLM template: an assistant turn whose content holds <think>...</think> renders as
    <|assistant|>\\n<think>REASONING</think>\\nANSWER (chat_template.jinja splits on </think>)."""
    enc = tok.apply_chat_template([{"role": "user", "content": prompt},
                                   {"role": "assistant", "content": answer}],
                                  tokenize=True, add_generation_prompt=False, return_dict=True)
    ids = enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "keys") else enc
    return list(ids)

def stream(name, split="train", config=None):
    from datasets import load_dataset
    kw = {"split": split, "streaming": True}
    if config: kw["name"] = config
    return load_dataset(name, **kw).shuffle(seed=SEED, buffer_size=2000)

def take(it, want, extract):
    out, bad = [], 0
    for ex in it:
        try:
            prompt, answer = extract(ex)
        except Exception:
            bad += 1; continue
        if not prompt or not answer or "</think>" not in answer:
            bad += 1; continue
        ids = render(prompt, answer)
        if len(ids) < 256:                      # too short to carry a thinking span
            bad += 1; continue
        out.append(ids[:SEQ])
        if len(out) >= want: break
    return out, bad

def rows_reasoning():
    plan = [
        ("glaiveai/reasoning-v1-20m", None, 0.50, lambda ex: (ex["prompt"], ex["response"])),
        ("open-r1/OpenR1-Math-220k", None, 0.25, lambda ex: (ex["problem"], ex["generations"][0])),
        ("open-r1/codeforces-cots", "solutions_py", 0.25, lambda ex: (ex["prompt"], ex["generation"])),
    ]
    rows = []
    for name, config, share, extract in plan:
        want = max(1, round(N * share))
        got, bad = take(stream(name, config=config), want, extract)
        full = sum(1 for r in got if len(r) == SEQ)
        log(f"{name}{'/'+config if config else ''}: {len(got)} rows (wanted {want}, skipped {bad}), {full} truncated at {SEQ}")
        rows += got
    random.Random(SEED).shuffle(rows)
    return rows[:N]

def main():
    rows = rows_reasoning() if SOURCE == "reasoning" else rows_offline()
    with open(OUT, "w") as f:
        for ids in rows:
            f.write(json.dumps({"input_ids": ids}) + "\n")
    lens = sorted(len(r) for r in rows) or [0]
    log(f"source={SOURCE} wrote {len(rows)} rows -> {OUT}; len min/median/max = {lens[0]}/{lens[len(lens)//2]}/{lens[-1]}")
    return 0 if rows else 1

if __name__ == "__main__":
    sys.exit(main())
