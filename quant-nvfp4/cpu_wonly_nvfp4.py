#!/usr/bin/env python3
"""CPU-only weight-only NVFP4 (W4A16) quant for GLM-4.6 pruned.

The W4A4 checkpoint degenerates in long reasoning (4-bit ACTIVATIONS lose the
intermediate-state precision reasoning needs; bf16 is clean). This drops
activation quant: weights stay NVFP4 (FP4 the GB10 serves), activations bf16.
Weight-only round-to-nearest needs NO calibration data and NO GPU -> runs on a
cheap CPU spot node, reusing llm-compressor's correct fused-MoE (3D expert)
packing instead of hand-rolling the pack format.

Env: WONLY_IN (bf16 in), WONLY_OUT (compressed out).
"""
from __future__ import annotations

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modeling.moe.linearize import load_quantizable_moe

IN = os.environ.get("WONLY_IN", "/data/pruned")
OUT = os.environ.get("WONLY_OUT", "/data/nvfp4a16")

# Same keep-in-bf16 set as the W4A4 recipe: head, norms, embeddings, MoE router
# gate, GLM MTP. Everything else (routed experts, shared_experts, full-attn
# q/k/v/o, dense mlp) -> NVFP4 weights.
IGNORE = [
    "re:.*lm_head",
    "re:.*norm.*",
    "re:.*embed_tokens$",
    "re:.*mlp\\.gate$",
    "re:.*shared_expert_gate$",
    "re:.*linear_attn.*",
    "re:.*eh_proj",
    "re:.*\\.mtp\\..*",
]


def log(m):
    print(f"[cpu_wonly] {m}", flush=True)


def main() -> int:
    log(f"in={IN} out={OUT}")
    import transformers.modeling_utils as _mu
    _mu.caching_allocator_warmup = lambda *a, **k: None

    log("loading bf16 CPU-resident (MoE linearized) ...")
    with load_quantizable_moe(AutoModelForCausalLM):
        model = AutoModelForCausalLM.from_pretrained(
            IN,
            dtype=torch.bfloat16,
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(IN, trust_remote_code=True)

    # NVFP4A16: 4-bit NVFP4 weights, NO input/activation quant -> no calibration.
    recipe = QuantizationModifier(targets="Linear", scheme="NVFP4A16", ignore=IGNORE)

    log("running weight-only oneshot (no dataset) ...")
    oneshot(model=model, recipe=recipe)

    log(f"saving compressed checkpoint to {OUT} ...")
    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT, save_compressed=True)
    tokenizer.save_pretrained(OUT)
    log("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
