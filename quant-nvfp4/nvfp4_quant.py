#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nvfp4_quant.py
==============

Model-deployment pipeline, stage 2 of 2.

ARCH SUPPORT: Qwen3.5-MoE AND GLM-4.6 (glm4_moe). The DEFAULT_IGNORE regexes are
arch-agnostic: `mlp.gate$` catches the router on both (not gate_proj); `norm.*`
catches all norms; `linear_attn.*`/`shared_expert_gate$` are Qwen-only no-ops on
GLM; `eh_proj`/`mtp` keep any GLM MTP/nextn head in bf16. GLM's per-expert and
shared_experts FFN Linears ARE quantized (desired). GLM needs NO config rewrap:
llm-compressor writes architectures=[Glm4MoeForCausalLM]/model_type=glm4_moe,
which vLLM loads natively (unlike Qwen3.5, which needed fix_config_np.py).

Pipeline:
    /data/abliterated  (pre-processed bf16 HF safetensors, Qwen3.5-MoE or GLM-4.6)
        -> NVFP4 weight+activation quantization with llm-compressor (compressed-tensors)
        -> /data/nvfp4   (vLLM-loadable compressed-tensors checkpoint)

Run host: AWS p4d.24xlarge (8x A100-40GB, 1152GB CPU RAM). The BF16 model loads
across the 8 GPUs via device_map="auto"; oneshot does data-dependent NVFP4
calibration in-place, then we save the compressed checkpoint.

Stack (pinned): transformers 5.12.1, llmcompressor (>=0.9.0 for Qwen3.5 MoE +
NVFP4 MoE support), compressed-tensors, datasets.

API verified against:
  * llm-compressor Qwen3.5 NVFP4-MoE example
    https://docs.vllm.ai/projects/llm-compressor/en/latest/key-models/qwen3.5/nvfp4-moe-example/
    -> oneshot(model, recipe, dataset, max_seq_length, num_calibration_samples,
               moe_calibrate_all_experts=True, data_collator=...)
    -> QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=[...])
  * llm-compressor W4A4-FP4 example
    https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/
    -> save: model.save_pretrained(SAVE_DIR, save_compressed=True)
       (single call writes the compressed weights + quant config; oneshot is
       in-place, no output_dir is passed to oneshot)
  * LLM Compressor 0.9.0 release notes (Red Hat): Qwen3 MoE NVFP4 supported.
  * GatedDeltaNet (GDN) / bjk110 SPARK NVFP4 ignore fix: vLLM FUSES the GDN
    linear-attention projections (in_proj_qkv / in_proj_z / in_proj_a / in_proj_b
    -> a single fused in_proj_qkvz kernel) and the causal conv1d. Quantizing any
    of those breaks the fused vLLM kernel, so EVERY `linear_attn.*` projection
    MUST be in the ignore list.

What gets quantized vs kept in BF16
-----------------------------------
NVFP4 (4-bit weights + 4-bit activations, per-group size-16 local scales +
per-tensor global scales) is applied ONLY to the "safe" dense Linear layers:
  * MoE expert packed projections   mlp.experts.gate_up_proj / mlp.experts.down_proj
  * shared-expert dense Linears      mlp.shared_expert.{gate,up,down}_proj
  * FULL-attention projections       self_attn.{q,k,v,o}_proj (the ~15 full-attn layers)
Everything below is KEPT IN BF16 via the ignore list (quantizing them is unsafe
or breaks vLLM):
  * lm_head                          (output head; precision-sensitive)
  * all norms (RMSNorm etc.)         re:.*norm.*
  * embeddings                       re:.*embed_tokens$
  * MoE router / gate                re:.*mlp[.]gate$     (tiny [184,4096] selector)
  * shared-expert gate               re:.*shared_expert_gate$
  * ALL GDN linear-attention projs   re:.*linear_attn.*   (vLLM fuses these -> must skip)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.modeling.moe.linearize import load_quantizable_moe  # direct-load linearized MoE (glm4_moe has mapping) -> no post-load 2D->3D->2D OOM


# ---------------------------------------------------------------------------
# IGNORE LIST  (regexes; what to KEEP in BF16 / NOT quantize)
# ---------------------------------------------------------------------------
# Order does not matter; these are matched against module names.
#
#   re:.*lm_head            -> output projection head
#   re:.*norm.*             -> every normalization layer (input/post-attn/q/k norms, final norm)
#   re:.*embed_tokens$      -> token embedding table
#   re:.*mlp[.]gate$        -> MoE router (NOTE the escaped dot; must NOT also catch
#                              shared_expert_gate or gate_up_proj -- anchored with $
#                              and the literal ".gate")
#   re:.*shared_expert_gate$-> the per-token shared-expert gating scalar weight
#   re:.*linear_attn.*      -> ALL GatedDeltaNet projections (in_proj_qkv/in_proj_z/
#                              in_proj_a/in_proj_b/conv1d/out_proj). vLLM fuses GDN
#                              into in_proj_qkvz; quantizing ANY of these breaks the
#                              fused kernel, so the whole subtree is ignored.
#
# We intentionally do NOT ignore `mlp.experts.*` or `shared_expert.{gate,up,down}_proj`
# or full-attn `self_attn.{q,k,v,o}_proj`: those are the layers we WANT in NVFP4.
# rene98c golden recipe: quantize ONLY the routed experts (mlp.experts.*); keep
# EVERYTHING else in BF16 (full-attn q/k/v/o, shared_expert, GDN linear_attn,
# router gate, norms, embed, lm_head). Our broken model also quantized full-attn
# + shared_expert with plain NVFP4 -> extra quality loss; this avoids that.
DEFAULT_IGNORE: list[str] = [
    "re:.*lm_head",
    "re:.*norm.*",
    "re:.*embed_tokens$",
    "re:.*mlp\\.gate$",            # MoE router (Qwen + GLM); $ so it does NOT catch gate_proj
    "re:.*shared_expert_gate$",    # Qwen shared-expert gate (GLM has none -> no-op)
    "re:.*linear_attn.*",          # Qwen GDN (GLM has none -> no-op)
    "re:.*eh_proj",                # GLM MTP/nextn head projection (keep bf16 if loaded)
    "re:.*\\.mtp\\..*",            # any GLM MTP subtree (keep bf16)
]

# Mixed recipe (docs/findings.md, step 1b): routed experts in NVFP4 (scheme), every other Linear in
# an 8-bit scheme (FP8_DYNAMIC: per-channel FP8 weights, dynamic per-token FP8 activations). The
# uniform recipe put attention, shared experts and the dense MLP of layers 0-2 into NVFP4 as well
# (4.63 bits/param overall); reference recipes keep them out of FP4 (Nemotron 3 Ultra: attention BF16,
# shared FP8, routed NVFP4; LibertAI GLM-5.3-Flash-NVFP4 and nvidia/DeepSeek-R1-0528-FP4: experts
# only). Estimate for 64 experts: ~94 GB vs 87.3 GB uniform. Module names below are the linearized
# per-expert names written by load_quantizable_moe (model.layers.N.mlp.experts.M.{gate,up,down}_proj).
MIXED_EXPERT_TARGETS: list[str] = ["re:.*mlp\\.experts\\..*"]
MIXED_REST_TARGETS: list[str] = [
    "re:.*self_attn\\.(q|k|v|o)_proj$",
    "re:.*shared_experts\\.(gate|up|down)_proj$",
    "re:.*mlp\\.(gate|up|down)_proj$",          # dense MLP of layers 0-2 (first_k_dense_replace)
]


def log(msg: str) -> None:
    print(f"[nvfp4_quant] {msg}", flush=True)


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None else default


def dir_size_gb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / (1024 ** 3)


# ---------------------------------------------------------------------------
# Calibration dataset
# ---------------------------------------------------------------------------

def build_calibration_dataset(tokenizer, num_samples: int, max_seq_len: int):
    """~`num_samples` chat-templated calibration rows from ultrachat_200k.

    Mirrors the llm-compressor Qwen3.5 example, but for a TEXT (causal-LM)
    tokenizer rather than a multimodal processor: messages are plain
    {role, content:str} (not the {type:text} content-list form the multimodal
    processor expects). apply_chat_template tokenizes each row to input_ids +
    attention_mask; the single-sample data_collator passes them straight through.

    If NVFP4_CALIB_JSONL is set (self-distilled reasoning traces from the bf16
    model, gen_calib.py), read pre-tokenized input_ids from it INSTEAD of
    ultrachat. This matches calibration activations to the real <think> inference
    distribution and fixes W4A4 reasoning degeneration (QuantLRM arXiv 2602.02581).
    """
    calib_jsonl = os.environ.get("NVFP4_CALIB_JSONL")
    if calib_jsonl and os.path.exists(calib_jsonl):
        from datasets import Dataset
        log(f"loading self-distilled calibration from {calib_jsonl} (pre-tokenized) ...")
        rows = []
        with open(calib_jsonl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                ids = json.loads(line)["input_ids"][:max_seq_len]
                rows.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
                if len(rows) >= num_samples:
                    break
        log(f"self-distilled calibration rows: {len(rows)}")
        return Dataset.from_list(rows)

    log(f"loading calibration data: ultrachat_200k train_sft[:{num_samples}] ...")
    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split=f"train_sft[:{num_samples}]",
    )
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_seq_len,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
            add_generation_prompt=False,
        )

    ds = ds.map(preprocess, batched=False, remove_columns=ds.column_names)
    return ds


def data_collator(batch):
    # llm-compressor calibrates one sample at a time. Add a batch dim for 1D rows:
    # the pre-tokenized JSONL path stores flat input_ids ([seq]), but the sequential
    # pipeline's causal-mask build indexes [batch, seq] -> a 1D tensor raises
    # "too many indices for tensor of dimension 1". The ultrachat path is already 2D.
    assert len(batch) == 1, "calibration runs with batch size 1"
    out = {}
    for key, value in batch[0].items():
        t = value if torch.is_tensor(value) else torch.tensor(value)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        out[key] = t
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="NVFP4-quantize an input bf16 abliterated Qwen3.5-MoE for vLLM.")
    parser.add_argument("--model", default=os.environ.get("NVFP4_IN", "/data/abliterated"))
    parser.add_argument("--out", default=os.environ.get("NVFP4_OUT", "/data/nvfp4"))
    parser.add_argument("--num-calibration-samples", type=int,
                        default=env_int("NVFP4_SAMPLES", 512))
    parser.add_argument("--max-seq-length", type=int,
                        default=env_int("NVFP4_MAXSEQ", 2048))
    parser.add_argument("--scheme", default=os.environ.get("NVFP4_SCHEME", "NVFP4A16"))
    parser.add_argument("--modifier", default=os.environ.get("NVFP4_MODIFIER", "gptq"),
                        choices=["gptq", "rtn"])
    parser.add_argument("--recipe", default=os.environ.get("NVFP4_RECIPE", "uniform"),
                        choices=["uniform", "mixed"],
                        help="uniform: --scheme on every Linear; mixed: --scheme on experts, --rest-scheme elsewhere")
    parser.add_argument("--rest-scheme", default=os.environ.get("NVFP4_REST_SCHEME", "FP8_DYNAMIC"))
    args = parser.parse_args()

    log("=" * 70)
    log(f"input model : {args.model}")
    log(f"output      : {args.out}")
    log(f"calib samples: {args.num_calibration_samples}, max_seq_len: {args.max_seq_length}")
    log(f"recipe      : {args.recipe}  scheme: {args.scheme}  modifier: {args.modifier}"
        + (f"  rest-scheme: {args.rest_scheme}" if args.recipe == "mixed" else ""))
    log(f"ignore list : {DEFAULT_IGNORE}")
    log("=" * 70)

    # Build the recipe FIRST so a bad scheme/modifier fails in seconds, before the
    # 282GB CPU-resident load. GPTQ = Hessian-aware error-compensated rounding
    # (needs the calibration dataset); RTN = round-to-nearest (data only sets the
    # global scales). Reasoning-calibrated GPTQ NVFP4A16 is the coherence fix:
    # weight-only NVFP4 (bf16 activations) + error feedback on <think> activations.
    Modifier = GPTQModifier if args.modifier == "gptq" else QuantizationModifier
    if args.recipe == "mixed":
        # Two config groups in one modifier. compressed-tensors resolves overlapping targets by
        # specificity (name, then regex, then class), and the groups below are disjoint regexes anyway.
        from compressed_tensors.quantization import preset_name_to_scheme
        groups = {
            "experts": preset_name_to_scheme(args.scheme, MIXED_EXPERT_TARGETS),
            "rest": preset_name_to_scheme(args.rest_scheme, MIXED_REST_TARGETS),
        }
        recipe = Modifier(config_groups=groups, ignore=DEFAULT_IGNORE)
    else:
        recipe = Modifier(targets="Linear", scheme=args.scheme, ignore=DEFAULT_IGNORE)
    log(f"recipe built: {type(recipe).__name__}({args.recipe})")

    # --- load (bf16, CPU-resident; llm-compressor onloads per-layer to GPU) ---
    log("loading input model (bf16, CPU-resident; oneshot onloads per-layer) ...")
    import transformers.modeling_utils as _mu
    _mu.caching_allocator_warmup = lambda *a, **k: None   # avoid full-model single pre-alloc OOM
    # Load on CPU (NO device_map) exactly like the official Qwen3.5 NVFP4-MoE
    # example (docs.vllm.ai/.../qwen3.5/nvfp4-moe-example). With device_map="auto"
    # the GPUs get pre-filled, and llm-compressor's post-load MoE linearization
    # (packed experts -> per-expert Linears) then offloads onto the already-full
    # GPU 0 -> CUDA OOM (observed on A100-40). With the model on CPU,
    # linearization offloads to CPU and oneshot's sequential pipeline onloads ONE
    # decoder block at a time to the GPU for calibration -> fits a 40GB card.
    # load with MoE experts already linearized (2D) via the registered conversion mapping,
    # so oneshot skips the post-load 2D->3D->2D linearization that OOMs a 1152GB node on GLM-4.6.
    with load_quantizable_moe(AutoModelForCausalLM):
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,           # transformers 5.x: `dtype`, not torch_dtype
            device_map=None,                # CPU-resident; oneshot handles GPU onloading
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- calibration data ---
    ds = build_calibration_dataset(tokenizer, args.num_calibration_samples, args.max_seq_length)

    # --- quantize (in-place) ---
    log("running oneshot NVFP4 calibration (moe_calibrate_all_experts=True) ...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
        moe_calibrate_all_experts=True,     # every expert sees calibration data (sparse-MoE quality)
        data_collator=data_collator,
    )

    # --- save compressed checkpoint (compressed-tensors + quant config in one call) ---
    log(f"saving compressed NVFP4 checkpoint to {args.out} ...")
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, save_compressed=True)
    tokenizer.save_pretrained(args.out)

    try:
        log(f"final on-disk size: {dir_size_gb(args.out):.1f} GB")
    except Exception as e:  # noqa: BLE001 -- size print is best-effort
        log(f"(could not compute output size: {e})")

    log("done. Load in vLLM with: vllm serve %s --quantization compressed-tensors" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
