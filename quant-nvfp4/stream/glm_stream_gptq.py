#!/usr/bin/env python3
"""Streaming per-layer GPTQ NVFP4A16 quant of pruned GLM-4.6 on a 128GB box.

llm-compressor can't run whole: from_pretrained materializes 302GB in RAM. This
quantizes ONE Glm4MoeDecoderLayer at a time with llm-compressor's PROVEN
quantizer via oneshot(pipeline="basic") — the same "частями" technique that
quantized the MTP head (training/quant/nvfp4_quant_mtp.py). No hand-rolled GPTQ.

Per layer L (streaming, one resident):
  load layer L bf16 from the local shard -> current hidden states H are L's
  input -> oneshot(GPTQModifier, scheme=NVFP4A16, pipeline="basic") on the bare
  layer, fed H (real per-layer GPTQ over L's activations) -> compress_module
  packs NVFP4 -> stash quantized tensors -> forward the fake-quant layer on H to
  get H_next (true sequential calibration) -> free layer.

Weight-only NVFP4A16: 4-bit weights, bf16 activations. Quantize routed experts,
shared_experts, dense mlp (first_k_dense layers), full-attn q/k/v/o. Keep bf16:
lm_head, norms, embed_tokens, router mlp.gate, biases.

Shards streamed JIT by the caller into STREAM_IN; the shard for layer L must be
present when L is processed. Output tensors buffered per-shard-group and written
+ pushed to S3 as layers complete.

Env: STREAM_IN, STREAM_OUT, CALIB_JSONL (pre-tokenized {"input_ids":[...]} lines),
CALIB_N, CALIB_SEQ, LAYERS (e.g. "0-2" for a validation run, default all).
"""
from __future__ import annotations
import os, sys, json, time, gc, re, types
import torch

IN   = os.environ.get("STREAM_IN", "/data/pruned")
OUT  = os.environ.get("STREAM_OUT", "/data/nvfp4-gptq")
CALIB_JSONL = os.environ.get("CALIB_JSONL", "/data/calib.jsonl")
CALIB_N   = int(os.environ.get("CALIB_N", "64"))
CALIB_SEQ = int(os.environ.get("CALIB_SEQ", "1024"))
LAYERS    = os.environ.get("LAYERS", "")   # "" = all; "0-2" = subset for validation
RESUME_HIDDEN = os.environ.get("RESUME_HIDDEN", "")  # load H_list at start_layer
SAVE_HIDDEN   = os.environ.get("SAVE_HIDDEN", "")    # save H_list after group
DEV = "cuda"

t0 = time.time()
def log(*a): print(f"[{(time.time()-t0)/60:6.1f}m]", *a, flush=True)

import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer
from transformers.models.glm4_moe.modeling_glm4_moe import (
    Glm4MoeDecoderLayer, Glm4MoeRotaryEmbedding,
)
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from compressed_tensors.compressors import compress_module

CFG = AutoConfig.from_pretrained(IN, trust_remote_code=True)
NL  = CFG.num_hidden_layers
H   = CFG.hidden_size
FKD = CFG.first_k_dense_replace

# what to quantize inside a layer (bf16 everything else via not-listed)
TARGETS = [
    r"re:^mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$",   # routed experts (post-linearize)
    r"re:^mlp\.shared_experts\.(gate_proj|up_proj|down_proj)$", # shared expert
    r"re:^mlp\.(gate_proj|up_proj|down_proj)$",                 # dense mlp (first_k_dense layers)
    r"re:^self_attn\.[qkv]_proj$",
    r"re:^self_attn\.o_proj$",
]

# ---- shard reader over the streamed local pruned dir ----
class Shards:
    def __init__(self, base):
        self.base = base
        self.wm = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"]
        self._open = {}
    def has(self, key): return key in self.wm
    def file_for(self, key): return self.wm.get(key)
    def get(self, key):
        sh = self.wm[key]
        if sh not in self._open:
            self._open[sh] = safe_open(os.path.join(self.base, sh), framework="pt")
        return self._open[sh].get_tensor(key)
    def close(self, sh):
        self._open.pop(sh, None)

def layer_shards_present(sh: Shards, li: int) -> bool:
    # every tensor of layer li must be in a file that exists on disk
    pfx = f"model.layers.{li}."
    files = set(f for k, f in sh.wm.items() if k.startswith(pfx))
    return all(os.path.exists(os.path.join(sh.base, f)) for f in files)

def load_layer_weights(sh: Shards, layer: nn.Module, li: int):
    pfx = f"model.layers.{li}."
    sd = {}
    for k in sh.wm:
        if k.startswith(pfx):
            sd[k[len(pfx):]] = sh.get(k).to(torch.bfloat16)
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    unexpected = [u for u in unexpected]
    if unexpected:
        log(f"  WARN layer {li} unexpected keys: {unexpected[:4]} ...")
    return sd.keys()

# ---- calibration: pre-tokenized reasoning traces -> embed -> running hidden ----
def load_calib_ids():
    rows = []
    with open(CALIB_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            ids = json.loads(line)["input_ids"][:CALIB_SEQ]
            if len(ids) >= 16:
                rows.append(ids)
            if len(rows) >= CALIB_N: break
    log(f"calib rows: {len(rows)} (seq<= {CALIB_SEQ})")
    return rows

def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_grad_enabled(False)
    sh = Shards(IN)
    tok = AutoTokenizer.from_pretrained(IN, trust_remote_code=True)

    layer_range = range(NL)
    if LAYERS:
        a, b = LAYERS.split("-") if "-" in LAYERS else (LAYERS, LAYERS)
        layer_range = range(int(a), int(b) + 1)
    start_layer = layer_range.start

    # Sequential calibration must survive process restarts (one process per shard
    # group; 92 layers won't fit output in RAM). RESUME_HIDDEN carries H_list = the
    # calib hidden states at the input of start_layer; SAVE_HIDDEN writes them at
    # group end. Embedding runs only for a group that starts at layer 0.
    if RESUME_HIDDEN and os.path.exists(RESUME_HIDDEN):
        blob = torch.load(RESUME_HIDDEN, map_location="cpu")
        H_list, ids_list = blob["H"], blob["ids"]
        assert blob["next_layer"] == start_layer, \
            f"resume hidden at layer {blob['next_layer']} != start {start_layer}"
        log(f"resumed {len(H_list)} hidden states at layer {start_layer}")
    else:
        assert start_layer == 0, f"start layer {start_layer} needs RESUME_HIDDEN"
        embed_w = sh.get("model.embed_tokens.weight").to(DEV, torch.bfloat16)
        calib = load_calib_ids()
        H_list, ids_list = [], []
        for ids in calib:
            t = torch.tensor(ids, device=DEV).unsqueeze(0)
            h = nn.functional.embedding(t, embed_w)      # [1, S, H]
            H_list.append(h.to("cpu", torch.bfloat16))
            ids_list.append(t.to("cpu"))
        del embed_w; torch.cuda.empty_cache()
        log(f"embedded {len(H_list)} calib samples")

    rotary = Glm4MoeRotaryEmbedding(config=CFG).to(DEV)

    for li in layer_range:
        while not layer_shards_present(sh, li):
            log(f"waiting for shard(s) of layer {li} ..."); time.sleep(20)
        lt = time.time()
        layer = Glm4MoeDecoderLayer(CFG, li).to(DEV, torch.bfloat16).eval()
        load_layer_weights(sh, layer, li)
        # oneshot pre_process() touches model.config._name_or_path and
        # model.save_pretrained (modify_save_pretrained wrapper); a bare
        # DecoderLayer has neither. Stub them (same trick as nvfp4_quant_mtp.py).
        layer.config = CFG
        if not getattr(CFG, "_name_or_path", None):
            CFG._name_or_path = IN
        # must be a BOUND method: modify_save_pretrained reads .__self__
        layer.save_pretrained = types.MethodType(lambda self, *a, **k: None, layer)

        # calib DataLoader feeding this layer real inputs (pipeline=basic -> layer(**batch))
        samples = []
        for h_cpu, ids_cpu in zip(H_list, ids_list):
            S = h_cpu.shape[1]
            pos = torch.arange(S).unsqueeze(0)
            samples.append((h_cpu, pos))
        def collate(_b): return _b[0]
        class DS(torch.utils.data.Dataset):
            def __len__(self): return len(samples)
            def __getitem__(self, i):
                h_cpu, pos = samples[i]
                h = h_cpu.to(DEV, torch.bfloat16)
                p = pos.to(DEV)
                pe = rotary(h, p)
                # no attention_mask key: llm-compressor's tensors_to_device
                # rejects None values; the layer defaults attention_mask=None.
                return {"hidden_states": h, "position_embeddings": pe,
                        "position_ids": p}
        loader = torch.utils.data.DataLoader(DS(), batch_size=1, collate_fn=collate)

        is_dense = li < FKD
        targets = [TARGETS[2], TARGETS[3], TARGETS[4]] if is_dense else \
                  [TARGETS[0], TARGETS[1], TARGETS[3], TARGETS[4]]
        recipe = GPTQModifier(targets=targets, scheme="NVFP4A16")
        # Force all-experts calibration on MoE layers. The native top-k router only
        # sends tokens to num_experts_per_tok of the n_routed_experts, so the rest
        # get no GPTQ Hessian and keep default qparams: weight_global_scale stays 1.0
        # (generate_gparam's uncalibrated fallback) -> fp8 per-group scale underflow
        # -> dead experts -> `!` garbage at serve (bare-layer oneshot(pipeline=basic)
        # does NOT apply llm-compressor's moe_calibration_context). Raise the router
        # top_k (and group selectors) to route every token to every expert during
        # calibration ONLY; restore before the sequential-calibration forward so H_next
        # uses real routing, and inference uses the saved config.num_experts_per_tok.
        restore = []
        if not is_dense:
            n_exp = int(CFG.n_routed_experts)
            if li == FKD:  # one-time diagnostic on the first MoE layer
                log(f"DIAG mlp class: {type(layer.mlp).__name__}")
                for nm, sub in layer.mlp.named_modules():
                    ints = {a: getattr(sub, a) for a in
                            ("top_k", "topk_group", "n_group", "num_experts_per_tok",
                             "num_experts", "n_routed_experts")
                            if isinstance(getattr(sub, a, None), int)}
                    if ints or nm in ("", "gate", "experts"):
                        log(f"DIAG   {nm or '<mlp>'}: {type(sub).__name__} {ints}")
            for sub in layer.mlp.modules():
                # route to every expert: top_k -> all experts
                for attr in ("top_k", "num_experts_per_tok"):
                    v = getattr(sub, attr, None)
                    if isinstance(v, int) and v > 0:
                        restore.append((sub, attr, v)); setattr(sub, attr, n_exp)
                # select every expert group (GLM groups experts; don't touch n_group,
                # just widen topk_group to n_group so no group is masked out)
                ng = getattr(sub, "n_group", None)
                tg = getattr(sub, "topk_group", None)
                if isinstance(ng, int) and ng > 0 and isinstance(tg, int) and tg > 0:
                    restore.append((sub, "topk_group", tg)); setattr(sub, "topk_group", ng)
            if li == FKD:
                log(f"DIAG all-experts patch set {len(restore)} attrs: "
                    f"{[(type(s).__name__, a, old) for s, a, old in restore]}")
        oneshot(model=layer, dataset=loader, recipe=recipe, pipeline="basic",
                moe_calibrate_all_experts=True, processor=False)
        for sub, attr, v in restore:
            setattr(sub, attr, v)

        # forward the fake-quant layer BEFORE compression to update running hidden
        # states (true sequential calibration). After oneshot finalize the modules
        # still hold .weight + quantized_forward (fake quant); compress_module packs
        # to weight_packed and deletes .weight, so the forward must run first.
        for idx in range(len(H_list)):
            h = H_list[idx].to(DEV, torch.bfloat16)
            p = torch.arange(h.shape[1], device=DEV).unsqueeze(0)
            pe = rotary(h, p)
            ho = layer(hidden_states=h, position_embeddings=pe,
                       attention_mask=None, position_ids=p)
            ho = ho[0] if isinstance(ho, tuple) else ho
            H_list[idx] = ho.to("cpu", torch.bfloat16)

        # pack targeted modules to NVFP4 (removes .weight)
        pats = [re.compile(t[3:]) for t in targets]
        for name, mod in layer.named_modules():
            if any(p.match(name) for p in pats):
                compress_module(mod)

        # write this layer to its own file + tiny key->file sidecar, then free it.
        # per-layer files stay <5GB (single presigned PUT) and upload in parallel;
        # finalize builds the index from the sidecars without the weights on disk.
        lt_tensors = {f"model.layers.{li}.{k}": v.to("cpu").contiguous()
                      for k, v in layer.state_dict().items()}
        fname = f"model-layer-{li:03d}.safetensors"
        save_file(lt_tensors, os.path.join(OUT, fname), metadata={"format": "pt"})
        with open(os.path.join(OUT, fname + ".map.json"), "w") as fh:
            json.dump({k: fname for k in lt_tensors}, fh)

        del layer, lt_tensors; gc.collect(); torch.cuda.empty_cache()
        log(f"layer {li} done in {(time.time()-lt):.0f}s -> {fname}")

    # checkpoint calib hidden states for the next group (input of layer_range.stop)
    if SAVE_HIDDEN:
        torch.save({"H": H_list, "ids": ids_list,
                    "next_layer": layer_range.stop}, SAVE_HIDDEN)
        # plain-text marker so the orchestrator can read resume point without torch
        with open(os.path.join(os.path.dirname(SAVE_HIDDEN) or ".", "next_layer"), "w") as fh:
            fh.write(str(layer_range.stop))
        log(f"saved hidden ({len(H_list)}) next_layer={layer_range.stop} -> {SAVE_HIDDEN}")

if __name__ == "__main__":
    sys.exit(main())
