#!/usr/bin/env python3
"""In-job finalize for the AWS NVFP4 GLM-4.6 quant. All per-layer files sit in
OUT locally. Extract the bf16 non-layer tensors from the pruned dir (embed_tokens,
lm_head, model.norm), build model.safetensors.index.json from every safetensors
file's keys, and copy config/tokenizer from the reference model."""
import os, json, glob, shutil
from safetensors import safe_open
from safetensors.torch import save_file

IN  = os.environ.get("STREAM_IN", "/data/pruned")
OUT = os.environ.get("STREAM_OUT", "/data/out")
REF = os.environ.get("REF_DIR", "/data/ref")   # reference model dir (config+tokenizer)

# 1) non-layer bf16 tensors
wm_pruned = json.load(open(os.path.join(IN, "model.safetensors.index.json")))["weight_map"]
want = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
opened, td = {}, {}
for k in want:
    sh = wm_pruned[k]
    if sh not in opened:
        opened[sh] = safe_open(os.path.join(IN, sh), framework="pt")
    td[k] = opened[sh].get_tensor(k)
    print("nonlayer", k, td[k].dtype, list(td[k].shape), flush=True)
save_file(td, os.path.join(OUT, "model-nonlayer.safetensors"), metadata={"format": "pt"})

# 2) index.json from all output files' keys
weight_map, total = {}, 0
for path in sorted(glob.glob(os.path.join(OUT, "*.safetensors"))):
    base = os.path.basename(path)
    total += os.path.getsize(path)
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            weight_map[k] = base
idx = {"metadata": {"total_size": total}, "weight_map": weight_map}
json.dump(idx, open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=1)
print("index tensors:", len(weight_map), "files:", len(set(weight_map.values())), flush=True)

# 3) config/tokenizer from reference (all except weights/index/_COMPLETE)
for f in os.listdir(REF):
    if f.endswith(".safetensors") or f == "model.safetensors.index.json" or f == "_COMPLETE":
        continue
    shutil.copy(os.path.join(REF, f), os.path.join(OUT, f))
    print("copied", f, flush=True)
print("finalize done", flush=True)
