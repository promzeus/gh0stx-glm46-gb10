---
license: mit
base_model: zai-org/GLM-4.6
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - glm4_moe
  - imatrix
  - mtp
  - speculative-decoding
  - dgx-spark
  - gb10
---

# gh0stx-glm46-gb10-GGUF

GLM-4.6 (355B-A32B MoE) quantized to run on a single NVIDIA GB10 (DGX Spark / ASUS GX10, 128 GB unified memory).
All 160 experts are kept. Routed experts are IQ2_XXS with an importance matrix, everything else is Q5_K, and the stock
MTP layer is included for speculative decoding.

- **File:** `glm46-abl-mtp-IQ2_XXS-Q5K.gguf`, 93.8 GiB, 2.26 bits per weight
- **Context on GB10:** 113k tokens with MTP, 177k without
- **Decode on GB10:** 17.1 tok/s with MTP, 11.5 tok/s without
- **Long reasoning:** no repetition loops in our 12-case loop test

Source weights: an abliterated bf16 derivative of `zai-org/GLM-4.6`. The MTP layer (`blk.92`) is the unmodified one
from `zai-org/GLM-4.6`.

sha256:

```
a6e2458fde9a7bc0cb4e3ad09d68d6fa9a6882edc5613ac9873bf8332f055f13  glm46-abl-mtp-IQ2_XXS-Q5K.gguf
```

## Run on GB10

```
llama-server -m glm46-abl-mtp-IQ2_XXS-Q5K.gguf --alias glm46 -ngl 999 -fa on \
  -ctk q4_0 -ctv q4_0 -np 1 -ub 2048 --load-mode none -fitt 11264 \
  --spec-type draft-mtp --spec-draft-n-max 1 --jinja --host 0.0.0.0 --port 8000
```

GB10 memory is shared with the OS. `-fitt 11264` leaves about 8 GiB free after load and about 5 GiB during a
112k-token request; the default 1 GiB margin made our box stop responding. Tested with llama.cpp `4ebdf2c` built with
`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121`.

## Speed on GB10

| mode | context | KV cache | decode, tok/s | MTP acceptance |
|---|---|---|---|---|
| no speculation | 32k | q8_0 | 11.45 | |
| MTP, n-max 1 | 113k | q4_0 | 17.14 | 0.80-0.94 |
| MTP, n-max 2 | 32k | q8_0 | 14.36 | 0.41-0.47 |
| MTP, n-max 3 | 32k | q8_0 | 12.17 | 0.26-0.33 |

| prompt length | prefill, tok/s | decode, tok/s |
|---|---|---|
| 57.6k | 197 | 4.10, with MTP 6.28 |
| 111.9k | 125 | 3.61 with MTP |

Decode is measured on 768-token answers (Russian, reasoning, code at temperature 0 and 1). KV type does not change
decode speed at depth (q8_0 and q4_0 both ~4.1 tok/s at 57.6k), so q4_0 doubles the context for free.

## Context on GB10

| setup | max context, tokens |
|---|---|
| MTP on, KV q4_0, `-fitt 11264` | 113,664 |
| no MTP, KV q4_0, `-fitt 8192` | 177,152 |
| no MTP, KV q8_0, `-fitt 8192` | 95,744 |

## Quality

Our loop test runs 6 prompts at temperature 0 and 1 with answers up to 4k tokens and flags repetition. This build: no
real loops in 12 cases (one false flag on a correct day-by-day enumeration). Pruned variants of the same checkpoint
(REAP, 64 of 160 experts) looped in 7-9 of 12 cases at 4.6-5.5 bits per weight.

## Recipe

- llama.cpp `4ebdf2c`: `convert_hf_to_gguf.py --outtype q8_0` with the stock MTP layer merged into the HF index.
- `llama-imatrix --no-repack -c 2048 --parse-special` on 400k tokens: reasoning traces (glaive, OpenR1-Math,
  codeforces-cots), English and Russian chat (OpenHermes-2.5 and its Russian translation) and code (Magicoder), all
  rendered through the GLM-4.6 chat template. PPL on that text with q8_0: 3.11.
- `llama-quantize`: `ffn_{gate,up,down}_exps` IQ2_XXS (267 tensors), `blk.92` experts Q4_K, attention, shared expert,
  dense layers 0-2, `token_embd`, `output` and the rest of `blk.92` Q5_K.

Pipeline, test scripts and raw results: https://github.com/promzeus/gh0stx-glm46-gb10

## Speculative decoding

The MTP head (`--spec-type draft-mtp`) loads `blk.92` only when asked; without it the layer is skipped and costs no
memory. EAGLE-3 or DFlash drafts with this target need a one-line llama.cpp patch for GLM4_MOE
(`dflash/patches/glm4-moe-layer-inp.patch` in the GitHub repo). The EAGLE-3 draft trained on GLM-4.7 gave 12.96 tok/s
here, below MTP.

## License

MIT, same as the base model.
