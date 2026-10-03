# Long-reasoning loops: cause and fix

Pruned variants of GLM-4.6 (REAP, 64 of 160 experts) looped on long `<think>` with every quantization and engine. The
full model with 2-bit experts does not loop. Loop-test table in `tests/README.md`.

## Cause: pruning

The REAP paper uses the "routed-selection count" metric as a frequency baseline, and at 50% pruning it collapses on
large models: Qwen3-Coder-480B 0.010 vs REAP 0.619, Kimi-K2 0.056 vs 0.624, GLM-4.5-Air 0.308 vs 0.515 (coding,
pass rate). The paper's reason: a rarely selected expert with a large contribution to the layer output goes first. We
pruned 60% with this metric and a 128 × 1024 calibration set; Cerebras uses 12k samples × 16k tokens for models over
110B. Even full REAP at 40% drops GPQA diamond (thinking) at Cerebras from 78.8 to 69.7.

A Q5_K_S GGUF of the pruned model on llama.cpp separated quantization from pruning: 5.51 bits, nearly lossless
quantization, still gave 9/12, the same as NVFP4 RTN on vLLM. The loops did not depend on format or engine, so the
damage was done before quantization.

The recipe that worked for Qwen3.5-397B does not carry over. Qwen has 512 experts of 1024 with top-10, and its cuts
kept 174-184 of 512; GLM-4.6 has 160 of 1536 with top-8, the cut kept 64 of 160, and nothing replaces a removed
expert. After pruning, Qwen was healed with LoRA on attention (r16, 150 steps, bf16 on 8 GPUs); the pruned GLM was
never healed, H100/H200 class hardware is ruled out. Qwen was checked with a short test: thinking off, 100 greedy
tokens with repetition_penalty 1.2, long reasoning was never measured.

## Fix: the full model at 2 bits

File size is parameters × bits. 151B × 5.51 bits gave 104 GB; 353B with IQ2_XXS experts (2.06 bits) and Q5_K for the
rest give 94 GiB. Active parameters per token are the same (32B), bytes per token are fewer, KV is the same: pruning
bought neither speed nor context. Result on 2026-10-03: 0 real loops in 12 cases (one false flag on a step-by-step
enumeration with a correct answer), decode 11.45 tok/s vs 8.5 for the pruned Q5_K_S. Build in `gguf/README.md`,
measurements in `serve/gx10/README.md`.

## Sampling

Official for GLM-4.6: temp 1.0, top_p 0.95, top_k 40. top_k 20 made the pruned NVFP4 loop more (10/12 vs 7/12).

## Sources

- [REAP paper, arXiv 2510.13999](https://arxiv.org/html/2510.13999)
- [cerebras/GLM-4.6-REAP-218B-A32B-FP8, benchmarks](https://huggingface.co/cerebras/GLM-4.6-REAP-218B-A32B-FP8)
- [GLM-4.6 model card](https://huggingface.co/zai-org/GLM-4.6)
- [unsloth/GLM-4.6-GGUF, UD-IQ1/IQ2 sizes](https://huggingface.co/unsloth/GLM-4.6-GGUF)
- [llama.cpp conversion/glm.py](https://github.com/ggml-org/llama.cpp/blob/master/conversion/glm.py)
