# tests

`loop/loop_test.py`: long-reasoning loop test, 6 cases (short, explain, math, code, reason, longcode) × temp {0.0,
1.0}, 512-4000 tokens, sampling top_p 0.95 (llama-server adds its own top_k 40 and min_p 0.05). Repetition detector:
the same word 8 times in a row, a 30-character chunk repeated 6+ times, 4-gram uniqueness below 0.5. A step-by-step
enumeration also trips it, so read a flagged case before calling it a loop. Env: `LOOP_BASE`, `LOOP_MODEL`,
`LOOP_OUT`. Official GLM-4.6 sampling: temp 1.0, top_p 0.95, top_k 40.

`loop/spec_bench.py`: speed and draft acceptance from llama-server's `timings` block. 6 requests of 768 tokens (Russian,
reasoning, code at t=0 and t=1) and a long case built from a document of `LONG_TOKENS` tokens. Env: `BENCH_BASE`,
`BENCH_MODEL`, `BENCH_OUT`, `BENCH_MAX`, `BENCH_SHORT`, `LONG_FILE`, `LONG_TOKENS`.

| model | loop test | data |
|---|---|---|
| full `glm46-abl-mtp-IQ2_XXS-Q5K.gguf`, KV q8_0, 32k | 1/12, false flag | `loop/full_iq2_loop_results.jsonl` |
| same, KV q4_0, `draft-mtp` n-max 1, 113k | 1/12, false flag | `loop/prod_q4q4_mtp1/` |
| pruned 64/160, GGUF Q5_K_S | 9/12 | `loop/q5ks_loop_results.jsonl` |
| pruned 64/160, NVFP4 GPTQ (vLLM) | 7/12, 10/12 with top_k 20 | |
| pruned 64/160, NVFP4 RTN (vLLM) | 9/12 | |

The false flag of the full model: `reason t=0` enumerates the days ("Is 3m >= 10m? No. The snail is still in the well"
7 times), finishes on its own (`fin=stop`) and answers correctly, Day 8. All t=1.0 cases are clean; `reason` and
`longcode` finished their answers on both temperatures (`longcode` 3-4k tokens, 7-12k characters of code). The pruned
Q5_K_S looped for real: `math t=1` repeated "I want to give the answer by giving the steps" up to the limit, `reason`
"A well is a well". Speed numbers are in `serve/gx10/README.md`, raw campaign data in `loop/campaign_full_iq2/`.
