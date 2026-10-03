# Context

Project memory: current state, decision history, pitfalls. Recipes and measurements live in the directory READMEs.

## State as of 2026-10-03

- Model `glm46-abl-mtp-IQ2_XXS-Q5K.gguf`: full GLM-4.6 (160 experts, top-8, MTP layer `blk.92`), IQ2_XXS experts with
  imatrix, Q5_K for the rest, 93.8 GiB. In S3 under `glm46-full-gguf/` (next to the q8_0 with MTP and the imatrix), on
  gx10 in `~/models/`, on Hugging Face as `promzeus/gh0stx-glm46-gb10-GGUF`.
- Quality: loop test without loops, 1/12 with a false flag (`tests/README.md`).
- Working configuration: KV q4_0, `draft-mtp` n-max 1, 17.14 tok/s, context 113,664 (`serve/gx10/README.md`).
- gx10: llama.cpp 4ebdf2c with the glm4-moe patch and the `q8_0-q4_0` FA kernel, drafts in `~/models/glm46-drafts/`.
  The vLLM container and the pruned NVFP4 model are removed. open-webui expects an OpenAI API on `localhost:8000`, no
  service is installed there yet.
- Cluster: pool `glm-gguf-full` (12 types, spot with on-demand fallback) kept for rebuilds; pool `glm-transfer`
  (4 vCPU) for transfers; corpus `corpus/calib_mix_1200x4k.jsonl` (`calib/README.md`).
- Open: a permanent service on port 8000 (owner's decision); an own DFlash drafter is bounded by data
  (`dflash/README.md`).

## History

- 2026-10-01. REAP pruning 160 → 64 experts (frequency metric, 60%), NVFP4A16 RTN and GPTQ on vLLM: long-reasoning
  loops, 9/12 and 7/12.
- 2026-10-02. The pruned model as GGUF Q5_K_S on llama.cpp: 9/12. The loops come from pruning, not quantization
  (`docs/findings.md`).
- 2026-10-03. Full model with IQ2_XXS experts: no loops. MTP n-max 1 x1.52. Working configuration at 113k context.
  Repository published as `promzeus/gh0stx-glm46-gb10`, model as `promzeus/gh0stx-glm46-gb10-GGUF`.

Code of the closed paths was removed on 2026-10-03: `prune/` (REAP), `quant-nvfp4/` (NVFP4 GPTQ and the streaming
quantizer), the pruned GGUF build, the bf16 test of the pruned model, the Qwen-era `dflash/` scripts. The last version
is in commit `f25137e`.

## Pitfalls

- gx10, memory. llama-server without `-c` and with the default `--fit` margin (1 GiB) drove the box to
  `NVRM: Out of memory` and a reboot. On GB10 `--fit` is about 3 GiB more optimistic than `MemAvailable`; the runner
  keeps a watchdog.
- gx10, reboot. Containers with restart policy `no` do not come back after it.
- gx10, delivery. The box has no AWS credentials: presigned URLs valid for 12 h and aria2c. DNS on the box failed, so
  aria2c needs `--max-tries=0`. Check that a process is alive with a PID file: `pgrep -f` over ssh matches its own
  command line.
- GGUF build. The AMX kernel in llama-imatrix died with SIGILL, worked around with `--no-repack`; the dry-run summary
  broke on `%6s`; `have()` confused an S3 error with a missing object; a ConfigMap fix reaches only the next pod
  (`gguf/README.md`).
- Cluster. r7i.48xlarge spot had no capacity for 9 hours; a pool over 12 types got r8i.24xlarge in seconds. Size nodes
  by the real bottleneck: a 100 GB transfer runs on 4 vCPU, not on the 96-vCPU build pool.
- Corpus. A dataset with a loading script, an OOM on a 1.9 GB JSON, `jinja2` required (`calib/README.md`).
- llama.cpp. GLM4_MOE does not expose layer inputs for EAGLE-3 and DFlash (`dflash/patches/`); FA kernels exist only
  for the KV type pairs in `GGML_CUDA_FA_QUANTS`.
- Background watchers. The shell here is zsh: `set -- $out` does not split words, and a comparison silently never
  fired.
- Network of the workstation. Internet and the route to the box dropped from time to time; retry AWS and gx10 commands.
