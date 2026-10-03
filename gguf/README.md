# gguf

Full GLM-4.6 in GGUF for a single GB10: `glm46-abliterated/` (bf16, 705 GB, 160 experts) plus MTP layer 92 from
`glm46-base/mtp.safetensors` → q8_0 → imatrix → trunk experts IQ2_XXS, MTP experts Q4_K, everything else Q5_K. Output
`glm46-full-gguf/glm46-abl-mtp-IQ2_XXS-Q5K.gguf`: 100,720,428,928 bytes (93.8 GiB, 2.26 BPW), sha256 `a6e2458f…`.

```
source infra/env.sh
kubectl config current-context                      # must be the work cluster
kubectl apply -f infra/karpenter/glm-gguf-full-nodeclass.yaml
tools/render-job.sh calib/glm46-calib-job.yaml | kubectl apply -f -      # imatrix corpus, ~10 min
kubectl create configmap glm-gguf-full-scripts -n "$K8S_NS" --from-file=gguf/glm-gguf-full.sh \
  --dry-run=client -o yaml | kubectl apply -f -
tools/render-job.sh gguf/glm-gguf-full-job.yaml | kubectl apply -f -
```

The node comes from the `glm-gguf-full` pool: any of 12 types with 96+ vCPU and 768+ GiB, spot with on-demand fallback.
The 379 GB q8_0 has to stay in page cache during imatrix, hence 768 GiB. On 2026-10-03 r7i.48xlarge spot did not launch
for 9 hours (`UnfulfillableCapacity`, placement score 1 of 10 in every zone); r8i.24xlarge spot (Xeon 6975P-C, 96 vCPU,
743 GiB, 2 NUMA nodes) launched in 3 seconds at $0.709/h, on-demand of the same type is $7.09/h. Disk gp3 1500Gi, 16000
IOPS, 1000 MB/s. S3 goes through awscli v2: the SA gets credentials via EKS Pod Identity, which s5cmd 2.3.0 (aws-sdk-go
v1) does not see. The checkpoint's `tokenizer_config.json` (transformers 5.x, `TokenizersBackend`) is replaced with the
stock one from `glm46-base/`, otherwise the converter's pinned transformers fails.

The MTP layer is merged into the HF index before conversion, and the converter writes it as `blk.92` (nextn). All 500
tensors of `mtp.safetensors` were checked against the GLM4_MOE `tensor_mapping` and map; it has no copies of the
embeddings or the head, those are shared with the trunk. llama.cpp loads the layer only with `--spec-type draft-mtp`
(`load_mtp`), otherwise `TENSOR_SKIP`. llama-imatrix does not run MTP, so the imatrix has nothing for `blk.92`: its
experts go to Q4_K and the rest of the layer to Q5_K, types that need no imatrix. llama-quantize takes the first
matching `--tensor-type`, so the `blk.92` patterns come first.

imatrix: `llama-imatrix --no-repack` on q8_0, ctx 2048, `--parse-special`, 400k tokens of the `calib/` corpus with
per-domain quotas. After a spot reclaim the q8_0 and the final imatrix are taken from S3; a partial imatrix is uploaded
every 10 minutes to `imatrix-partial.gguf`, and the next pod continues with `--in-file` and `--chunk N`.

Run of 2026-10-03 (UTC, from the pod log, successful attempt):

| stage | start | duration | result |
|---|---|---|---|
| setup (apt, awscli, llama.cpp 4ebdf2c, cpu torch, build quantize/imatrix), vocab pre-flight | 08:59 | 2 min | |
| download of 39 shards | 09:01 | 14 min | 658 GiB, 680 MB/s |
| merge of `mtp.safetensors` into the index | 09:15 | 12 s | 500 tensors of layer 92 |
| conversion to q8_0 | 09:15 | 44.5 min | 379.3 GB, 1759 tensors, 23 in `blk.92`, write 135 MB/s |
| sha256 and upload of q8_0 | 10:00 | 22 min | |
| llama-imatrix, 48 threads, `--numa distribute` | 10:48 | 3 h 13 min | 62.3 s per 2048-token pass, RSS 351 GiB, PPL 3.1148 ± 0.016 |
| dry-run and IQ2_XXS quantization | 14:01 | 34 min | |
| sha256 and upload | 14:35 | 6 min | `_COMPLETE` at 14:41 |

Types from the dry-run (MiB): IQ2_XXS 82,604.5 in 267 tensors (experts of 89 MoE layers), Q5_K 11,128.7 in 654, Q4_K
2,025.0 in 3 (`blk.92` experts). 13 tensors were quantized without imatrix: 11 from `blk.92`, `output` and
`token_embd`. From the first pod to `_COMPLETE` took 5 h 42 min including two failed attempts; Karpenter removed the
node 7 minutes after the job ended. The q8_0 (379 GB) and the imatrix stay in S3, so another expert type (IQ2_XS, for
example) is one quantization stage away.

Pitfalls of this run:
- `llama-imatrix` built with `GGML_NATIVE=ON` on Xeon 6975P-C died with `Illegal instruction` 2 s after start. Node
  dmesg: `trap invalid opcode ... in libggml-cpu.so[768da]`; `objdump -d` shows `tileloadd` in
  `tinygemm_kernel_amx<block_q8_0>`. The CPU and the kernel support AMX, the cause inside ggml was not investigated.
  Workaround: `--no-repack` (`no_extra_bufts`), no AMX buffers are created. The core dump of a process with the q8_0
  mapped took 5 minutes to write, so the script sets `ulimit -c 0`.
- The dry-run summary grepped for `type = q8_0`, while llama-quantize prints the type with `%6s` (`type =   q8_0`):
  grep under `pipefail` killed the script right before quantization. Found by a review before the long stages, fixed.
- `have()` treated any `aws s3 ls` error as a missing object, so an S3 hiccup would have restarted download and
  conversion. Now rc 1 means missing and other codes are retried.
- A ConfigMap is mounted through an atomic symlink swap: a running bash keeps reading its old copy of the script, a fix
  reaches only the next pod.
