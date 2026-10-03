# gh0stx-glm46-gb10

GLM-4.6 (355B-A32B MoE, 160 experts) on a single NVIDIA GB10 (ASUS GX10 / DGX Spark, 121.6 GiB of unified memory):
the full model without pruning in GGUF, IQ2_XXS experts with an importance matrix, Q5_K for the rest, the built-in MTP
layer for speculative decoding, served by llama.cpp.

| metric | value |
|---|---|
| file | 93.8 GiB, 2.26 BPW |
| long-reasoning loop test | no loops: 1/12, a false flag (`tests/README.md`) |
| decode | 11.45 tok/s, 17.14 with `draft-mtp` n-max 1 |
| context | 113,664 tokens with MTP and KV q4_0; up to 177,152 without MTP |

Model on Hugging Face: https://huggingface.co/promzeus/gh0stx-glm46-gb10-GGUF (card in `hf/README.md`, upload job in
`hf/hf-upload-job.yaml`). Run command, systemd service and measurements: `serve/gx10/README.md`. Why not pruning:
`docs/findings.md`. State and history: `docs/context.md`.

## Directories

| directory | contents | runs on |
|---|---|---|
| `gguf/` | full-model build: bf16 → q8_0 with MTP → imatrix → IQ2_XXS + Q5_K | k8s, r8i.24xlarge spot |
| `calib/` | calibration corpus for the imatrix | k8s, small node |
| `serve/gx10/` | llama.cpp on the box: working configuration, systemd service with a memory guard, test runner, pull from S3 | gx10 |
| `tests/` | loop test, speed and speculation bench, raw results | gx10 |
| `dflash/` | speculative decoding: llama.cpp patch for GLM4_MOE, ready-made drafts, notes for an own DFlash | gx10 |
| `hf/` | model card and the job that uploads the GGUF from S3 to Hugging Face | k8s, 4 vCPU |
| `docs/` | state, history, analysis of the loops | |
| `tools/` | `render-job.sh`: substitutes variables from `infra/env.sh` into manifests | |
| `infra/` | `env.sh` with account and bucket, Karpenter objects; not in git | EKS |

Hardware rule: if a step fits gx10, it runs on gx10. If not, the cheapest AWS instance that fits it, sized by the
step's real bottleneck. No H100/H200.

## Artifacts in S3

Bucket and account are set in `infra/env.sh`; prefixes inside the bucket:

| prefix | contents | size |
|---|---|---|
| `glm46-base/` | stock zai-org/GLM-4.6 + `mtp.safetensors` | 710 GB bf16 |
| `glm46-abliterated/` | bf16 from upstream, input of the build | 705 GB bf16 |
| `glm46-full-gguf/` | `glm46-abl-mtp-IQ2_XXS-Q5K.gguf`, q8_0 with MTP, imatrix, logs, `_COMPLETE` | 94 + 353 GiB |
| `corpus/calib_mix_1200x4k.jsonl` | calibration corpus | 1200 rows, 1.63M tokens |
| `glm46/` | copies of the scripts that jobs fetch from S3 | |

The `glm46-pruned*` and `glm46-nvfp4*` prefixes are left over from the closed pruning path.

## Cluster

EKS with Karpenter, pods reach S3 through EKS Pod Identity. Cluster, namespace, SA and bucket names and the Karpenter
objects live in `infra/` and are not in git. Variables the scripts and manifests expect: `AWS_PROFILE`, `AWS_REGION`,
`AWS_ACCOUNT`, `S3_BUCKET`, `EKS_CLUSTER`, `K8S_NS`, `K8S_SA`, `KUBE_CONTEXT`. Before any mutating command,
`kubectl config current-context` must show the work cluster. Nodes are disposable: a taint on the NodePool, a toleration
and `karpenter.sh/do-not-disrupt: "true"` on the pod, and Karpenter removes the empty node itself.
