# calib

Calibration corpus for the imatrix of the full model (`gguf/`): `corpus/calib_mix_1200x4k.jsonl` in S3. Every row is
one dialogue rendered through the GLM-4.6 `chat_template.jinja` (`{"domain", "text"}`), up to 4096 tokens. Reasoning
rows must contain `</think>` and have at least 768 tokens, plain rows at least 128.

| domain | source | rows | tokens |
|---|---|---|---|
| reasoning | glaiveai/reasoning-v1-20m (prompt/response) | 240 | 392,535 |
| math | open-r1/OpenR1-Math-220k (problem/generations[0]) | 120 | 423,895 |
| code_cot | open-r1/codeforces-cots, `solutions_py` | 120 | 474,481 |
| ru | d0rj/OpenHermes-2.5-ru (sharegpt) | 360 | 172,159 |
| general | teknium/OpenHermes-2.5 (sharegpt, `refs/convert/parquet`) | 240 | 103,163 |
| code | ise-uiuc/Magicoder-OSS-Instruct-75K | 120 | 61,579 |

1200 rows, 1.63M tokens in total. The GGUF build takes 400k tokens from it with per-domain quotas (reasoning 25%,
math 15%, code_cot 15%, ru 20%, general 15%, code 10%), set in `gguf/glm-gguf-full-job.yaml`.

```
source infra/env.sh
aws s3 cp calib/calib_mix_jsonl.py "$S3_BUCKET/glm46/calib_mix_jsonl.py"
tools/render-job.sh calib/glm46-calib-job.yaml | kubectl apply -f -
```

The job takes about 10 minutes on any small node with a 12Gi memory limit. It failed on 2026-10-03 on these:
`IlyaGusev/ru_turbo_alpaca` ships a loading script that `datasets` no longer runs; `teknium/OpenHermes-2.5` is one
1.9 GB JSON file that streaming reads whole and gets OOMKilled at 12Gi, so it is read from `refs/convert/parquet`;
`apply_chat_template` needs `jinja2`.
