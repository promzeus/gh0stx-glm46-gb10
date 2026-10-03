# serve/gx10

Box: ASUS GX10 / GB10, CUDA sees 121.6 GiB of memory shared with the OS, sm_121. Model
`~/models/glm46-abl-mtp-IQ2_XXS-Q5K.gguf` (built in `gguf/`), server llama.cpp. open-webui on the box runs in the host
network and calls `http://localhost:8000/v1`.

## Working configuration

Verified on 2026-10-03, raw data in `tests/loop/prod_q4q4_mtp1/`:

```
~/llama.cpp/build/bin/llama-server -m ~/models/glm46-abl-mtp-IQ2_XXS-Q5K.gguf --alias glm46 -ngl 999 -fa on \
  -ctk q4_0 -ctv q4_0 -np 1 -ub 2048 --load-mode none -fitt 11264 \
  --spec-type draft-mtp --spec-draft-n-max 1 --jinja --host 0.0.0.0 --port 8000
```

| metric | value |
|---|---|
| context chosen by `--fit` | 113,664 tokens |
| free after load | 8.0 GiB |
| free after a 111,931-token request | 4.9 GiB |
| loop test | 1/12, false flag on a day-by-day enumeration, answer Day 8 correct |
| decode, 768 tokens | 17.14 tok/s, MTP acceptance 0.80-0.94 |
| prefill of 111,931 tokens | 125.4 tok/s, 15 min |
| decode at 111,931 depth | 3.61 tok/s, acceptance 0.95 |

With an explicit `-c 102400` instead of `--fit`, about 6 GiB should stay free under the same load (estimate from 0.101
GiB of KV per 1k tokens at q4_0, not measured).

## llama.cpp on the box

`~/llama.cpp` at 4ebdf2c with `dflash/patches/glm4-moe-layer-inp.patch` (needed for `draft-eagle3` and `draft-dflash`
with a GLM4_MOE target). Build, nvcc from `/usr/local/cuda-13.0`, about 10 minutes on 20 cores:

```
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;q8_0-q4_0"
cmake --build build --config Release --target llama-server llama-cli -j 16
```

CUDA FA kernels are compiled only for the KV type pairs in `GGML_CUDA_FA_QUANTS`. Without the `q8_0-q4_0` pair the
server logs `no FlashAttention vector kernel compiled ... converting K and V to f16 instead (slow)`, and decode at
57.6k depth was 2.67 tok/s instead of 3.25 (measured on the pruned model). Flags of this version: `-fa on|off|auto`,
`--load-mode none` replaces the removed `--no-mmap`, `-a` sets the model name for the API.

## Memory

GB10 memory is shared with the OS. A run without `-c`, where `--fit` kept its default 1 GiB margin, drove the box into
`NVRM: Out of memory` on 2026-10-03: the box stopped answering even ping and needed a reboot. `--fit` reads free CUDA
memory about 3 GiB more optimistically than `MemAvailable`, and a long request adds another 3.2 GiB over the load. Hence
`-fitt 11264` in the working configuration. `run_full_test.sh` runs a watchdog that kills llama-server when
`MemAvailable` drops below 3 GiB.

## Measurements

Campaign `plan_full_iq2.txt` on 2026-10-03, raw data in `tests/loop/campaign_full_iq2/`. Bench `tests/loop/spec_bench.py`:
6 requests of 768 tokens (Russian, reasoning, code at t=0 and t=1), 32k, KV q8_0.

| mode | decode, tok/s | draft acceptance | used after load |
|---|---|---|---|
| no speculation | 11.45 | | 102.7 GiB |
| `--spec-type draft-mtp --spec-draft-n-max 1` | 17.39 | 0.80-0.93 | 105.3 GiB |
| `draft-mtp`, n-max 2 | 14.36 | 0.41-0.47 | 105.3 GiB |
| `draft-mtp`, n-max 3 | 12.17 | 0.26-0.33 | 105.3 GiB |
| `-md tw_glm47_eagle3_bf16.gguf --spec-type draft-eagle3 -ngld all` | 12.96 | 0.25-0.44 | 104.8 GiB |
| `-md GLM-4.5-DRAFT-0.6B-32k-Q4_0.gguf --spec-type draft-simple -ngld all` | 14.73 | 0.30-0.61 | 103.7 GiB |

The GLM MTP head predicts only the first token well: with n-max 2 and 3 overall acceptance drops and decode is slower.
The EAGLE-3 draft was trained on GLM-4.7 and is accepted less than MTP on this target; the general 0.6B draft is also
behind (`dflash/README.md`).

Depth of 57.6k tokens (`-c 65536 -ub 2048`):

| KV | prefill, tok/s | decode at depth, tok/s | used after load |
|---|---|---|---|
| q8_0 / q8_0 | 197.1 | 4.10 | 110.4 GiB |
| q4_0 / q4_0 | 199.1 | 4.13 | 104.4 GiB |
| q8_0 / q4_0 | 191.5 | 4.14 | 107.5 GiB |
| q8_0 / q8_0, `draft-mtp` n-max 1 | 196.1 | 6.28, acceptance 0.95 | 114.1 GiB |

KV type does not change decode speed at depth, the attention kernel is the bound; q4_0 / q4_0 takes almost half the
memory of q8_0. Context limit via `--fit` (`-fitt 8192 -ub 2048`, no MTP): 177,152 tokens with KV q4_0 and 95,744 with
q8_0.

## Scripts

`run_full_test.sh` starts llama-server for one configuration, records memory and the chosen context, runs the loop test
and/or the bench and stops the server. Env: `M` (model), `TAG`, `CTX` (a number or `auto` for `--fit`), `CTK`/`CTV`,
`SPEC`, `EXTRA`, `PHASES` (`loop`, `bench`), `LONG_TOKENS`, `BENCH_SHORT`, `FIT_MARGIN`, `MIN_AVAIL_MIB`. Outputs go to
`~/glm46_runs/$TAG/`. The long bench case reads the llama.cpp docs on the box (`/tmp/long_doc.txt`).

`run_campaign.sh` runs `run_full_test.sh` over the lines of a plan file and writes a summary to
`~/glm46_runs/campaign.log`. Plan line: `TAG|CTX|CTK|CTV|PHASES|LONG_TOKENS|BENCH_SHORT|SPEC|EXTRA`; the plan for the
measurements above is `plan_full_iq2.txt`.

```
ssht gx10 'M=~/models/glm46-abl-mtp-IQ2_XXS-Q5K.gguf setsid bash ~/run_campaign.sh ~/plan_full_iq2.txt \
  > /tmp/campaign.log 2>&1 < /dev/null &'
```

`presign_pull.sh` prints a pull script from S3 to the box: gx10 has no AWS credentials, so it uses presigned URLs
valid for 12 h, `aria2c -c -x 16 --max-tries=0` and `sha256sum -c` against the `.sha256` next to every file. The
output carries temporary keys and goes to a scratch dir, not to git. The box link is 16-20 MiB/s, 94 GiB take about
1.5 hours.

```
source infra/env.sh
serve/gx10/presign_pull.sh '~/models' glm46-full-gguf/glm46-abl-mtp-IQ2_XXS-Q5K.gguf > "$SCRATCH/pull.sh"
cat "$SCRATCH/pull.sh" | ssht gx10 'cat > /tmp/pull.sh'
ssht gx10 'setsid bash /tmp/pull.sh > /tmp/pull.log 2>&1 < /dev/null &'
```

Check that a pull is alive by `/tmp/pull.pid` (`kill -0`): `pgrep -f` over ssh matches its own command line.
