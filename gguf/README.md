# gguf

`glm46-pruned/` (bf16, per-expert имена) → GGUF q8_0 → Q5_K_S и Q4_K_M → `glm46-pruned-gguf/`.
Цель: прогнать прунутую модель на gx10 другим движком и форматом около 5 бит, до NVFP4.

Запуск:

```
source infra/env.sh                        # git-ignored: bucket, region, namespace, SA
kubectl config current-context             # must be the work cluster
kubectl apply -f infra/karpenter/glm-gguf-nodeclass.yaml -f infra/karpenter/glm-gguf-nodepool.yaml
kubectl create configmap glm-gguf-scripts -n "$K8S_NS" --from-file=gguf/glm-gguf-convert.sh \
  --dry-run=client -o yaml | kubectl apply -f -
tools/render-job.sh gguf/glm-gguf-job.yaml | kubectl apply -f -
kubectl logs -n "$K8S_NS" -l job=glm-gguf-convert -f
```

Нода m7i.4xlarge on-demand ($0.8568/ч по прайсу региона), gp3 600Gi: пик на диске 302 ГБ входа
плюс 160 ГБ q8_0, 600Gi держит nodefs выше порога eviction kubelet в 10%. Образ `python:3.12-bookworm`,
llama.cpp клонируется в джобе, `llama-quantize` собирается там же. S3 через awscli v2: SA получает
креды через EKS Pod Identity, s5cmd 2.3.0 на aws-sdk-go v1 их не видит.

Особенности чекпоинта: config объявляет `num_nextn_predict_layers: 1`, тензоров слоя 92 нет, поэтому
`--no-mtp`. `tokenizer_config.json` от transformers 5.17 (`TokenizersBackend`) заменяется стоковым из
префикса стока, пин конвертера transformers 4.57.6 иначе падает. K-кванты делаются из q8_0 через
`llama-quantize --allow-requantize`.

На gx10: забрать `glm46-pruned-Q5_K_S.gguf` (104.2 ГБ) той же схемой, что NVFP4 (presigned URL + aria2c -c
или локальный цикл с кредами), сверить `.sha256`, поднять `llama-server`, прогнать loop-тест.
Прогон 2026-10-02, llama.cpp 926862e, 2 ч 47 м от старта пода до `_COMPLETE`: download 282 ГБ 27 мин, convert в q8_0
25 мин, каждый K-квант 25 мин, sha256 плюс upload 18 мин на файл, пик диска 443 ГБ.

| файл | байт | BPW |
|---|---|---|
| `glm46-pruned-q8_0.gguf` | 160 726 140 128 | 8.50 |
| `glm46-pruned-Q5_K_S.gguf` | 104 150 058 208 | 5.51 |
| `glm46-pruned-Q4_K_M.gguf` | 91 131 761 888 | 4.82 |

Q6_K по таблице mradermacher для REAP-218B в пересчёте на 151B ≈124 ГБ, не входит. KV на 32k: f16 12.3 ГБ, q8_0 6.2 ГБ.

## Полная модель, эксперты IQ2_XXS

`glm46-abliterated/` (bf16 705 ГБ, 160 экспертов) плюс MTP-слой 92 из `glm46-base/mtp.safetensors` → q8_0 → imatrix →
эксперты ствола IQ2_XXS, эксперты MTP Q4_K, остальное Q5_K → `glm46-full-gguf/`. Пруненная модель петляет
(`tests/README.md`), полная на 2 битах экспертов встаёт в тот же объём на gx10.

```
source infra/env.sh
kubectl apply -f infra/karpenter/glm-gguf-full-nodeclass.yaml
tools/render-job.sh prune/glm46-calib-job.yaml | kubectl apply -f -          # корпус для imatrix, ~10 мин
kubectl create configmap glm-gguf-full-scripts -n "$K8S_NS" --from-file=gguf/glm-gguf-full.sh \
  --dry-run=client -o yaml | kubectl apply -f -
tools/render-job.sh gguf/glm-gguf-full-job.yaml | kubectl apply -f -
```

Нода из пула `glm-gguf-full`: любой из 12 типов с 96+ vCPU и 768+ GiB, spot с откатом на on-demand. q8_0 на 375 ГБ
во время imatrix должен целиком лежать в page cache, отсюда 768 GiB. 2026-10-03 r7i.48xlarge spot не поднимался
9 часов (`UnfulfillableCapacity`, placement score 1 из 10 во всех зонах). r8i.24xlarge spot (Xeon 6975P-C, 96 vCPU,
743 GiB, 2 NUMA) поднялся за 3 секунды по $0.709/ч, on-demand того же типа $7.09/ч. Диск gp3 1500Gi, 16000 IOPS,
1000 МБ/с; шарды качаются на 680 МБ/с.

MTP-слой вливается в индекс HF до конвертации, конвертер пишет его как `blk.92` (nextn). 500 тензоров
`mtp.safetensors` сверены с `tensor_mapping` GLM4_MOE, все маппятся; копий эмбеддингов и головы в нём нет, они общие
со стволом. llama.cpp грузит слой только с `--spec-type draft-mtp` (`load_mtp`), иначе `TENSOR_SKIP`, памяти он
тогда не занимает. llama-imatrix MTP не исполняет, imatrix для `blk.92` пуст; эксперты слоя идут в Q4_K, остальное
слоя в Q5_K, этим типам imatrix не нужен. llama-quantize берёт первый совпавший `--tensor-type`, поэтому шаблоны
`blk.92` стоят первыми.

imatrix: `llama-imatrix` на q8_0, ctx 2048, `--parse-special`, 400k токенов `corpus/calib_mix_1200x4k.jsonl` с
квотами по доменам: reasoning 25%, math 15%, code_cot 15%, ru 20%, general 15%, code 10%. Корпус собирает
`prune/calib_mix_jsonl.py` (1200 строк, 1.63M токенов). После spot reclaim q8_0 и итоговый imatrix берутся из S3;
частичный imatrix выгружается каждые 10 минут в `imatrix-partial.gguf`, следующий под продолжает с `--in-file` и
`--chunk N`.

Прогон 2026-10-03 на r8i.24xlarge spot, по стадиям (UTC, из лога пода):

| стадия | начало | длительность | результат |
|---|---|---|---|
| сетап (apt, awscli, llama.cpp 4ebdf2c, torch cpu, сборка quantize/imatrix), vocab pre-flight | 08:59 | 2 мин | |
| скачивание 39 шардов | 09:01 | 14 мин | 658 GiB, 680 МБ/с |
| вливание `mtp.safetensors` в индекс | 09:15 | 12 с | 500 тензоров слоя 92, индекс 44689 |
| конвертация в q8_0 | 09:15 | 44.5 мин | 379.3 ГБ, 1759 тензоров, 23 в `blk.92`, `nextn_predict_layers` 1, запись 135 МБ/с |
| sha256 q8_0 | 10:00 | 18 мин | |
| выгрузка q8_0 в S3 | 10:18 | 4 мин | |
| текст imatrix | 10:22 | 5 с | 471 строка, 404 720 токенов, 197 чанков по 2048, все квоты доменов заполнены |
| llama-imatrix, 48 потоков, `--numa distribute` | 10:22 | 2 с | упал: SIGILL, см. ниже |
| попытка 3: сетап и скачивание q8_0 из S3 | 10:38 | 10 мин | 379 ГБ за 8.7 мин |
| llama-imatrix с `--no-repack`, 48 потоков, `--numa distribute` | 10:48 | ETA 3 ч 25 мин | 62.3 с на проход 2048 токенов (32.9 ток/с), PPL первого чанка 3.958, RSS 351 GiB |

Падение imatrix 2026-10-03 10:22: `Illegal instruction (core dumped)` через 2 с после старта первого чанка;
лог пода показал 10:27, потому что 5 минут писался core dump процесса с отображённым q8_0. dmesg ноды:
`traps: llama-imatrix trap invalid opcode ... in libggml-cpu.so[768da]`. По дизассемблеру `objdump -d` это
`tileloadd (%rcx,%rbx,1),%tmm0` в `tinygemm_kernel_amx<block_q8_0>`. Сборка с `GGML_NATIVE=ON` на Xeon 6975P-C
включает AMX_INT8, CPU и ядро ОС AMX поддерживают (`amx_tile amx_int8` в cpuinfo, xstate 0x602e7 с битами
XTILECFG/XTILEDATA), но ядро ggml на первом `tileloadd` получает #UD. Причину внутри ggml не разбирал. Обход:
`--no-repack` (`no_extra_bufts`) у llama-imatrix, AMX-буферы не создаются, q8_0 идёт обычным путём AVX-512.
Скрипт ставит `ulimit -c 0`. Вторая попытка джобы стартовала со старой копией скрипта, её под удалён вручную.

Ревью скрипта до долгих стадий (воркфлоу: 4 ревьюера по направлениям, проверка каждой находки на исходниках
llama.cpp 4ebdf2c на gx10) нашло два дефекта, оба исправлены до квантования. Сводка dry-run искала `type = q8_0`,
llama-quantize печатает тип через `%6s` (`type =   q8_0`), grep под `pipefail` ронял скрипт перед квантованием.
`have()` считал любую ошибку `aws s3 ls` отсутствием объекта; теперь rc 1 значит «нет», остальное повторяется.

Оценка размера по числу параметров: эксперты ствола 336B × 2.06 бит = 86.6 ГБ, остальное ствола 16.8B × 5.5 бит =
11.6 ГБ, MTP 2.3 ГБ. Файл около 100 ГБ, в памяти без MTP около 98 ГБ (91.4 GiB).
