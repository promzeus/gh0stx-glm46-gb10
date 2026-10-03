# gguf

Полная GLM-4.6 в GGUF под один GB10: `glm46-abliterated/` (bf16 705 ГБ, 160 экспертов) плюс MTP-слой 92 из
`glm46-base/mtp.safetensors` → q8_0 → imatrix → эксперты ствола IQ2_XXS, эксперты MTP Q4_K, остальное Q5_K. Выход
`glm46-full-gguf/glm46-abl-mtp-IQ2_XXS-Q5K.gguf`: 100 720 428 928 байт (93.8 GiB, 2.26 BPW), sha256 `a6e2458f…`.

```
source infra/env.sh
kubectl config current-context                      # must be the work cluster
kubectl apply -f infra/karpenter/glm-gguf-full-nodeclass.yaml
tools/render-job.sh calib/glm46-calib-job.yaml | kubectl apply -f -      # корпус для imatrix, ~10 мин
kubectl create configmap glm-gguf-full-scripts -n "$K8S_NS" --from-file=gguf/glm-gguf-full.sh \
  --dry-run=client -o yaml | kubectl apply -f -
tools/render-job.sh gguf/glm-gguf-full-job.yaml | kubectl apply -f -
```

Нода из пула `glm-gguf-full`: любой из 12 типов с 96+ vCPU и 768+ GiB, spot с откатом на on-demand. q8_0 на 379 ГБ
во время imatrix должен целиком лежать в page cache, отсюда 768 GiB. 2026-10-03 r7i.48xlarge spot не поднимался
9 часов (`UnfulfillableCapacity`, placement score 1 из 10 во всех зонах); r8i.24xlarge spot (Xeon 6975P-C, 96 vCPU,
743 GiB, 2 NUMA) поднялся за 3 секунды по $0.709/ч, on-demand того же типа $7.09/ч. Диск gp3 1500Gi, 16000 IOPS,
1000 МБ/с. S3 через awscli v2: SA получает креды через EKS Pod Identity, s5cmd 2.3.0 на aws-sdk-go v1 их не видит.
`tokenizer_config.json` чекпоинта (transformers 5.x, `TokenizersBackend`) заменяется стоковым из `glm46-base/`, пин
конвертера иначе падает.

MTP-слой вливается в индекс HF до конвертации, конвертер пишет его как `blk.92` (nextn). 500 тензоров
`mtp.safetensors` сверены с `tensor_mapping` GLM4_MOE, все маппятся; копий эмбеддингов и головы в нём нет, они общие
со стволом. llama.cpp грузит слой только с `--spec-type draft-mtp` (`load_mtp`), иначе `TENSOR_SKIP`. llama-imatrix
MTP не исполняет, imatrix для `blk.92` пуст: эксперты слоя идут в Q4_K, остальное слоя в Q5_K, этим типам imatrix не
нужен. llama-quantize берёт первый совпавший `--tensor-type`, поэтому шаблоны `blk.92` стоят первыми.

imatrix: `llama-imatrix --no-repack` на q8_0, ctx 2048, `--parse-special`, 400k токенов корпуса `calib/` с квотами по
доменам. После spot reclaim q8_0 и итоговый imatrix берутся из S3; частичный imatrix выгружается каждые 10 минут в
`imatrix-partial.gguf`, следующий под продолжает с `--in-file` и `--chunk N`.

Прогон 2026-10-03 (UTC, из лога пода, удачная попытка):

| стадия | начало | длительность | результат |
|---|---|---|---|
| сетап (apt, awscli, llama.cpp 4ebdf2c, torch cpu, сборка quantize/imatrix), vocab pre-flight | 08:59 | 2 мин | |
| скачивание 39 шардов | 09:01 | 14 мин | 658 GiB, 680 МБ/с |
| вливание `mtp.safetensors` в индекс | 09:15 | 12 с | 500 тензоров слоя 92 |
| конвертация в q8_0 | 09:15 | 44.5 мин | 379.3 ГБ, 1759 тензоров, 23 в `blk.92`, запись 135 МБ/с |
| sha256 и выгрузка q8_0 | 10:00 | 22 мин | |
| llama-imatrix, 48 потоков, `--numa distribute` | 10:48 | 3 ч 13 мин | 62.3 с на проход 2048 токенов, RSS 351 GiB, PPL 3.1148 ± 0.016 |
| dry-run и квантование IQ2_XXS | 14:01 | 34 мин | |
| sha256 и выгрузка | 14:35 | 6 мин | `_COMPLETE` в 14:41 |

Типы по dry-run (MiB): IQ2_XXS 82 604.5 в 267 тензорах (эксперты 89 MoE-слоёв), Q5_K 11 128.7 в 654, Q4_K 2 025.0 в 3
(эксперты `blk.92`). Без imatrix квантовались 13 тензоров: 11 из `blk.92`, `output` и `token_embd`. От первого пода до
`_COMPLETE` 5 ч 42 мин с учётом двух упавших попыток, нода снята Karpenter через 7 минут после конца джобы. В S3 рядом
лежат q8_0 (379 ГБ) и imatrix: другой квант экспертов (например IQ2_XS) собирается одной стадией квантования.

Грабли этого прогона:
- `llama-imatrix` со сборкой `GGML_NATIVE=ON` на Xeon 6975P-C падал с `Illegal instruction` через 2 с после старта.
  dmesg ноды: `trap invalid opcode ... in libggml-cpu.so[768da]`; по `objdump -d` это `tileloadd` в
  `tinygemm_kernel_amx<block_q8_0>`. CPU и ядро ОС AMX поддерживают, причину внутри ggml не разбирал. Обход:
  `--no-repack` (`no_extra_bufts`), AMX-буферы не создаются. Core dump процесса с отображённым q8_0 писался 5 минут,
  скрипт ставит `ulimit -c 0`.
- Сводка dry-run искала `type = q8_0`, а llama-quantize печатает тип через `%6s` (`type =   q8_0`): grep под
  `pipefail` ронял скрипт прямо перед квантованием. Нашлось ревью до долгих стадий, исправлено.
- `have()` считал любую ошибку `aws s3 ls` отсутствием объекта, сбой S3 запустил бы скачивание и конвертацию заново.
  Теперь rc 1 значит «нет», остальные коды повторяются.
- ConfigMap монтируется атомарной подменой симлинка: идущий bash дочитывает старую копию скрипта, правка доходит
  только до следующего пода.
