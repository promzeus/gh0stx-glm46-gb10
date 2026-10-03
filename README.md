# gh0stx-glm46-gb10

GLM-4.6 (355B-A32B MoE, 160 экспертов) на одном NVIDIA GB10 (ASUS GX10 / DGX Spark, 121.6 GiB общей памяти): полная
модель без прунинга в GGUF, эксперты IQ2_XXS с imatrix, остальное Q5_K, встроенный MTP-слой для спекуляции, serve в
llama.cpp.

| метрика | значение |
|---|---|
| файл | 93.8 GiB, 2.26 BPW |
| loop-тест длинного reasoning | без петель: 1/12, флаг ложный (`tests/README.md`) |
| декод | 11.45 ток/с, с `draft-mtp` n-max 1 17.14 |
| контекст | 113 664 токена с MTP и KV q4_0; без MTP до 177 152 |

Команда запуска и замеры в `serve/gx10/README.md`, разбор, почему не прунинг, в `docs/findings.md`, состояние и
история в `docs/context.md`.

## Каталоги

| каталог | что | где считается |
|---|---|---|
| `gguf/` | сборка полной модели: bf16 → q8_0 с MTP → imatrix → IQ2_XXS + Q5_K | k8s, r8i.24xlarge spot |
| `calib/` | калибровочный корпус для imatrix | k8s, маленькая нода |
| `serve/gx10/` | llama.cpp на боксе, рабочая конфигурация, раннер тестов, пулл с S3 | gx10 |
| `tests/` | loop-тест, бенч скорости и спекуляции, сырые результаты | gx10 |
| `dflash/` | спекуляция: правка llama.cpp для GLM4_MOE, готовые драфты, заметки к своему DFlash | gx10 |
| `docs/` | состояние, история, разбор петель | |
| `tools/` | `render-job.sh`: подстановка переменных из `infra/env.sh` в манифесты | |
| `infra/` | `env.sh` с аккаунтом и бакетом, Karpenter-объекты; в git не идёт | EKS |

Правило выбора железа: шаг влезает в gx10, он идёт на gx10. Не влезает, тогда самый дешёвый инстанс AWS, который его
вмещает. H100/H200 не используем.

## Артефакты в S3

Бакет и аккаунт заданы в `infra/env.sh`, ниже префиксы внутри бакета:

| префикс | что | размер |
|---|---|---|
| `glm46-base/` | сток zai-org/GLM-4.6 + `mtp.safetensors` | 710 ГБ bf16 |
| `glm46-abliterated/` | bf16 от апстрима, вход сборки | 705 ГБ bf16 |
| `glm46-full-gguf/` | `glm46-abl-mtp-IQ2_XXS-Q5K.gguf`, q8_0 с MTP, imatrix, логи, `_COMPLETE` | 94 + 353 GiB |
| `corpus/calib_mix_1200x4k.jsonl` | калибровочный корпус | 1200 строк, 1.63M токенов |
| `glm46/` | копии скриптов, которые джобы качают из S3 | |

Префиксы `glm46-pruned*` и `glm46-nvfp4*` остались от закрытого пути прунинга.

## Кластер

EKS с Karpenter, доступ подов к S3 через EKS Pod Identity. Имена кластера, namespace, SA, бакета и Karpenter-объекты
лежат в `infra/` и в git не попадают. Переменные, которые ждут скрипты и манифесты: `AWS_PROFILE`, `AWS_REGION`,
`AWS_ACCOUNT`, `S3_BUCKET`, `EKS_CLUSTER`, `K8S_NS`, `K8S_SA`, `KUBE_CONTEXT`. Перед любой изменяющей командой
`kubectl config current-context` должен показать рабочий кластер. Ноды одноразовые: taint на NodePool, toleration и
`karpenter.sh/do-not-disrupt: "true"` на поде, пустую ноду Karpenter снимает сам.
