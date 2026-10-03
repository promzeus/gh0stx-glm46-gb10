# gh0stx-glm46-gb10

GLM-4.6 (355B-A32B, glm4_moe) на одном GB10 (ASUS GX10 / DGX Spark, 121.6 GiB общей памяти). Рабочий путь:
полная модель в GGUF, эксперты IQ2_XXS с imatrix, остальное Q5_K, MTP-слой как `blk.92` для `--spec-type draft-mtp`,
serve в llama.cpp. Первый путь, REAP-прунинг до 64 из 160 экспертов с NVFP4 или Q5_K_S, давал петли в длинном
reasoning (7-9 из 12 кейсов loop-теста) и закрыт, разбор в `docs/findings.md`. Вход обоих путей: bf16-чекпоинт,
подготовленный апстримом. Текущий статус и история в `docs/context.md`.

## Директории по задачам

| директория | задача | где считается |
|---|---|---|
| `gguf/` | полная модель: bf16 → q8_0 → imatrix → эксперты IQ2_XXS + Q5_K; пруненный вариант Q5_K_S / Q4_K_M | r8i.24xlarge spot (полная), m7i.4xlarge (пруненная) |
| `serve/gx10/` | llama.cpp и vLLM на боксе, раннер тестов, пулл с S3, флаги под sm_121 | gx10 |
| `tests/` | loop-тест длинного reasoning, бенч скорости и спекуляции, bf16-проверка связности | gx10 или нода |
| `dflash/` | спекулятивный декод: правка llama.cpp для GLM4_MOE, готовые драфты (MTP, EAGLE-3, 0.6B), заметки к своему DFlash | gx10 |
| `prune/` | REAP expert-prune (путь закрыт), генератор калибровочного корпуса | r7i.48xlarge spot, CPU |
| `quant-nvfp4/` | NVFP4A16 GPTQ через llm-compressor (путь закрыт) | g6e.16xlarge on-demand |
| `infra/` | `env.sh` с аккаунтом и бакетом, Karpenter NodeClass/NodePool по воркладам; git-ignored | EKS |
| `docs/` | контекст и находки | |

Правило выбора железа: шаг входит в 128 ГБ gx10, он идёт на gx10. Не входит, тогда минимальный
инстанс AWS, который его вмещает. H100/H200 не используем.

## Артефакты в S3

Бакет и аккаунт заданы в `infra/env.sh` (в git не идёт), ниже префиксы внутри бакета:

| префикс | что | размер |
|---|---|---|
| `glm46-base/` | сток zai-org/GLM-4.6 + `mtp.safetensors` | 710 ГБ bf16 |
| `glm46-abliterated/` | bf16 от апстрима, вход прунинга и полной GGUF-сборки | 705 ГБ bf16 |
| `glm46-full-gguf/` | полная модель: q8_0 с MTP (379.3 ГБ), imatrix, IQ2_XXS + Q5_K, `_COMPLETE` | см. `gguf/README.md` |
| `corpus/calib_mix_1200x4k.jsonl` | калибровочный корпус: reasoning, math, code, ru, general | 1200 строк, 1.63M токенов |
| `glm46-pruned/` | после REAP, 64 эксперта, per-expert имена, без MTP | 302 ГБ bf16, 16 шардов |
| `glm46-nvfp4a16/` | RTN W4A16 без калибровки (`quant-nvfp4/cpu_wonly_nvfp4.py`) | петлит |
| `glm46-nvfp4-gptq/` | битый стрим-выход, оставлен для сравнения | не трогать |
| `glm46-nvfp4-gptq-full/` | GPTQ NVFP4A16 пруненной модели, `_COMPLETE`, петлит 7/12 | 87.3 ГБ, 5 шардов |
| `glm46-pruned-gguf/` | GGUF из `gguf/`, `_COMPLETE` | q8_0 160.7, Q5_K_S 104.2, Q4_K_M 91.1 ГБ |
| `glm46/` | копия скриптов пайплайна | |

## Кластер

EKS с Karpenter, доступ подов к S3 через EKS Pod Identity (без IRSA-аннотации). Имена кластера,
namespace, SA, бакета и Karpenter-объекты лежат в `infra/` и в git не попадают. Переменные, которые
ждут скрипты и манифесты: `AWS_PROFILE`, `AWS_REGION`, `AWS_ACCOUNT`, `S3_BUCKET`, `EKS_CLUSTER`,
`K8S_NS`, `K8S_SA`, `KUBE_CONTEXT`. Перед любой изменяющей командой `kubectl config current-context`
должен показать рабочий кластер. Квоты: on-demand G 200 vCPU, spot G 64 vCPU, standard 1152 vCPU. Ноды одноразовые:
taint на NodePool, toleration и `karpenter.sh/do-not-disrupt: "true"` на поде, пустую ноду Karpenter
убирает сам.

## Только на gx10

serve-шаблон vLLM `glm46a16_serve.sh` для удалённой пруненной NVFP4 лежит в `~/` на боксе, в репозитории его нет.
