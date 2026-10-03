# GLM-4.6 NVFP4A16 GPTQ — контекст и итог (2026-10-01)

## Цель
Калиброванный NVFP4A16 GPTQ прунутого GLM-4.6 (151B REAP, 92 слоя, 64 routed-эксперта,
Glm4MoeForCausalLM), чтобы убрать петли в длинном reasoning, которыми страдал RTN-квант
(glm46-nvfp4a16). Гипотеза: петли = грубость кванта весов, чинит GPTQ на калибре.

## Итог
Калибровка починила развал, но петли не сняла. Loop-тест `w4a16_serve_test.py` (6 кейсов × temp{0,1}):

| config | LOOP/12 |
|---|---|
| a16 (RTN) | 9 |
| gptq-full, top_p 0.95 | 7 |
| gptq-full, top_p + top_k20 | 10 |

Лучший — gptq-full на top_p (7/12). GPTQ обошёл RTN на math(обе temp) и code t1.0. top_k20 усилил
повтор (math и code t1.0 вернулись в LOOP). `reason` и `longcode` петляют на обеих temp во всех
конфигах, плюс языковой дрейф в китайский в `<think>`.

## Что получилось
Первый стрим-квант выдавал `!`-мусор: эксперты не калиброваны, `weight_global_scale=1.0` → fp8
per-group underflow → мёртвые эксперты. Полномодельный путь это починил: эксперты с реальным
global_scale (verify 0 дефолтных из 15595 сэмплов), модель связная.

## Что не получилось
Петли длинного reasoning не ушли. Гипотеза «петли = дефект кванта, чинит GPTQ» не подтвердилась:
лучший 7/12, длинное рассуждение валится независимо от RTN/GPTQ и от выборки. Остаток похож на урон
от REAP-прунинга самого reasoning, но ЭТО НЕ ДОКАЗАНО (см. «bf16-база»).

## bf16-база: чего не хватает для вывода
Прунутый bf16 (до кванта) гоняли, но несравнимо с loop-тестом. Джоба `tests/loop/glm-bf16test-job.yaml`,
скрипт `tests/loop/bf16_coherence_test.py` (джоба берёт копию из S3 `glm46/`), результаты `tests/loop/bf16_coherence_results.jsonl`.
- Движок: `transformers.generate()`, БЕЗ vLLM (чтобы исключить и aeon-форк), `device_map=auto`.
- Нода: 8-GPU spot (g6e.48xl/p4de/p5/p5e), cpu 48 / mem 700Gi / gpu 8.
- Выборка: обе temp (0.0, 1.0), при temp>0 top_p0.95 + top_k20.
- Кейсы: только 4 коротких (`short`, `explain`, `math 15*23`, `code reverse string`). reason и
  longcode ОТСУТСТВУЮТ — ровно те, что петляют в кванте.
- Длина: `max_new_tokens=150`. Петли вылезают на длинном `<think>` (3-4к токенов), 150 туда не доходят.
- Оценка: сохраняется сырой `text`, БЕЗ verdict/детектора петель.

Итог: чистой bf16-базы на тех же 12 кейсах с длинной генерацией НЕТ. Приписать остаточные петли
прунингу против кванта нельзя. Закрыть — прогнать `w4a16_serve_test.py` (512-4000 токенов, детектор)
на прунутом bf16 на 8-GPU ноде (gx10 122ГБ его не держит).

## Полный пайплайн
База `zai-org/GLM-4.6` (glm4_moe, 355B, ~32B активных, 92 слоя, hidden 5120, 160 эксп top-8, 1 shared,
first_k_dense_replace=3 → слои 0-2 dense, 3-91 MoE). Пре-обработка весов bf16 делается апстримом и
в этот репозиторий не входит; её выход лежит в S3 как вход прунинга.

1. **PRE (upstream)** — пре-обработка bf16 вне репозитория; выход staged в S3 как вход PRUNE.
2. **PRUNE (REAP)** — см. ниже.
3. **QUANT (NVFP4A16 GPTQ)** — см. «Квант».
4. **DEPLOY** — vLLM на gx10, `--kv-cache-dtype fp8 --max-model-len 120000`; FP4 GEMM только sm_121a.

## Кластер
EKS-кластер, namespace и SA из `infra/env.sh` (git-ignored); доступ подов к S3 через EKS Pod Identity,
Karpenter. Ноды одноразовые: taint + `karpenter.sh/do-not-disrupt` держат на прогоне, Karpenter
консолидирует пустую. Диск — через EC2NodeClass blockDeviceMappings. kubectl-контекст перед
изменяющими командами проверять (рабочий кластер из `infra/env.sh`, прод-кластеры чужие).
Квоты: on-demand G 200 vCPU (g6e.16xl=64 влезает), spot G 64 vCPU (g6e.48xl=192 нет → спот больших сух).

## Прунинг (REAP)
Скрипт `prune/pipeline_reap_glm.py`, джоба `prune/glm46-prune-job.yaml`,
nodeclass glm-prune. Нода r7i.48xlarge spot (1536ГБ CPU, DEVICE_MAP=cpu, без GPU — прун I/O+CPU),
корень 1350Gi, cpu 80 / mem 1400Gi, backoffLimit 3. Вход: staged bf16 (~705ГБ) → выход
`glm46-pruned` (~300ГБ bf16).

Метод: REAP expert-prune, `KEEP_FRAC=0.40` → 64 из 160 экспертов/блок → 353B→~151B. Метрика важности —
per-domain **счётчик routed-selection** по доменно-сбалансированному калибру (`calib_corpus.py`,
CALIB_TARGET 128, MAXLEN 1024), объединение round-robin по доменам (не выкосить редкий домен).
Эксперты у GLM в transformers 5.17 — FUSED `Glm4MoeExperts` (3D-паки `gate_up_proj [n_exp,3072,5120]`,
`down_proj [n_exp,5120,1536]`); прун = срез dim0 по keep + срез `gate.weight`/`e_score_correction_bias`
+ `experts.num_experts=n_keep` + config. shared_experts не трогаются, n_group=1/topk_group=1.
Метрика считает только частоту выбора эксперта, не влияние на reasoning — вероятный источник остаточных петель.

Грабли прунинга: `from_pretrained` открывает каждый шард из индекса — `--exclude mtp.safetensors`
ронял загрузку, синкать полный набор; `from_pretrained` mmap-ит вход, `rm` мид-прогоне не освобождает
блоки (нужен корень ≥1500Gi, вход+выход рядом).

## Квант (рабочий путь)
`quant-nvfp4/nvfp4_quant.py`: `load_quantizable_moe` линеаризует эксперты в LinearExperts2D →
`oneshot(moe_calibrate_all_experts=True)` sequential pipeline → `save_pretrained(save_compressed=True)`.
Нужен весь 282ГБ bf16 CPU-resident + 1 GPU для onload по блоку. Калибр офлайн `glm_calib_gen.py`
(256 строк, seq 2048), NVFP4A16 + GPTQ. Джоба `quant-nvfp4/glm-requant-full-job.yaml` + nodeclass
`glm-requant` (`infra/karpenter/glm-requant-nodeclass.yaml`). Нода on-demand g6e.16xlarge (1×L40S 48GB, 512ГБ RAM,
64 vCPU, ~$5.5/ч, прогон ~1ч45м). GPU на блоке ~47.7/48ГБ. In-job VERIFY expert global_scale!=1.0 до заливки.

Баг, который ловили: JSONL-калибр даёт 1D input_ids, sequential-пайплайн индексировал 1D двумя
индексами → `IndexError: too many indices for tensor of dimension 1`. Фикс — collator добавляет batch-dim.

## Квант (стрим-путь — тупик)
`quant-nvfp4/stream/glm_stream_gptq.py` квантует по одному Glm4MoeDecoderLayer через `oneshot(pipeline="basic")`
на голом слое. НЕ калибрует MoE-эксперты (attention+shared считаются, эксперты — нет) → global_scale=1.0
→ `!`-мусор. Патч роутера (top_k→n_routed_experts на калибровку) не помог: basic-пайплайн на bare-layer
не делает qparam-init экспертов. Валидатор на слоях 0-4 (shard1) `quant-nvfp4/stream/glm-stream-val-job.yaml` ловит за ~10 мин.
Стрим-джоба `quant-nvfp4/stream/glm-stream-job.yaml` + nodeclass `infra/karpenter/glm-stream-nodeclass.yaml`.

## S3 (бакет, аккаунт и профиль в `infra/env.sh`)
Префиксы внутри бакета:
- `glm46-base/` — сток + `mtp.safetensors` (nextn-голова для спекулятивного serve).
- `glm46-abliterated/` — pre-prune bf16 (upstream), вход PRUNE (705ГБ).
- `glm46-pruned/` — выход PRUNE (~300ГБ bf16, 16 шардов + index) → вход QUANT.
- `glm46-nvfp4a16/` — старый RTN-квант (эталон формата, per-expert, петлит).
- `glm46-nvfp4-gptq/` — БИТЫЙ стрим-выход (не трогать, для сравнения).
- `glm46-nvfp4-gptq-full/` — ГОДНЫЙ выход (5 шардов, 87.3ГБ, 54021 тензор, per-expert, `_COMPLETE`).
- `glm46/` — скрипты: `pipeline_reap_glm.py`, `nvfp4_quant.py`,
  `glm_stream_gptq.py`, `glm_calib_gen.py`, `finalize_job.py`, `gen_calib.py`, `bf16_coherence_test.py`,
  `w4a16_serve_test.py`; результаты `bf16_coherence_results.jsonl`.

## gx10 (бокс, GB10, 122ГБ unified, sm_121a)
- `~/models/glm46-nvfp4-gptq-full/` — скачан (87.34ГБ, байт в байт с S3).
- старый битый `~/models/glm46-nvfp4-gptq` УДАЛЁН (освобождал диск под пулл).
- serve `vllm-glm46-full` поднят (aeon-vllm:v024, 32k, порт 8000) — держит GPU; гасить
  `docker rm -f vllm-glm46-full`, если нужен gh0stx.
- тест: `~/w4a16_serve_test.py` (top_p), `~/w4a16_serve_test_topk.py` (с top_k20); результаты
  `~/glm46gptqfull_test.jsonl`, `~/glm46gptqfull_topk_test.jsonl`.
- serve-шаблон `~/glm46a16_serve.sh`. aeon-vllm:v024 БЕЗ llmcompressor (только compressed_tensors).

## Доставка S3→gx10 (грабли)
gx10 без AWS-профиля. SSO-токен ноута протухает ~1ч → креды на боксе мрут. Рабочие схемы:
- локальный цикл `pull_gx10.sh` со свежими кредами на попытку (venv `/tmp/awsenv` boto3 после
  `aws sso login --profile $AWS_PROFILE`); `aws s3 sync` идемпотентен, резюм на обрыве;
- presigned URL (12ч, системный `aws s3 presign`, ~10с/файл), aria2c `-c` на боксе.
Бокс-цикл с вечным retry НЕ годится: креды не обновляет. Диск: снести старый битый перед пуллом (No space).
Скрипты в scratchpad сессии: `pull_gx10.sh`, `gx10_sync.sh`/`gx10_launch.sh`, `gx10_pull_presigned.sh`.

## GGUF (2026-10-02)
Джоба `gguf/glm-gguf-job.yaml` на m7i.4xlarge on-demand, gp3 600Gi, llama.cpp 926862e: `glm46-pruned/` →
q8_0 → Q5_K_S и Q4_K_M → `glm46-pruned-gguf/` с `_COMPLETE`. 2 ч 47 м, без ретраев. Размеры, BPW и тайминги
шагов в `gguf/README.md`. Две правки конвертера подтвердились в прогоне: `--no-mtp` и стоковый
`tokenizer_config.json`. Пик диска 443 ГБ из 600.

## Q5_K_S на gx10 (2026-10-02, вечер)
Запрет на бокс снят пользователем. Под пулл снесены `~/models/glm46-pruned` (битый, 3/16 шардов), `~/nf4_cache`
(64 ГБ, DONE 2026-09-09), образы `aeon-vllm-ultimate` и `specforge-ready`, build cache; было 21 ГБ свободно,
стало 134. Пулл presigned + aria2c 1 ч 40 м при 16-20 МиБ/с, sha256 сошёлся. llama.cpp собран на боксе под
sm_121 (`~/llama.cpp`, 4ebdf2c). Порт 8001 занят чужим `~/proxy.py`, llama-server на 8011. Загрузка 85 с,
RAM 108 из 121 ГБ при `-c 32768 -ctk/-ctv q8_0`. Декод 8.5 ток/с (print_timing), пользователю хватает без DFlash.
Раннер `serve/gx10/run_q5ks_test.sh`, выход скопирован в `tests/loop/q5ks_loop_results.jsonl`.

Результат: 9/12 LOOP (41 мин). Чистые `short t=1`, `code t=0`, `code t=1`. Петлят оба greedy (контроль),
`explain t=1`, `math t=1`, `reason t=1`, `longcode t=1`. Вердикт: урон до кванта, прунинг. Детали и
следующий прогон в `prune/README.md`. vLLM-контейнер поднят обратно.

## Полная модель на 2 битах (2026-10-03)
Решение и обоснование в `docs/findings.md`, рецепт в `gguf/README.md`. Корпус `corpus/calib_mix_1200x4k.jsonl`
собран (1200 строк, 1.63M токенов); две грабли по дороге: `IlyaGusev/ru_turbo_alpaca` со скриптом загрузки
(`datasets` его больше не исполняет, заменён на `d0rj/OpenHermes-2.5-ru`) и OOMKilled на 12Gi при стриминге
`teknium/OpenHermes-2.5` (один JSON на 1.9 ГБ, читается из `refs/convert/parquet`). Джоба `glm-gguf-full` висела
9 часов: r7i.48xlarge spot не поднимался. Новый пул `glm-gguf-full` на 12 типов взял r8i.24xlarge spot за
$0.709/ч, под стартовал 08:59 UTC. На gx10 лежат раннер `~/run_full_test.sh` (serve, loop-тест, бенч скорости с
`timings`, восстановление vLLM) и `~/spec_bench.py`.

## gx10: зависание и длинный контекст (2026-10-03)
09:18 UTC прогон llama-server без `-c` (`--fit` с запасом 1 GiB) на 97 GiB модели забрал общую память: `NVRM: Out of
memory`, snapd по watchdog, бокс перестал отвечать, пользователь перезагрузил его в 09:29. После перезагрузки не
поднялись контейнеры с restart policy `no`: `vllm-glm46-full` (восстановлен раннером) и один чужой контейнер
(не трогал). Раннер `serve/gx10/run_full_test.sh` теперь держит запас 8 GiB и сторожа по `MemAvailable`.
Замеры длинного контекста и пересборка с FA-ядром `q8_0-q4_0` в `serve/gx10/README.md`.

Ревью скрипта сборки нашло блокер: сводка dry-run искала `type = q8_0`, а llama-quantize печатает `type =   q8_0`
(`%6s`), grep под `pipefail` ронял скрипт прямо перед квантованием. Исправлено в ConfigMap; текущий под читает
старую копию и упадёт на этой строке после выгрузки q8_0 и imatrix, повтор джобы возьмёт их из S3. `have()`
теперь отличает «нет объекта» (rc 1) от сбоя S3 и повторяет.

## Полная модель собрана (2026-10-03 14:41 UTC)
`glm46-full-gguf/glm46-abl-mtp-IQ2_XXS-Q5K.gguf`, 93.8 GiB, детали и тайминги в `gguf/README.md`. Попытка 1 упала на
SIGILL в AMX-ядре llama-imatrix, попытку 2 удалил вручную (старая копия скрипта), попытка 3 прошла с `--no-repack`.
Битый `glm46-pruned-Q5_K_S.gguf` с gx10 удалён, пулл новой модели идёт. Репозиторий опубликован как
`promzeus/gh0stx-glm46-gb10` (публичный).

## Пулл и первый запуск (2026-10-03)
Пулл через `presign_pull.sh` с 14:43 UTC шёл на 20 МиБ/с. В 15:40 aria2c вышел на 70% (65 из 93 GiB): на боксе отказал DNS
(`Name resolution ... failed: DNS server returned general failure`), дефолтных 5 попыток не хватило. Проверка живости
через `pgrep -f` находила собственную команду ssh и показывала «RUNNING», падение заметил вручную. Генератор теперь
ставит `--max-tries=0 --retry-wait=10` и PID-файл. Докачка с 65 GiB 15:52-16:17, `sha256sum -c` OK
(`a6e2458f…`). Первый запуск в кампании: сервер поднялся за 75 с, при 32k и KV q8_0 занято 102.7 GiB, доступно 18.9.

## Loop-тест полной модели (2026-10-03 16:17-16:42 UTC)
Петель нет: 1/12 по детектору, и этот флаг ложный (перебор дней в задаче про улитку, ответ Day 8 верный). Пруненные
варианты давали 9/12 и 7/12. Декод 10.9-11.4 ток/с против 8.5 у пруненной Q5_K_S. Детали в `tests/README.md`.

## Кампания на gx10 (2026-10-03 16:17-17:57 UTC)
12 прогонов, таблицы в `serve/gx10/README.md`, сырые данные в `tests/loop/campaign_full_iq2/`. Лучшая спекуляция:
встроенный MTP с n-max 1, 11.45 → 17.39 ток/с; на глубине 57.6k 4.10 → 6.28. EAGLE-3 от GLM-4.7 и 0.6B-драфт хуже.
Тип KV на скорость на глубине не влияет; предел контекста 177k при q4_0 / q4_0. Моё фоновое ожидание кампании не
срабатывало: Bash-инструмент здесь zsh, `set -- $out` не делит строку на слова, сравнение падало с
«integer expression expected». Итог прочитан вручную после окончания.

## Рабочая конфигурация (2026-10-03 18:20-19:00 UTC)
KV q4_0 / q4_0, `draft-mtp` n-max 1, `--fit` с запасом 11 GiB: контекст 113 664, loop-тест без петель, 17.14 ток/с,
на глубине 112k 3.61 ток/с. Команда и таблица в `serve/gx10/README.md`. vLLM с пруненной NVFP4 раннер поднял обратно.

## Чистка gx10 (2026-10-03 19:10 UTC)
По решению пользователя удалены контейнер `vllm-glm46-full` и `~/models/glm46-nvfp4-gptq-full` (82 ГБ). На диске бокса
121 ГБ свободно. Раннер больше не поднимает vLLM (`RESTORE_VLLM=0` по умолчанию). open-webui ждёт OpenAI API на
`localhost:8000`. GGUF полной модели проверен по метаданным: 160 экспертов, top-8, 93 блока с MTP, контекст 202 752.

## Открыто
Поставить рабочую конфигурацию постоянным сервисом на порт 8000 (решение пользователя). Свой DFlash упирается в
данные (`dflash/README.md`).
