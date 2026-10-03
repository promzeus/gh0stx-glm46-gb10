# serve/gx10

Бокс: ASUS GX10 / GB10, 128 ГБ unified, CUDA видит 121.69 GiB, sm_121a, `~/models/`.

vLLM-контейнер `vllm-glm46-full` (aeon-vllm:v024, пруненная NVFP4 `glm46-nvfp4-gptq-full`, порт 8000) и сама модель
удалены 2026-10-03 по решению пользователя: модель петляла в 7 из 12 кейсов. open-webui работает в host-сети и ходит
на `http://localhost:8000/v1`, новая модель для него ставится на 8000. Образ aeon-vllm:v024 на боксе оставлен.

Проверки по `docs/findings.md` до любых трат:

```
docker inspect vllm-glm46-full | grep -A40 '"Cmd"'
docker logs vllm-glm46-full 2>&1 | grep -iE 'marlin|cutlass|flashinfer|moe.*backend|nvfp4'
```

Флаги под sm_121 из гайдов по DGX Spark (CUTLASS FP4 на SM121 даёт мусор, гонка в Marlin лечится atomic add):

```
VLLM_USE_FLASHINFER_MOE_FP4=0 VLLM_NVFP4_GEMM_BACKEND=marlin VLLM_MXFP4_USE_MARLIN=1 VLLM_MARLIN_USE_ATOMIC_ADD=1
--moe-backend marlin --attention-backend TRITON_ATTN --gpu-memory-utilization 0.85
```

Известно годные образы: `nvcr.io/nvidia/vllm:26.04-py3`, `vllm/vllm-openai:cu130-nightly`.

llama.cpp для GGUF из `gguf/`. На боксе собран 2026-10-02 в `~/llama.cpp` (commit 4ebdf2c, `cmake -B build
-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121`, nvcc из `/usr/local/cuda-13.0`, сборка `llama-server` и `llama-cli`
на 20 ядрах около 10 минут). Флаги текущей версии: `-fa on|off|auto`, `--load-mode none` вместо убранного
`--no-mmap` (на unified memory без mmap нет двойного резидентства файла и CUDA-буферов), `-a glm46` даёт имя
модели для API. Контейнер `vllm-glm46-full` держит GPU, перед llama-server его `docker stop`, после теста
`docker start`.

`run_q5ks_test.sh` делает это целиком: стоп vLLM, llama-server на 8011 (8001 держит чужой `~/proxy.py`) с `-c 32768 -ngl 999 -fa on -ctk q8_0
-ctv q8_0 -np 1 --jinja`, ожидание `/health`, loop-тест `~/w4a16_serve_test.py` (top_p 0.95, таймаут запроса
поднят до 1800 с) с `W4A16_BASE=http://127.0.0.1:8011/v1`, выход `~/glm46_q5ks_test.jsonl`, лог сервера
`/tmp/llama_q5ks.log`, лог теста `/tmp/test_q5ks.log`, маркер `TESTDONE`. Запуск detached:

```
cat serve/gx10/run_q5ks_test.sh | ssht gx10 'cat > /tmp/run_q5ks_test.sh'
ssht gx10 'setsid bash /tmp/run_q5ks_test.sh > /tmp/test_q5ks.log 2>&1 < /dev/null &'
```

`run_full_test.sh` обобщает его на любую конфигурацию через env: `M` (модель), `TAG`, `CTX` (число или `auto`),
`CTK`/`CTV`, `SPEC` (например `--spec-type draft-mtp --spec-draft-n-max 3`), `PHASES` (`loop`, `bench`),
`LONG_TOKENS`. `CTX=auto` убирает `-c`: `--fit` (включён по умолчанию) подбирает наибольший контекст, который
влезает в память устройства, это и есть замер предела KV. Память у GB10 общая с ОС: с запасом `--fit` по умолчанию
(1 GiB) бокс 2026-10-03 перестал отвечать даже на ping, поэтому раннер ставит `-fitt 8192` (`FIT_MARGIN`) и держит
сторожа, который убивает llama-server при `MemAvailable` ниже 3 GiB (`MIN_AVAIL_MIB`). Бенч `tests/loop/spec_bench.py` читает
из ответа llama-server блок `timings`: скорость prefill и декода, `draft_n` и `draft_n_accepted`. Длинный кейс
берёт документацию llama.cpp с бокса (`/tmp/long_doc.txt`, 759 КБ). Выходы в `~/glm46_runs/$TAG/`.

```
ssht gx10 'M=~/models/glm46-abl-mtp-IQ2_XXS-Q5K.gguf TAG=auto-q8q4 CTX=auto CTK=q8_0 CTV=q4_0 PHASES=bench \
  setsid bash ~/run_full_test.sh > /tmp/test_auto-q8q4.log 2>&1 < /dev/null &'
```

Пулл GGUF: presigned URL на 12 ч (`aws s3 presign --expires-in 43200`) и `aria2c -c -x 8 -s 8` на боксе,
канал бокса около 16-20 МиБ/с, 104 ГБ идут полтора часа; в конце `sha256sum -c`. Диск бокса 916 ГБ был
заполнен на 98%; под Q5_K_S сносились битый пулл `~/models/glm46-pruned` (3 шарда из 16), кэш `~/nf4_cache`
(64 ГБ, 2026-09-09, маркер DONE) и неиспользуемые docker-образы.

Бюджет памяти: при utilization 0.87 под веса и KV 114 ГБ; текущий NVFP4 87.3 ГБ; вариант с attention и
shared в FP8 94 ГБ; Q5_K_S 104.2 ГБ с KV q8_0 входит, Q4_K_M 91.1 ГБ с запасом.

Длинный контекст, замер 2026-10-03 на `glm46-pruned-Q5_K_S.gguf` (97.0 GiB), `-c 65536 -ctk q8_0 -ctv q4_0`:

| метрика | значение |
|---|---|
| занято после загрузки | 112.4 GiB из 121.6, доступно 9.2 |
| остаток сверх модели, KV и базы (compute-буферы) | 2.5 GiB |
| prefill 57.6k токенов | 170 ток/с, 5.6 мин |
| декод на глубине 57.6k | 3.25 ток/с (на коротком контексте 8.5) |

FA-ядра CUDA собираются только для пар из `GGML_CUDA_FA_QUANTS`, по умолчанию `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`.
Для пары `q8_0-q4_0` сервер пишет `no FlashAttention vector kernel compiled ... converting K and V to f16 instead
(slow)`, декод на 57.6k был 2.67 ток/с. Сборка на боксе пересобрана с
`-DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;q8_0-q4_0"`, предупреждение ушло, декод 3.25 ток/с.
Расчёт по полосе даёт около 6 ток/с (23 ГБ активных весов плюс 8.8 ГБ KV на шаг при 196 ГБ/с), упор в ядро
attention на квантованном KV. Сравнение типов KV на глубине делается на новой модели.

Пулл с S3 на бокс: у gx10 нет AWS-кредов, `presign_pull.sh` печатает скрипт с presigned URL на 12 ч, `aria2c -c -x 16`
и `sha256sum -c` по `.sha256`, который сборочные джобы кладут рядом с каждым файлом. Вывод содержит временные ключи,
поэтому пишется в scratch, не в git:

```
source infra/env.sh
serve/gx10/presign_pull.sh '~/models' glm46-full-gguf/glm46-abl-mtp-IQ2_XXS-Q5K.gguf > "$SCRATCH/pull.sh"
cat "$SCRATCH/pull.sh" | ssht gx10 'cat > /tmp/pull.sh'
ssht gx10 'setsid bash /tmp/pull.sh > /tmp/pull.log 2>&1 < /dev/null &'
```

`run_campaign.sh` гоняет `run_full_test.sh` по строкам плана подряд, vLLM между прогонами не поднимает и
восстанавливает после последнего, сводку пишет в `~/glm46_runs/campaign.log`. Строка плана:
`TAG|CTX|CTK|CTV|PHASES|LONG_TOKENS|BENCH_SHORT|SPEC|EXTRA`. План для новой модели `plan_full_iq2.txt`: loop-тест и
скорость без спекуляции, MTP с n-max 1-3, EAGLE-3 и 0.6B-драфт (`dflash/README.md`), декод на глубине 50k по типам
KV, предел контекста через `--fit` с запасом 8 GiB.

```
ssht gx10 'M=~/models/glm46-abl-mtp-IQ2_XXS-Q5K.gguf setsid bash ~/run_campaign.sh ~/plan_full_iq2.txt \
  > /tmp/campaign.log 2>&1 < /dev/null &'
```

Замеры полной модели `glm46-abl-mtp-IQ2_XXS-Q5K.gguf`, кампания `plan_full_iq2.txt` 2026-10-03 (бенч
`tests/loop/spec_bench.py`: 6 запросов по 768 токенов, RU, reasoning, code при t=0 и t=1, 32k, KV q8_0):

| режим | декод, ток/с | приём драфта | занято после загрузки |
|---|---|---|---|
| без спекуляции | 11.45 | | 102.7 GiB |
| `--spec-type draft-mtp --spec-draft-n-max 1` | 17.39 | 0.80-0.93 | 105.3 GiB |
| `draft-mtp`, n-max 2 | 14.36 | 0.41-0.47 | 105.3 GiB |
| `draft-mtp`, n-max 3 | 12.17 | 0.26-0.33 | 105.3 GiB |

| `-md tw_glm47_eagle3_bf16.gguf --spec-type draft-eagle3 -ngld all` | 12.96 | 0.25-0.44 | 104.8 GiB |
| `-md GLM-4.5-DRAFT-0.6B-32k-Q4_0.gguf --spec-type draft-simple -ngld all` | 14.73 | 0.30-0.61 | 103.7 GiB |

MTP-голова GLM угадывает хорошо только первый токен: с n-max 2 и 3 общий приём падает, и скорость ниже, чем с 1.
Рабочая настройка `draft-mtp` с n-max 1, x1.52 к базе. EAGLE-3 обучен на GLM-4.7 и на нашей цели принимается хуже
MTP, 0.6B-драфт общий и тоже уступает.

Глубина 57.6k токенов (`-c 65536 -ub 2048`, длинный кейс бенча):

| KV | prefill, ток/с | декод на глубине, ток/с | занято после загрузки |
|---|---|---|---|
| q8_0 / q8_0 | 197.1 | 4.10 | 110.4 GiB |
| q4_0 / q4_0 | 199.1 | 4.13 | 104.4 GiB |
| q8_0 / q4_0 | 191.5 | 4.14 | 107.5 GiB |
| q8_0 / q8_0, `draft-mtp` n-max 1 | 196.1 | 6.28, приём 0.95 | 114.1 GiB |

Тип KV на скорость декода на глубине не влияет, упор в само ядро attention. q4_0 / q4_0 почти вдвое дешевле q8_0
по памяти. MTP на глубине даёт x1.53, как и на коротком контексте.

Предел контекста через `--fit` (`CTX=auto`, `-fitt 8192`, `-ub 2048`): 177 152 токена при KV q4_0 / q4_0 и 95 744 при
q8_0 / q8_0. После загрузки доступно 5.1-5.4 GiB, а не 8: `--fit` считает свободную память CUDA на GB10 примерно на
3 GiB оптимистичнее, чем `MemAvailable`. Сторож (3 GiB) не сработал.

Рабочая конфигурация, проверена 2026-10-03 (`tests/loop/prod_q4q4_mtp1/`):

```
~/llama.cpp/build/bin/llama-server -m ~/models/glm46-abl-mtp-IQ2_XXS-Q5K.gguf --alias glm46 -ngl 999 -fa on \
  -ctk q4_0 -ctv q4_0 -np 1 -ub 2048 --load-mode none -fitt 11264 \
  --spec-type draft-mtp --spec-draft-n-max 1 --jinja --host 0.0.0.0 --port 8011
```

| метрика | значение |
|---|---|
| контекст, выбранный `--fit` | 113 664 токена |
| доступно после загрузки | 8.0 GiB |
| доступно после запроса на 111 931 токен | 4.9 GiB |
| loop-тест | 1/12, ложный флаг на переборе дней, ответ Day 8 верный |
| декод, 768 токенов | 17.14 ток/с, приём MTP 0.80-0.94 |
| prefill 111 931 токена | 125.4 ток/с, 15 мин |
| декод на глубине 111 931 | 3.61 ток/с, приём 0.95 |

Во время длинного запроса занятость растёт на 3.2 GiB сверх загрузки. С явным `-c 102400` вместо `--fit` под той же
нагрузкой должно оставаться около 6 GiB (расчёт по 0.101 GiB KV на 1k токенов при q4_0, не мерил).
