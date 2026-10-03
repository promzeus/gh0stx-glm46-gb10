# serve/gx10

Бокс: ASUS GX10 / GB10, CUDA видит 121.6 GiB, память общая с ОС, sm_121. Модель `~/models/glm46-abl-mtp-IQ2_XXS-Q5K.gguf`
(сборка в `gguf/`), сервер llama.cpp. open-webui на боксе работает в host-сети и ходит на `http://localhost:8000/v1`.

## Рабочая конфигурация

Проверена 2026-10-03, сырые данные в `tests/loop/prod_q4q4_mtp1/`:

```
~/llama.cpp/build/bin/llama-server -m ~/models/glm46-abl-mtp-IQ2_XXS-Q5K.gguf --alias glm46 -ngl 999 -fa on \
  -ctk q4_0 -ctv q4_0 -np 1 -ub 2048 --load-mode none -fitt 11264 \
  --spec-type draft-mtp --spec-draft-n-max 1 --jinja --host 0.0.0.0 --port 8000
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

С явным `-c 102400` вместо `--fit` под той же нагрузкой должно оставаться около 6 GiB (расчёт по 0.101 GiB KV на 1k
токенов при q4_0, не мерил).

## llama.cpp на боксе

`~/llama.cpp` на 4ebdf2c с правкой `dflash/patches/glm4-moe-layer-inp.patch` (нужна для `draft-eagle3` и
`draft-dflash` с целью GLM4_MOE). Сборка, nvcc из `/usr/local/cuda-13.0`, около 10 минут на 20 ядрах:

```
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;q8_0-q4_0"
cmake --build build --config Release --target llama-server llama-cli -j 16
```

FA-ядра CUDA собираются только для пар типов KV из `GGML_CUDA_FA_QUANTS`. Без пары `q8_0-q4_0` сервер пишет
`no FlashAttention vector kernel compiled ... converting K and V to f16 instead (slow)` и на глубине 57.6k декодировал
2.67 ток/с вместо 3.25 (замер на пруненной модели). Флаги этой версии: `-fa on|off|auto`, `--load-mode none` вместо
убранного `--no-mmap`, `-a` задаёт имя модели для API.

## Память

Память у GB10 общая с ОС. Прогон без `-c`, где `--fit` оставлял запас по умолчанию 1 GiB, 2026-10-03 довёл бокс до
`NVRM: Out of memory`, бокс перестал отвечать даже на ping и потребовал перезагрузки. `--fit` считает свободную память
CUDA примерно на 3 GiB оптимистичнее, чем `MemAvailable`, а длинный запрос добавляет ещё 3.2 GiB сверх загрузки.
Отсюда `-fitt 11264` в рабочей конфигурации. `run_full_test.sh` держит сторожа, который убивает llama-server при
`MemAvailable` ниже 3 GiB.

## Замеры

Кампания `plan_full_iq2.txt` 2026-10-03, сырые данные в `tests/loop/campaign_full_iq2/`. Бенч `tests/loop/spec_bench.py`:
6 запросов по 768 токенов (RU, reasoning, code при t=0 и t=1), 32k, KV q8_0.

| режим | декод, ток/с | приём драфта | занято после загрузки |
|---|---|---|---|
| без спекуляции | 11.45 | | 102.7 GiB |
| `--spec-type draft-mtp --spec-draft-n-max 1` | 17.39 | 0.80-0.93 | 105.3 GiB |
| `draft-mtp`, n-max 2 | 14.36 | 0.41-0.47 | 105.3 GiB |
| `draft-mtp`, n-max 3 | 12.17 | 0.26-0.33 | 105.3 GiB |
| `-md tw_glm47_eagle3_bf16.gguf --spec-type draft-eagle3 -ngld all` | 12.96 | 0.25-0.44 | 104.8 GiB |
| `-md GLM-4.5-DRAFT-0.6B-32k-Q4_0.gguf --spec-type draft-simple -ngld all` | 14.73 | 0.30-0.61 | 103.7 GiB |

MTP-голова GLM угадывает хорошо только первый токен: с n-max 2 и 3 общий приём падает и скорость ниже. EAGLE-3 обучен
на GLM-4.7 и на нашей цели принимается хуже MTP, 0.6B-драфт общий и тоже уступает (`dflash/README.md`).

Глубина 57.6k токенов (`-c 65536 -ub 2048`):

| KV | prefill, ток/с | декод на глубине, ток/с | занято после загрузки |
|---|---|---|---|
| q8_0 / q8_0 | 197.1 | 4.10 | 110.4 GiB |
| q4_0 / q4_0 | 199.1 | 4.13 | 104.4 GiB |
| q8_0 / q4_0 | 191.5 | 4.14 | 107.5 GiB |
| q8_0 / q8_0, `draft-mtp` n-max 1 | 196.1 | 6.28, приём 0.95 | 114.1 GiB |

Тип KV на скорость декода на глубине не влияет, упор в ядро attention; q4_0 / q4_0 почти вдвое дешевле q8_0 по памяти.
Предел контекста через `--fit` (`-fitt 8192 -ub 2048`, без MTP): 177 152 токена при KV q4_0 и 95 744 при q8_0.

## Скрипты

`run_full_test.sh` поднимает llama-server под одну конфигурацию, пишет память и выбранный контекст, гоняет loop-тест
и/или бенч и гасит сервер. Env: `M` (модель), `TAG`, `CTX` (число или `auto` для `--fit`), `CTK`/`CTV`, `SPEC`,
`EXTRA`, `PHASES` (`loop`, `bench`), `LONG_TOKENS`, `BENCH_SHORT`, `FIT_MARGIN`, `MIN_AVAIL_MIB`. Выходы в
`~/glm46_runs/$TAG/`. Длинный кейс бенча берёт документацию llama.cpp с бокса (`/tmp/long_doc.txt`).

`run_campaign.sh` гоняет `run_full_test.sh` по строкам плана подряд и пишет сводку в `~/glm46_runs/campaign.log`.
Строка плана: `TAG|CTX|CTK|CTV|PHASES|LONG_TOKENS|BENCH_SHORT|SPEC|EXTRA`, план замеров выше `plan_full_iq2.txt`.

```
ssht gx10 'M=~/models/glm46-abl-mtp-IQ2_XXS-Q5K.gguf setsid bash ~/run_campaign.sh ~/plan_full_iq2.txt \
  > /tmp/campaign.log 2>&1 < /dev/null &'
```

`presign_pull.sh` печатает скрипт пулла с S3 на бокс: у gx10 нет AWS-кредов, поэтому presigned URL на 12 ч,
`aria2c -c -x 16 --max-tries=0` и `sha256sum -c` по `.sha256` рядом с каждым файлом. Вывод содержит временные ключи и
пишется в scratch, не в git. Канал бокса 16-20 МиБ/с, 94 GiB идут около 1.5 часа.

```
source infra/env.sh
serve/gx10/presign_pull.sh '~/models' glm46-full-gguf/glm46-abl-mtp-IQ2_XXS-Q5K.gguf > "$SCRATCH/pull.sh"
cat "$SCRATCH/pull.sh" | ssht gx10 'cat > /tmp/pull.sh'
ssht gx10 'setsid bash /tmp/pull.sh > /tmp/pull.log 2>&1 < /dev/null &'
```

Живость пулла проверять по `/tmp/pull.pid` (`kill -0`): `pgrep -f` из ssh находит собственную командную строку.
