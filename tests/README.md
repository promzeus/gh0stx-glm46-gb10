# tests

`loop/loop_test.py`: loop-тест длинного reasoning, 6 кейсов (short, explain, math, code, reason, longcode) × temp {0.0,
1.0}, 512-4000 токенов, выборка top_p 0.95 (llama-server добавляет свои top_k 40 и min_p 0.05). Детектор повторов:
слово 8 раз подряд, 30-символьный фрагмент 6 раз и больше, уникальность 4-грамм ниже 0.5. Пошаговое перечисление
тоже даёт флаг, поэтому флагнутый кейс надо прочитать. Env: `LOOP_BASE`, `LOOP_MODEL`, `LOOP_OUT`. Официальная
выборка GLM-4.6: temp 1.0, top_p 0.95, top_k 40.

`loop/spec_bench.py`: скорость и приём драфта через блок `timings` llama-server. 6 запросов по 768 токенов (RU,
reasoning, code при t=0 и t=1) и длинный кейс из документа на `LONG_TOKENS` токенов. Env: `BENCH_BASE`, `BENCH_MODEL`,
`BENCH_OUT`, `BENCH_MAX`, `BENCH_SHORT`, `LONG_FILE`, `LONG_TOKENS`.

| модель | loop-тест | данные |
|---|---|---|
| полная `glm46-abl-mtp-IQ2_XXS-Q5K.gguf`, KV q8_0, 32k | 1/12, флаг ложный | `loop/full_iq2_loop_results.jsonl` |
| она же, KV q4_0, `draft-mtp` n-max 1, 113k | 1/12, флаг ложный | `loop/prod_q4q4_mtp1/` |
| пруненная 64/160, GGUF Q5_K_S | 9/12 | `loop/q5ks_loop_results.jsonl` |
| пруненная 64/160, NVFP4 GPTQ (vLLM) | 7/12, с top_k 20 10/12 | |
| пруненная 64/160, NVFP4 RTN (vLLM) | 9/12 | |

Ложный флаг полной модели: `reason t=0` перебирает дни («Is 3m >= 10m? No. The snail is still in the well» 7 раз),
заканчивает сам (`fin=stop`) и отвечает верно, Day 8. Все кейсы t=1.0 чистые, `reason` и `longcode` на обеих
температурах закончили ответ сами (`longcode` 3-4 тыс. токенов, код 7-12 тыс. символов). Пруненная Q5_K_S петляла
по-настоящему: `math t=1` до лимита повторяла «I want to give the answer by giving the steps», `reason` — «A well is a
well». Замеры скорости: `serve/gx10/README.md`, сырые данные кампании в `loop/campaign_full_iq2/`.
