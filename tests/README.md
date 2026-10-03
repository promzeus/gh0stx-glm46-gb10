# tests

`loop/`: loop-тест длинного reasoning, 6 кейсов (short, explain, math, code, reason, longcode) × temp {0.0, 1.0},
512-4000 токенов, детектор повторов, выборка temp 1.0 + top_p 0.95 (+ top_k 20 в варианте topk).
`loop/w4a16_serve_test.py` шлёт top_p 0.95 (конфиг лучшего результата 7/12), `loop/w4a16_serve_test_topk.py` добавляет top_k 20.
Оба скопированы с gx10 2026-10-02. Сервер задаётся `W4A16_BASE`, имя модели `W4A16_MODEL`, выход `W4A16_OUT`.
Результаты по `glm46-nvfp4-gptq-full` (vLLM, gx10): 7/12 LOOP на top_p, 10/12 с top_k 20, RTN 9/12.
Результаты по `glm46-pruned-Q5_K_S.gguf` (llama-server на gx10, 2026-10-02, тот же скрипт top_p):
9/12 LOOP, чистые только `short t=1` и оба `code`; `math t=1` зацикливается на "I want to give the
answer by giving the steps", `reason` на "A well is a well". Полный выход в `loop/q5ks_loop_results.jsonl`.
llama-server добавляет свои дефолты сэмплера (top_k 40, min_p 0.05) к top_p 0.95 из скрипта.
Результаты по полной модели `glm46-abl-mtp-IQ2_XXS-Q5K.gguf` (llama-server на gx10, 2026-10-03, 32k, KV q8_0, тот же
скрипт): 1/12 по детектору, флаг ложный. `reason t=0` перебирает дни («Is 3m >= 10m? No. The snail is still in the
well» 7 раз), заканчивает сам (`fin=stop`) и отвечает верно, Day 8; детектор считает петлёй любой фрагмент,
повторённый 6 раз и больше. Все кейсы t=1.0 чистые, `reason` и `longcode` на обеих температурах закончили ответ
сами (`longcode` 3752 и 3051 токен, код 8-12 тыс. символов). Декод 10.9-11.4 ток/с. Выход в
`loop/full_iq2_loop_results.jsonl`.
Рабочая конфигурация (KV q4_0 / q4_0, `draft-mtp` n-max 1, контекст 113 664): тот же результат, 1/12 с тем же ложным
флагом на `reason t=0` (Day 8 верно на обеих температурах), длинные кейсы закончены сами. Выход в
`loop/prod_q4q4_mtp1/`. Петель от 4-битного KV на 12 кейсах нет.
Официальная выборка GLM-4.6: temp 1.0, top_p 0.95, top_k 40.

`loop/bf16_coherence_test.py`: 4 коротких промпта, 150 токенов, `transformers.generate`, без детектора.
Результаты в `bf16_coherence_results.jsonl`. С loop-тестом несравним.
