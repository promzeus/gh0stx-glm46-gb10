# GLM-4.6 петли: что нашлось в сообществе и как разделить причины (2026-10-02)

Дополняет `glm46-gptq-context.md`. Пайплайн prune → quant → deploy даёт три места,
где могли появиться петли. Ни одно пока не измерено на длинной генерации по отдельности.

## Ранжирование подозреваемых

**Прунинг.** Метрика «счётчик routed-selection» в статье REAP идёт как frequency-baseline и на
50% прунинга разваливается на больших моделях: Qwen3-Coder-480B 0.010 против REAP 0.619,
Kimi-K2 0.056 против 0.624, GLM-4.5-Air 0.308 против 0.515 (coding, pass-rate). Причина по
статье: редко выбираемый эксперт с большим вкладом в выход слоя вылетает первым. У нас 60%
с этой метрикой, калибр 128 × 1024, у Cerebras для моделей от 110B 12k сэмплов × 16k токенов.
Даже с полноценным REAP на 40% GPQA diamond (thinking) у Cerebras падает 78.8 → 69.7, AIME25
держится 90 → 90.

**Стек vLLM на GB10 (sm_121).** Четыре задокументированных бага NVFP4-пути:
- CUTLASS FP4 выбирается на SM121 ложно (`is_device_capability_family(120)` матчит SM12x),
  PTX `cvt .e2m1x2` на SM121 нет. Кернел пишет один скаляр на всю строку: все логиты равны,
  argmax даёт токен 0, в byte-level BPE это восклицательный знак. Это сигнатура мусора
  нашего стрим-кванта. Объяснение «global_scale=1.0 →
  мёртвые эксперты» нужно сверить с логами обоих сервов: какой MoE-бэкенд брал каждый.
- Гонка потоков в Marlin MoE, недетерминированные неверные выходы. Фикс
  `VLLM_MARLIN_USE_ATOMIC_ADD=1` или 128 потоков на блок.
- Triton downcast через тот же PTX.
- Апстрим-фикс корня: PR #37725 «Software E2M1 conversion for SM12x».
Известный годный образ `nvcr.io/nvidia/vllm:26.04-py3`. Рабочий набор флагов из гайдов:

```
VLLM_USE_FLASHINFER_MOE_FP4=0
VLLM_NVFP4_GEMM_BACKEND=marlin
VLLM_MXFP4_USE_MARLIN=1
VLLM_MARLIN_USE_ATOMIC_ADD=1
--moe-backend marlin --attention-backend TRITON_ATTN --gpu-memory-utilization 0.85
```

**Рецепт кванта.** 87.3 ГБ / 151B = 4.63 бит на параметр: attention, shared, dense 0-2 и
роутер тоже в NVFP4. Эталонные рецепты так не делают: NVIDIA Nemotron 3 Ultra держит attention
BF16, shared FP8, routed NVFP4; LibertAI GLM-5.3-Flash квантует только
`experts.*.{gate,up,down}_proj`; nvidia/DeepSeek-R1-0528-FP4 аналогично. Калибр 256 × 2048
без `<think>`-трейсов. Вес этого подозреваемого ниже: RTN → GPTQ сдвинул 9 → 7, не убрал петли.

## Лестница под правило «gx10 первым»

Правило: шаг входит в 128 ГБ gx10, делаем на gx10. Не входит физически, тогда минимальный
инстанс AWS. Тест везде один: `w4a16_serve_test.py`, 12 кейсов, детектор.

Ступень 0, gx10, бесплатно:
1. `docker inspect vllm-glm46-full` (Cmd, Env) и
   `docker logs vllm-glm46-full 2>&1 | grep -iE 'marlin|cutlass|flashinfer|moe.*backend|nvfp4'`.
2. Один промпт пять раз при temp 0. Разные ответы при greedy означают гонку в кернеле.
3. Перезапуск с флагами выше, те же 12 кейсов.
4. Тот же чекпоинт на другом образе: годный NGC-образ из раздела про стек или
   `vllm/vllm-openai:cu130-nightly`, `--moe-backend marlin`. Это закрывает ось «стек GB10»
   без AWS: петли на трёх образах с Marlin означают чекпоинт, не стек.
5. Сток GLM-4.6 через официальный API на тех же 12 кейсах: порог детектора. Если сток при
   temp 0 сам петляет на `reason`, эти клетки не урон, судим по temp 1.0.

Ступень 1, gx10: прунутая модель до NVFP4, другим движком и форматом. llama.cpp (NVIDIA
даёт playbook под Spark), GGUF Q5_K_S. Размеры по таблице mradermacher для REAP-218B
(Q5_K_S 150.5 ГБ, Q4_K_M 131.9 ГБ, Q6_K 179.4 ГБ) в пересчёте на 151B: Q6_K ≈ 124 ГБ не входит.
Фактически собрано (`glm46-pruned-gguf/`): Q5_K_S 104.2 ГБ (5.51 BPW), Q4_K_M 91.1 ГБ (4.82 BPW). KV на 32k: f16 12.3 ГБ, q8_0 6.2 ГБ. Q5_K_S с KV
q8_0 и буферами ≈ 113 ГБ входит, Q4_K_M с запасом.

Конверсия `convert_hf_to_gguf.py` стримит тензоры, RAM нужен небольшой, диск 300 ГБ вход +
104 ГБ выход. Есть диск и канал на gx10, конвертируем на боксе. Нет, тогда минимальный AWS:
CPU-инстанс класса m7i.4xlarge (16 vCPU, 64 ГБ RAM) с gp3 на 500 ГБ, читает `glm46-pruned`
из S3 в регионе, кладёт GGUF в S3, пулл 104 ГБ на gx10 той же схемой, что 87 ГБ раньше.
Формат проверен по `model.safetensors.index.json` в S3: эксперты лежат per-expert
(`model.layers.N.mlp.experts.N.{gate,up,down}_proj.weight`, по 5696 тензоров каждого вида,
89 слоёв × 64), фьюзнутый 3D-формат был только в памяти transformers 5.17. Конвертер
llama.cpp (`conversion/glm.py`, `Glm4MoeModel`) берёт такие имена без патча. Две правки
всё же нужны: `--no-mtp`, потому что config.json объявляет `num_nextn_predict_layers: 1`,
а тензоров слоя 92 в чекпоинте нет; и стоковый `tokenizer_config.json` из `glm46-base/`
вместо сохранённого transformers 5.17 (`tokenizer_class: TokenizersBackend`), который пин
конвертера transformers 4.57.6 не знает. `tokenizer.json` прунутого и стокового равны по
структуре (vocab 151329, merges 318088, 36 added tokens), отличие в одном байте форматирования.
`chat_template.jinja` gguf-py читает сам. Локальный `--vocab-only --no-mtp` пре-флайт на этих
файлах прошёл. Рецепт и джоба: `gguf/`. Диск: `--outtype q8_0` сразу (160 ГБ), потом
`llama-quantize --allow-requantize` в Q5_K_S и Q4_K_M, пик 302 + 160 ГБ, gp3 600Gi, чтобы
остаться ниже порога eviction kubelet в 10% свободного nodefs.

| llama.cpp Q5 на gx10 | nvfp4 на других образах gx10 | вывод |
|---|---|---|
| петли | | урон до кванта (прунинг) |
| чисто | петли | рецепт NVFP4 или бит мало |
| чисто | чисто | aeon-форк |

Ступень 1b, если виноват рецепт. Ре-квант не входит на gx10 (300 ГБ bf16 CPU-resident),
минимальный AWS под это g6e.16xlarge, как раньше: experts NVFP4 GPTQ, остальное FP8 (94 ГБ),
калибр с `<think>`-трейсами. Serve на gx10 с marlin-флагами. Вариант без AWS: оставить
llama.cpp Q5_K_S или Q4_K_M на gx10 как рабочий serve, если скорость устраивает.

Ступень 2, если урон до кванта. 705 ГБ resident не входит никуда, кроме r7i.48xlarge spot:
пере-прунинг с REAP-saliency вместо частоты, калибр от 512 × 4k с `<think>`-трейсами.
Новый pruned проверять ступенью 1 на gx10 до кванта.

Вердикт 2026-10-02: ступень 1 дала петли, Q5_K_S 9/12 LOOP против 7/12 у NVFP4 (`tests/README.md`).
Урон до кванта, ступень 1b снята с очереди.

Решение 2026-10-03: вместо пере-прунинга полная модель с экспертами на 2 битах (`gguf/README.md`). Подход,
который работал на Qwen3.5-397B, сюда не переносится. У Qwen 512 экспертов по 1024 и top-10, срезы оставляли
174-184 из 512; у GLM-4.6 160 по 1536 и top-8, срез оставил 64 из 160, заменить выбитого эксперта нечем. После
пруна Qwen лечили LoRA на attention (r16, 150 шагов, bf16 на 8 GPU); на пруненной GLM heal не делали, класс
H100/H200 закрыт. Qwen проверяли `test_coherence.py`: thinking выключен, 100 токенов greedy с repetition_penalty
1.2; длинный reasoning на нём не мерили. Файл задают параметры × биты: 151B × 5.51 бит дали 104 ГБ, 353B с
экспертами IQ2_XXS (2.06 бит) и остальным Q5_K дают около 100 ГБ. Активных параметров на токен столько же (32B),
байт на токен меньше, KV тот же: прун не давал ни скорости, ни контекста.

## Бюджет памяти gx10

Конфиг GLM-4.6: 92 слоя, 3 dense, 64 эксперта 5120×1536, 8 KV-голов × 128. CUDA видит
121.69 GiB; при utilization 0.87 это 114 ГБ под веса и KV. KV fp8: 188 КБ/токен, 32k = 6.2 ГБ,
120k = 22.6 ГБ. bf16 KV вдвое больше.

| блок | параметров | NVFP4 | FP8 | BF16 |
|---|---|---|---|---|
| routed experts, 64 | 134.4B | 75.6 ГБ | | |
| routed experts, 72 | 151.2B | 85.0 ГБ | | |
| attention q/k/v/o | 12.5B | 7.0 | 12.5 | 25.0 |
| shared + dense MLP | 2.7B | 1.5 | 2.7 | 5.4 |
| embed + lm_head | 1.55B | | | 3.1 |

Варианты: сейчас 87.2 ГБ; 64 эксперта, attention и shared в FP8: 94 ГБ, 32k входит с запасом,
120k не входит (94 + 22.6 > 114); 72 эксперта всё NVFP4: 97 ГБ; 72 эксперта + attention FP8:
103 ГБ, 32k впритык. Цель `--max-model-len 120000` держит только текущие 87 ГБ.

В llm-compressor это два `config_groups`: NVFP4A16 + GPTQ на `re:.*experts.*`, FP8 на
остальные Linear, ignore `lm_head` и `mlp.gate`. Код: `quant-nvfp4/nvfp4_quant.py --recipe mixed`,
калибр с трейсами `glm_calib_gen.py CALIB_SOURCE=reasoning`, ручки в `glm-requant-full-job.yaml`.

## Если виноват прунинг

REAP-saliency вместо частоты: S_j = mean по токенам, где эксперт j выбран, от g_j(x)·‖f_j(x)‖₂.
Считается из тех же forward-проходов, что и счётчик. Реализации: репозиторий Cerebras
(glm4_moe поддержан, GLM-4.6-REAP сделан им) или `REAPModifier` в llm-compressor. Калибр
от 1k сэмплов, seq 8-16k, домены: английские `<think>`-трейсы (сгенерировать стоком через API
или с bf16 на p5e), код, tool-calling, agentic. C4-only калибр в статье давал 0% на коде.
«Think Before You Prune» (2511.18864): калибр из собственных reasoning-трейсов модели
сохраняет длинный CoT лучше стороннего текста. Код: `prune/pipeline_reap_glm.py METRIC=reap`, калибр
через `CALIB_JSONL`; статистики обеих метрик пишутся за один проход, keep-set пересчитывается без
повторной загрузки.

## Выборка

Официальная для GLM-4.6: temp 1.0, top_p 0.95, top_k 40. Наш top_k 20 уже и дал 10/12.

## Источники

- [REAP paper, arXiv 2510.13999](https://arxiv.org/html/2510.13999)
- [cerebras/GLM-4.6-REAP-218B-A32B-FP8, бенчмарки](https://huggingface.co/cerebras/GLM-4.6-REAP-218B-A32B-FP8)
- [llm-compressor REAP example](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/reap_expert_pruning/)
- [Think Before You Prune, arXiv 2511.18864](https://arxiv.org/pdf/2511.18864)
- [ai-muninn: четыре бага SM121](https://ai-muninn.com/en/blog/part1-why-your-dgx-spark-says-exclamation-marks)
- [NVIDIA forum: Marlin fix NVFP4 on SM121](https://forums.developer.nvidia.com/t/marlin-fix-nvfp4-actually-works-on-sm121-dgx-spark/365119)
- [vLLM SM121 gotchas](https://conselara.dev/notes/vllm-dgx-spark-sm121-gotchas/)
- [DGX Spark vLLM playbook](https://vlaicu.io/posts/dgx-vllm/)
- [Nemotron 3 Ultra NVFP4 recipe](https://developer.nvidia.com/blog/creating-the-nvidia-nemotron-3-ultra-nvfp4-checkpoint-with-nvidia-model-optimizer/)
- [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)
- [nvidia/DeepSeek-R1-0528-FP4](https://huggingface.co/nvidia/DeepSeek-R1-0528-FP4)
- [GLM-4.6 model card](https://huggingface.co/zai-org/GLM-4.6)
- [DGX Spark handbook](https://huggingface.co/blog/exolabs/the-dgx-spark-handbook)
- [vLLM Transformers backend, MoE](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/transformers/moe/)
- [llama.cpp conversion/glm.py](https://github.com/ggml-org/llama.cpp/blob/master/conversion/glm.py)
- [mradermacher GLM-4.6-REAP-218B GGUF sizes](https://huggingface.co/mradermacher/GLM-4.6-REAP-218B-A32B-GGUF)
