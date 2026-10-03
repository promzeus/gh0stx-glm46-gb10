# prune

`pipeline_reap_glm.py`: REAP expert-prune staged bf16 (`glm46-abliterated/`) → `glm46-pruned/`. transformers 5.17
держит экспертов фьюзнутыми (`Glm4MoeExperts`, `gate_up_proj [n_exp, 3072, 5120]`,
`down_proj [n_exp, 5120, 1536]`), прун режет dim0 по keep, режет `gate.weight` и
`e_score_correction_bias`, ставит `experts.num_experts` и config. `save_pretrained` пишет per-expert
имена, в S3 лежит обычный формат. shared_experts не трогаются, MTP в выход не идёт.

Метрика важности, env `METRIC`. `count` (дефолт, так собран `glm46-pruned/`): счётчик routed-selection
по доменам. `reap`: saliency из статьи REAP, среднее по токенам, где эксперт выбран, от `g_j(x) · ‖f_j(x)‖₂`;
`g_j` это `topk_weights` роутера (с нормировкой и `routed_scaling_factor`), `f_j(x)` выход эксперта до
взвешивания, считается forward-хуком на `Glm4MoeExperts` повтором матмулов эксперта для его токенов, так
что MoE-часть калибровочного прохода считается дважды. Обе статистики пишутся за один проход в
`expert_stats_glm.json` по доменам; с готовым файлом Phase A пропускается, и keep-set под другую метрику
пересчитывается без повторной загрузки 705 ГБ. В логе и в `importance_glm.json` есть пересечение
keep-set двух метрик по блокам. Объединение по доменам round-robin по рангу внутри домена, эксперты
с нулём выборов в домене не берутся.

Калибр: `calib_corpus.assemble()` (домен-сбалансированный, `CALIB_TARGET` 128, `MAXLEN` 1024, в репозиторий
не входит) или `CALIB_JSONL` со строками `{"domain": ..., "text": ...}`. Такой файл собирает `calib_mix_jsonl.py`
(джоба `glm46-calib-job.yaml`, выход `corpus/calib_mix_1200x4k.jsonl`): трейсы `<think>` из glaive, OpenR1-Math и
codeforces-cots, русский OpenHermes-2.5-ru, OpenHermes-2.5 и Magicoder, всё через chat_template GLM. Тот же корпус
идёт в imatrix полной модели (`gguf/`).
Бюджет Phase A на CPU задаёт `CALIB_TARGET × MAXLEN`: 128 × 1024 прошло в первом прогоне, цифр по времени
Phase A в логе не осталось.

| env | значение в прогоне |
|---|---|
| `MODEL_ID` | `/data/abl` |
| `PRUNED_DIR` / `OUT_DIR` | выход |
| `KEEP_FRAC` | 0.40 → 64 из 160 |
| `DEVICE_MAP` | cpu |
| `CALIB_TARGET`, `MAXLEN` | 128, 1024 |
| `METRIC` | count (reap не гонялся) |

Нода: r7i.48xlarge spot, NodePool `cpu-prune`, корень 1350Gi, cpu 80 / mem 1400Gi, backoffLimit 3.
Грабли: `from_pretrained` открывает каждый шард из индекса, `--exclude mtp.safetensors` роняет загрузку,
синкать полный набор; вход mmap-ится, `rm` мид-прогоне блоки не освобождает, вход и выход должны
помещаться рядом.

Вердикт 2026-10-02: `glm46-pruned` сломан самим прунингом. Q5_K_S (5.51 BPW, почти без потерь кванта) на
llama-server даёт 9/12 LOOP на loop-тесте, NVFP4 на vLLM 7/12, два независимых стека с одной картиной:
начало reasoning связное, через 300-600 токенов фраза зацикливается (`tests/loop/q5ks_loop_results.jsonl`).
Совпадает со статьёй REAP: частотная метрика на 50% разваливается на больших моделях, у нас 60%
(`KEEP_FRAC=0.40`) и калибр 128 × 1024 против 12k × 16k у Cerebras. Следующий прогон: `METRIC=reap`,
калибр из reasoning-трейсов через `CALIB_JSONL`; `KEEP_FRAC` 0.40 оставляет 151B и ~85 ГБ NVFP4,
0.50 даёт ~190B и ~105 ГБ, на боксе с KV впритык. Прогон не запущен.
