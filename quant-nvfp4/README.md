# quant-nvfp4

Рабочий путь `nvfp4_quant.py`: `load_quantizable_moe` линеаризует экспертов в LinearExperts2D,
`oneshot(moe_calibrate_all_experts=True)` sequential pipeline, `save_pretrained(save_compressed=True)`.
Весь bf16 (302 ГБ) CPU-resident плюс одна GPU для onload по блоку, пик на GPU 47.7/48 ГБ.
Калибр офлайн `glm_calib_gen.py` (256 строк, seq 2048, без `<think>`), схема NVFP4A16 + GPTQ.
In-job VERIFY до заливки: ни у одного эксперта `weight_global_scale` не остался дефолтным 1.0.

| env | значение в прогоне |
|---|---|
| `NVFP4_IN` / `NVFP4_OUT` | `glm46-pruned/` → `glm46-nvfp4-gptq-full/` |
| `NVFP4_SCHEME` | NVFP4A16 |
| `NVFP4_MODIFIER` | GPTQ |
| `NVFP4_CALIB_JSONL` | выход `glm_calib_gen.py` |

Нода: g6e.16xlarge on-demand (1 × L40S 48 ГБ, 512 ГБ RAM, 64 vCPU), NodePool `nvfp4-quant` из
`infra/karpenter/`, прогон около 1 ч 45 м. Баг, который ловили: JSONL-калибр даёт 1D `input_ids`,
sequential-пайплайн индексирует их двумя индексами, фикс в collator (batch-dim для 1D).

`cpu_wonly_nvfp4.py`: RTN W4A16 без калибровки на CPU, это `glm46-nvfp4a16/` (петлит).
`gen_calib.py`: self-distilled калибр из `<think>`-трейсов прунутого bf16, нужна GPU-нода под
генерацию, в прогоне не использовался.

`stream/`: по-слойный путь через `oneshot(pipeline="basic")`, тупик для MoE: экспертов не калибрует,
`global_scale=1.0`, мусор на выходе. Оставлен как документация.

Рецепт 1b из `docs/findings.md` лежит в тех же скриптах, прогона нет. `nvfp4_quant.py --recipe mixed`
(`NVFP4_RECIPE=mixed`): два `config_groups` в одном GPTQModifier, `NVFP4A16` на `re:.*mlp\.experts\..*`,
`FP8_DYNAMIC` (веса FP8 per-channel, активации FP8 per-token динамически) на attention q/k/v/o,
shared_experts и dense MLP слоёв 0-2; ignore прежний (`lm_head`, нормы, embed, `mlp.gate`). Оценка
размера 94 ГБ против 87.3. `glm_calib_gen.py CALIB_SOURCE=reasoning`: строки из публичных `<think>`-трейсов
(glaiveai/reasoning-v1-20m, open-r1/OpenR1-Math-220k, open-r1/codeforces-cots `solutions_py`), прогнанные
через chat_template GLM, чтобы в калибре был поток `<|assistant|>\n<think>…</think>`; трейсы в стиле
DeepSeek-R1, не собственные. Обе ручки выставляются в `env` джобы `glm-requant-full-job.yaml`
(`NVFP4_RECIPE`, `CALIB_SOURCE`), выход mixed идёт в `glm46-nvfp4-mixed/`.
