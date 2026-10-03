# dflash

Спекулятивный декод для GLM-4.6 в llama.cpp на gx10. Цель: GGUF `glm46-abl-mtp-IQ2_XXS-Q5K` из `gguf/`, MTP-слой
в нём есть как `blk.92`. На Qwen3.5-397B DFlash на этом боксе поднял декод с ~6 до 22 ток/с (дообученный драфтер,
`~/gb10/training/README.md` на gx10). Публичного DFlash- или EAGLE-3-драфтера под GLM-4.5/4.6 на HF нет (поиск
2026-10-03 по eagle, dflash, dspark, speculator и по авторам z-lab, RedHatAI, modal-labs, lmsys, nvidia).

## llama.cpp и цель GLM4_MOE

На 4ebdf2c `draft-eagle3` и `draft-dflash` с целью GLM4_MOE падают на
`src/llama-graph.cpp:1384: GGML_ASSERT(t_layer_inp[il] != nullptr && "layer input tensor is null")`: граф
`src/models/glm4-moe.cpp` не отдаёт входы слоёв. Остальные 19 графов (qwen3moe, minimax-m2, deepseek4 и др.)
отдают. Правка `patches/glm4-moe-layer-inp.patch`, одна строка в начале цикла слоёв:

```
res->t_layer_inp[il] = inpL;
```

С правкой вход слоя `id+1` совпал с `l_out-<id>` из `cb_eval` бит в бит (max|diff| 0) на 32 токенах, при двух
ubatch по 16 и при упаковке двух последовательностей в один батч (`tools/hsdump.cpp`, `tools/packtest.cpp`, CPU-сборка
4ebdf2c на маке, случайная модель из `test-llama-archs -a glm4moe`). Выход последнего слоя (`lid = n_layer`) падает и с
правкой: glm4-moe режет строки последнего слоя по `inp_out_ids`. DFlash с id 91 потребует ещё правку по образцу
`mimo2.cpp`; дефолтные id SpecForge для 92 слоёв (`[1,23,45,67,89]`) её не задевают.

Обвязка на случайной крошечной GLM4_MOE (приём там нулевой, проверяется только запуск):

| режим | 4ebdf2c | с правкой |
|---|---|---|
| без спекуляции, `ngram-mod` | работает | работает |
| `draft-mtp`, `draft-mtp` + `ngram-mod`, `--spec-synth-len` | работает | работает |
| `draft-eagle3` | падает | работает |
| `draft-simple` (0.6B-драфт) | работает | работает |
| `draft-dflash`, фичи до предпоследнего слоя | падает | работает |
| `draft-dflash` с фичей последнего слоя | падает | падает |

На gx10 правка применена к `~/llama.cpp` и llama-server пересобран 2026-10-03 (вместе с FA-ядром `q8_0-q4_0`,
см. `serve/gx10/README.md`).

## Готовые драфты

`scripts/get_drafts_glm46.sh` скачивает и конвертирует оба внешних драфта и сверяет sha256. Оба лежат на gx10 в
`~/models/glm46-drafts/`.

| драфт | флаги | размер | что известно |
|---|---|---|---|
| MTP `blk.92` из целевого GGUF | `--spec-type draft-mtp --spec-draft-n-max 1..3` | веса цели, ~2.3 ГБ эксперты Q4_K; KV 1 слоя, 128 MiB на 32k в f16 | PR #26534; GLM-4.5-Air на Strix Halo (unified memory): приём 88.5/71.5/62.9%, 1.25/1.24/1.20x при n-max 1/2/3; полный GLM-4.5 UD-IQ2_M 1.07x; на CUDA приём MTP ниже, чем на Vulkan (#26750, открыт) |
| EAGLE-3 `tw_glm47_eagle3_bf16.gguf` | `-md <файл> --spec-type draft-eagle3` | 1.2 ГБ bf16, 594M параметров, словарь драфта 32000 | обучен на GLM-4.7-FP8, max_length 1024; на родной цели AL 2.76, 1.69x (8xH200, B=1, temp 0); в d2t нет `<think>` и tool-токенов; на DGX Spark EAGLE-3 для gpt-oss-120b замедлял до 0.46-0.71x (PR #18039) |
| `GLM-4.5-DRAFT-0.6B-32k-Q4_0.gguf` | `-md <файл> --spec-type draft-simple` | 431 МБ | Qwen2.5-0.5B со словарём GLM, обучен на общем корпусе; автор сообщает 94% приёма на рефакторинге кода и ~30% в чате, нужен temp 0 или top_k 1 |
| n-gram | `--spec-type ngram-mod` | ~16 МБ | драфтит при дословном повторе от 48 токенов; на петлях раздувает приём |

Модельный драфт (`-md`) и `draft-mtp` вместе не грузятся: у сервера один контекст драфта. `ngram-mod` с ними
сочетается. Конвертация EAGLE-3: GLM-4.7 и GLM-4.6 имеют одинаковые `config.json` и `tokenizer.json` (одни git blob на HF),
поэтому `--target-model-dir` указывает на мелкие файлы GLM-4.6; повтор конвертации на 4ebdf2c даёт тот же sha256
`fd5a1288…`.

## Свой DFlash под GLM-4.6

Форма драфтера по статье (arXiv 2602.06036v2) и SpecForge: 5 слоёв Qwen3, hidden 5120 (равен цели, `fc` строится на
`n_target_layers × hidden`), 32×128 q-голов, 8 KV, intermediate 17408, блок 16, фичи со слоёв `[1,23,45,67,89]`,
эмбеддинги и голова общие с целью. Это ~1.73B параметров: 3.2 GiB в bf16, ~1.8 ГБ в Q8_0, плюс KV драфта на
n_ctx цели (5 слоёв по 4 KiB на токен в f16) и 40 MiB host-буфера на извлекаемый слой при `-b 2048`.
`mask_token_id` 151330 (`[MASK]`, строка эмбеддинга ненулевая); `vocab_size` 151552.

Стоимость обучения упирается в данные. Рецепт статьи: ~800K промптов, ответы генерирует сама цель, 6 эпох. На
gx10 при 8.5 ток/с это ~1.1B токенов генерации, годы; кэш фич такого корпуса для 5 слоёв ~77 ТБ (51 200 байт на
токен в bf16). Опыт Qwen на этом боксе: драфтер с нуля на ~2.7K self-distill сэмплах не обучился (unseen AL 1.2),
рабочим стал warm-start от официального драфтера z-lab (1 эпоха, 2671 ответ, ~2.5 ч на GB10). Официального драфтера
для warm-start под GLM-4.6 нет. Драфтер под другую цель на изменённой модели даёт приём около нуля (abliterated
Qwen3.8-27B с чужим DFlash2: AL 1.5, медленнее, чем без спекуляции).

Фичи для обучения снимаются самим llama.cpp, без vLLM: `llama_set_embeddings_layer_inp(ctx, lid, true)` до decode и
`llama_get_embeddings_layer_inp(ctx, lid)` после (`src/llama-ext.h:110-115`), нужна правка выше. `tools/hsdump.cpp`
проверяет этот путь на случайных токенах; для реального съёма ему нужен вход token ids и маска из файла.

## Скрипты Qwen-периода

`scripts/vllm_hs_extractor.py`: снимает hidden-states с целевых слоёв через forward-хуки vLLM на боксе,
input_ids строит тем же `build_eagle3_dataset`, что и `train_dflash`, чтобы хэши совпали с offline-backend.
`scripts/vllm_hs_extractor_mtp_calib.py`: тот же съём под калибровку keep-set MTP-головы. `scripts/offline_dflash_patch.py`:
offline-backend для SpecForge, `train_dflash` учится на снятых hidden-states без живого teacher.
`scripts/v4-dflash.json`: конфиг драфта под Qwen3.5-397B (hidden 4096, vocab 248320); для GLM-4.6 меняются hidden,
vocab, mask, target_layer_ids и bos/eos.
