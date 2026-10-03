# calib

Калибровочный корпус для imatrix полной модели (`gguf/`): `corpus/calib_mix_1200x4k.jsonl` в S3. Каждая строка —
один диалог через `chat_template.jinja` GLM-4.6 (`{"domain", "text"}`), до 4096 токенов. Reasoning-строки обязаны
содержать `</think>` и иметь от 768 токенов, обычные от 128.

| домен | источник | строк | токенов |
|---|---|---|---|
| reasoning | glaiveai/reasoning-v1-20m (prompt/response) | 240 | 392 535 |
| math | open-r1/OpenR1-Math-220k (problem/generations[0]) | 120 | 423 895 |
| code_cot | open-r1/codeforces-cots, `solutions_py` | 120 | 474 481 |
| ru | d0rj/OpenHermes-2.5-ru (sharegpt) | 360 | 172 159 |
| general | teknium/OpenHermes-2.5 (sharegpt, `refs/convert/parquet`) | 240 | 103 163 |
| code | ise-uiuc/Magicoder-OSS-Instruct-75K | 120 | 61 579 |

Итого 1200 строк, 1.63M токенов. Сборка GGUF берёт из корпуса 400k токенов с квотами по доменам (reasoning 25%,
math 15%, code_cot 15%, ru 20%, general 15%, code 10%), квоты заданы в `gguf/glm-gguf-full-job.yaml`.

```
source infra/env.sh
aws s3 cp calib/calib_mix_jsonl.py "$S3_BUCKET/glm46/calib_mix_jsonl.py"
tools/render-job.sh calib/glm46-calib-job.yaml | kubectl apply -f -
```

Джоба идёт около 10 минут на любой маленькой ноде, лимит памяти 12Gi. Грабли, на которых она падала 2026-10-03:
`IlyaGusev/ru_turbo_alpaca` содержит скрипт загрузки, который `datasets` больше не исполняет; `teknium/OpenHermes-2.5`
— один JSON на 1.9 ГБ, стриминг читает его целиком и ловит OOMKilled на 12Gi, поэтому он читается из
`refs/convert/parquet`; `apply_chat_template` требует `jinja2`.
