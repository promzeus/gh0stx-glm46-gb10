#!/usr/bin/env bash
# Fetch the two ready-made draft models usable with a GLM-4.6 GGUF target in llama.cpp 4ebdf2c:
#   1. thoughtworks/GLM-4.7-FP8-Eagle3 (EAGLE-3, 1 layer, target layers [2,46,89]) converted to GGUF with
#      GLM-4.6 small files as --target-model-dir. GLM-4.7 and GLM-4.6 share config.json and tokenizer
#      byte for byte (same git blobs on HF); only chat_template.jinja differs.
#      Needs dflash/patches/glm4-moe-layer-inp.patch in the serving llama.cpp (--spec-type draft-eagle3).
#   2. jukofyork/GLM-4.5-DRAFT-0.6B-v3.0 GGUF (Qwen2.5-0.5B with the GLM vocab, --spec-type draft-simple).
# Env: LLAMA = llama.cpp checkout at 4ebdf2c (convert_hf_to_gguf.py), PY = python with the converter
#      requirements, W = work dir.
set -euo pipefail
LLAMA=${LLAMA:?path to llama.cpp 4ebdf2c}; PY=${PY:-python3}; W=${W:-./drafts}
HF=https://huggingface.co
mkdir -p "$W/tw_eagle3" "$W/glm46_small"

fetch() { [ -s "$2" ] || curl -fL --retry 5 -o "$2" "$1"; }
check() { echo "$2  $1" | shasum -a 256 -c -; }

fetch "$HF/thoughtworks/GLM-4.7-FP8-Eagle3/resolve/main/config.json" "$W/tw_eagle3/config.json"
fetch "$HF/thoughtworks/GLM-4.7-FP8-Eagle3/resolve/main/model.safetensors" "$W/tw_eagle3/model.safetensors"
check "$W/tw_eagle3/model.safetensors" 770b811326f1be2f2881d47b00871d9ef724dad72dcffbdf20a574824043522f
for f in config.json generation_config.json tokenizer.json tokenizer_config.json chat_template.jinja; do
  fetch "$HF/zai-org/GLM-4.6/resolve/main/$f" "$W/glm46_small/$f"
done

(cd "$LLAMA" && "$PY" convert_hf_to_gguf.py "$W/tw_eagle3" --target-model-dir "$W/glm46_small" \
   --outtype bf16 --outfile "$W/tw_glm47_eagle3_bf16.gguf")
check "$W/tw_glm47_eagle3_bf16.gguf" fd5a12888d42d422a62b37a8d58a3e9e15bf6f2403cade30030ba253f069f2f0

fetch "$HF/jukofyork/GLM-4.5-DRAFT-0.6B-v3.0-GGUF/resolve/main/GLM-4.5-DRAFT-0.6B-32k-Q4_0.gguf" \
  "$W/GLM-4.5-DRAFT-0.6B-32k-Q4_0.gguf"
check "$W/GLM-4.5-DRAFT-0.6B-32k-Q4_0.gguf" 058d9af36c4c84b3ed3ea7756692724b3bfa6a36e9b1a8349444605766efb71d
ls -la "$W"/*.gguf
