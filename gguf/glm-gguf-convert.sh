#!/usr/bin/env bash
# GLM-4.6 pruned bf16 (HF safetensors, per-expert expert names) -> GGUF q8_0
# -> K-quants (llama-quantize --allow-requantize) -> S3. CPU-only node.
# Disk on /data: 302 GB input + 160 GB q8_0 at peak, then q8_0 + one K-quant.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore

: "${SRC_PREFIX:?}" "${BASE_PREFIX:?}" "${DST_PREFIX:?}"
QUANTS="${QUANTS:-Q5_K_S Q4_K_M}"
UPLOAD_Q8="${UPLOAD_Q8:-1}"
THREADS="$(nproc)"
WORK=/data; HF=$WORK/hf; OUT=$WORK/out; LLAMA=/opt/llama.cpp
mkdir -p "$HF" "$OUT"

log()  { echo "[$(date -u +%FT%TZ)] $*"; }
disk() { df -h "$WORK" | tail -1; }

log "node: ${THREADS} cpu, $(free -g | awk '/Mem:/{print $2}') GiB ram"; disk

# --- tools ---------------------------------------------------------------------
# S3 I/O via awscli v2: it supports EKS Pod Identity (the SA has a pod identity
# association, no IRSA annotation). s5cmd 2.3.0 is on aws-sdk-go v1 and fails
# with NoCredentialProviders here.
log "apt: cmake unzip"
apt-get update -qq >/dev/null && apt-get install -y -qq cmake unzip >/dev/null
log "awscli v2"
curl -sSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscli.zip
unzip -q /tmp/awscli.zip -d /tmp && /tmp/aws/install >/dev/null && rm -rf /tmp/awscli.zip /tmp/aws
aws --version
aws configure set default.s3.max_concurrent_requests 32
aws configure set default.s3.max_queue_size 10000
aws configure set default.s3.multipart_chunksize 64MB
aws sts get-caller-identity --query Arn --output text   # pod-identity credentials check
aws s3 ls "$SRC_PREFIX/" >/dev/null

# --- llama.cpp -----------------------------------------------------------------
log "clone llama.cpp"
git clone --depth 1 --quiet https://github.com/ggml-org/llama.cpp "$LLAMA"
COMMIT="$(git -C "$LLAMA" rev-parse --short HEAD)"; log "llama.cpp $COMMIT"
cd "$LLAMA"
log "pip: converter deps (torch cpu)"
pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -q -r requirements/requirements-convert_hf_to_gguf.txt
log "build llama-quantize"
cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF >/dev/null
cmake --build build --target llama-quantize -j"$THREADS" >/dev/null
ls -la build/bin/llama-quantize

# --- small files; tokenizer_config.json from the stock model -------------------
# The pruned dir was saved by transformers 5.17: its tokenizer_config.json says
# tokenizer_class=TokenizersBackend, unknown to transformers 4.57 pinned by the
# converter. The stock tokenizer_config.json loads the same tokenizer.json.
log "download small files"
aws s3 cp "$SRC_PREFIX/" "$HF/" --recursive --exclude "*.safetensors*" --only-show-errors
aws s3 cp "$BASE_PREFIX/tokenizer_config.json" "$HF/tokenizer_config.json" --only-show-errors
ls -la "$HF"
python3 - <<'PY'
import json
c = json.load(open('/data/hf/config.json'))
assert c['n_routed_experts'] == 64, c['n_routed_experts']
assert c['num_hidden_layers'] == 92, c['num_hidden_layers']
print('config ok: experts', c['n_routed_experts'], 'layers', c['num_hidden_layers'],
      'nextn', c.get('num_nextn_predict_layers'))
PY

# --- vocab-only pre-flight, before the 302 GB download -------------------------
log "pre-flight: vocab-only"
python3 convert_hf_to_gguf.py "$HF" --vocab-only --no-mtp --outfile "$OUT/preflight-vocab.gguf"
rm -f "$OUT/preflight-vocab.gguf"

# --- weights -------------------------------------------------------------------
log "download shards"; disk
aws s3 cp "$SRC_PREFIX/" "$HF/" --recursive --exclude "*" --include "*.safetensors" --include "model.safetensors.index.json" --only-show-errors
log "downloaded: $(du -sh "$HF" | cut -f1)"; disk

# --- convert -> q8_0 -------------------------------------------------------------
Q8="$OUT/glm46-pruned-q8_0.gguf"
log "convert -> q8_0 (--no-mtp: config declares 1 nextn layer, checkpoint has none)"
python3 convert_hf_to_gguf.py "$HF" --outtype q8_0 --no-mtp --outfile "$Q8"
ls -la "$Q8"; disk
log "rm hf input"
rm -rf "$HF"; disk

# --- quantize + upload -----------------------------------------------------------
upload() {
  local f="$1" b; b="$(basename "$f")"
  log "sha256 $b"; (cd "$(dirname "$f")" && sha256sum "$b" > "$b.sha256")
  log "upload $b"
  aws s3 cp "$f" "$DST_PREFIX/$b" --only-show-errors
  aws s3 cp "$f.sha256" "$DST_PREFIX/$b.sha256" --only-show-errors
  aws s3 ls "$DST_PREFIX/$b"
}
for q in $QUANTS; do
  F="$OUT/glm46-pruned-$q.gguf"
  log "quantize $q"
  "$LLAMA/build/bin/llama-quantize" --allow-requantize "$Q8" "$F" "$q" "$THREADS" | tail -5
  ls -la "$F"; disk
  upload "$F"
  rm -f "$F" "$F.sha256"
done
if [ "$UPLOAD_Q8" = "1" ]; then upload "$Q8"; fi

# --- manifest ----------------------------------------------------------------------
{
  echo "source: $SRC_PREFIX"
  echo "llama.cpp: $COMMIT"
  echo "converter: convert_hf_to_gguf.py --outtype q8_0 --no-mtp"
  echo "k-quants: llama-quantize --allow-requantize from q8_0: $QUANTS"
  echo "tokenizer_config.json: $BASE_PREFIX (transformers 4.57 compat); tokenizer.json, chat_template.jinja: source"
  echo "finished: $(date -u +%FT%TZ)"
} > "$OUT/MANIFEST.txt"
aws s3 cp "$OUT/MANIFEST.txt" "$DST_PREFIX/MANIFEST.txt" --only-show-errors
aws s3 cp "$OUT/MANIFEST.txt" "$DST_PREFIX/_COMPLETE" --only-show-errors
log "done"; aws s3 ls "$DST_PREFIX/"
