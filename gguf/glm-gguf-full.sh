#!/usr/bin/env bash
# GLM-4.6 full bf16 (abliterated, HF safetensors, 705 GB) + stock MTP layer 92 -> GGUF q8_0 -> imatrix
# -> trunk experts IQ2_XXS, MTP experts Q4_K, everything else Q5_K -> S3.
# CPU-only node with >= 768 GiB RAM (the 375 GB q8_0 stays in page cache during imatrix) and a 1500Gi disk.
# Resumable across spot retries: the q8_0 GGUF, a partial imatrix (every ~10 min) and the final imatrix
# go to S3 as soon as they exist and are reused by the next attempt.
# Disk on /data: 705 GB hf + ~380 GB q8_0 at peak; hf is removed right after conversion.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore
ulimit -c 0   # a crash with the 375 GB q8_0 mapped spent 5 min writing a core dump

: "${SRC_PREFIX:?}" "${BASE_PREFIX:?}" "${DST_PREFIX:?}" "${CALIB_URI:?}"
LLAMA_COMMIT="${LLAMA_COMMIT:-4ebdf2c74acce30883d8e34b7c70b3eb8146f2fe}"   # same build as llama-server on gx10
WITH_MTP="${WITH_MTP:-1}"
IMATRIX_TOKENS="${IMATRIX_TOKENS:-400000}"
IMATRIX_CTX="${IMATRIX_CTX:-2048}"
IMATRIX_SHARES="${IMATRIX_SHARES:-reasoning=0.25 math=0.15 code_cot=0.15 ru=0.20 general=0.15 code=0.10}"
BASE_FTYPE="${BASE_FTYPE:-IQ2_XXS}"
EXPERT_TYPE="${EXPERT_TYPE:-iq2_xxs}"
MTP_EXPERT_TYPE="${MTP_EXPERT_TYPE:-q4_k}"
REST_TYPE="${REST_TYPE:-q5_k}"
OUT_NAME="${OUT_NAME:-glm46-abl-mtp-IQ2_XXS-Q5K}"
THREADS="$(nproc)"
CORES="$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)"
NUMA_NODES="$(lscpu | awk -F: '/^NUMA node\(s\)/{gsub(/ /,"",$2); print $2}')"
WORK=/data; HF=$WORK/hf; TOK=$WORK/tok; OUT=$WORK/out; LOGS=$OUT/logs; LLAMA=/opt/llama.cpp
Q8=$OUT/glm46-abl-mtp-q8_0.gguf
IMX=$OUT/imatrix-glm46-abl-q8_0-c${IMATRIX_CTX}.gguf
PART_KEY="$DST_PREFIX/imatrix-partial.gguf"
mkdir -p "$HF" "$TOK" "$OUT" "$LOGS"

log()  { echo "[$(date -u +%FT%TZ)] $*"; }
disk() { df -h "$WORK" | tail -1; }
# aws s3 ls exits 1 when nothing matches; any other non-zero code is an S3/network error and is retried,
# so a transient failure does not look like "absent" and trigger a full re-download and re-convert.
have() {
  local rc=0
  for _ in 1 2 3 4 5; do
    if aws s3 ls "$1" >/dev/null 2>&1; then return 0; else rc=$?; fi
    [ "$rc" -eq 1 ] && return 1
    sleep 20
  done
  log "have(): aws s3 ls $1 failed with rc=$rc 5 times"; exit 1
}
upload() {
  local f="$1" d b; d="$(dirname "$f")"; b="$(basename "$f")"
  log "sha256 $b"; (cd "$d" && sha256sum "$b" > "$b.sha256")
  log "upload $b ($(du -h "$f" | cut -f1))"
  aws s3 cp "$f" "$DST_PREFIX/$b" --only-show-errors
  aws s3 cp "$f.sha256" "$DST_PREFIX/$b.sha256" --only-show-errors
  aws s3 ls "$DST_PREFIX/$b"
}
push_logs() { aws s3 cp "$LOGS/" "$DST_PREFIX/logs/" --recursive --only-show-errors || true; }
UPLOADER=""
cleanup() { local rc=$?; [ -n "$UPLOADER" ] && kill "$UPLOADER" 2>/dev/null; log "exit rc=$rc"; push_logs; }
trap cleanup EXIT

log "node: ${THREADS} vcpu, ${CORES} cores, ${NUMA_NODES} numa, $(free -g | awk '/Mem:/{print $2}') GiB ram"; disk

# --- tools ---------------------------------------------------------------------
log "apt: cmake unzip"
apt-get update -qq >/dev/null && apt-get install -y -qq cmake unzip >/dev/null
log "awscli v2 (EKS Pod Identity)"
curl -sSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscli.zip
unzip -q /tmp/awscli.zip -d /tmp && /tmp/aws/install >/dev/null && rm -rf /tmp/awscli.zip /tmp/aws
aws configure set default.s3.max_concurrent_requests 32
aws configure set default.s3.max_queue_size 10000
aws configure set default.s3.multipart_chunksize 128MB
aws sts get-caller-identity --query Arn --output text
aws s3 ls "$SRC_PREFIX/" >/dev/null

# --- llama.cpp at a pinned commit ---------------------------------------------------
log "llama.cpp $LLAMA_COMMIT"
mkdir -p "$LLAMA" && cd "$LLAMA"
git init -q && git remote add origin https://github.com/ggml-org/llama.cpp
git fetch -q --depth 1 origin "$LLAMA_COMMIT" && git checkout -q FETCH_HEAD
log "pip: converter deps (torch cpu)"
pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install -q -r requirements/requirements-convert_hf_to_gguf.txt
log "build llama-quantize llama-imatrix"
cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF >/dev/null
cmake --build build --target llama-quantize llama-imatrix -j"$THREADS" >/dev/null
ls -la build/bin/llama-quantize build/bin/llama-imatrix

# --- small files; tokenizer_config.json from the stock model -------------------
# The abliterated dir was saved by transformers 5.x (tokenizer_class TokenizersBackend), unknown to the
# transformers version pinned by the converter. The stock file loads the same tokenizer.json.
log "download small files"
aws s3 cp "$SRC_PREFIX/" "$HF/" --recursive --exclude "*.safetensors*" --only-show-errors
aws s3 cp "$BASE_PREFIX/tokenizer_config.json" "$HF/tokenizer_config.json" --only-show-errors
cp "$HF"/config.json "$HF"/tokenizer.json "$HF"/tokenizer_config.json "$HF"/chat_template.jinja "$TOK/"
ls -la "$HF"
python3 - <<'PY'
import json
c = json.load(open('/data/hf/config.json'))
assert c['n_routed_experts'] == 160, c['n_routed_experts']
assert c['num_hidden_layers'] == 92, c['num_hidden_layers']
print('config ok: experts', c['n_routed_experts'], 'layers', c['num_hidden_layers'],
      'nextn', c.get('num_nextn_predict_layers'), 'ctx', c.get('max_position_embeddings'))
PY

# --- vocab-only pre-flight, before the 705 GB download -------------------------
log "pre-flight: vocab-only"
python3 convert_hf_to_gguf.py "$HF" --vocab-only --no-mtp --outfile "$OUT/preflight-vocab.gguf"
rm -f "$OUT/preflight-vocab.gguf"

# --- q8_0 trunk (+ MTP layer 92): reuse from S3 or convert -------------------------------
if have "$DST_PREFIX/$(basename "$Q8")"; then
  log "q8_0 found in S3, downloading"
  aws s3 cp "$DST_PREFIX/$(basename "$Q8")" "$Q8" --only-show-errors; ls -la "$Q8"; disk
else
  log "download shards"; disk
  aws s3 cp "$SRC_PREFIX/" "$HF/" --recursive --exclude "*" --include "*.safetensors" \
    --include "model.safetensors.index.json" --only-show-errors
  log "downloaded: $(du -sh "$HF" | cut -f1)"; disk
  MTP_FLAG="--no-mtp"
  if [ "$WITH_MTP" = 1 ]; then
    # The abliterated checkpoint has no layer 92; the stock MTP layer is merged into the HF index so the
    # converter writes it as blk.92 (nextn). llama.cpp loads it only with --spec-type draft-mtp.
    log "merge stock MTP layer from $BASE_PREFIX/mtp.safetensors"
    aws s3 cp "$BASE_PREFIX/mtp.safetensors" "$HF/mtp.safetensors" --only-show-errors
    python3 - <<'PY'
import json
from safetensors import safe_open
idx = '/data/hf/model.safetensors.index.json'; j = json.load(open(idx))
with safe_open('/data/hf/mtp.safetensors', 'pt') as f: keys = list(f.keys())
l92 = [k for k in keys if k.startswith('model.layers.92.')]
other = [k for k in keys if not k.startswith('model.layers.92.')]
print('mtp.safetensors:', len(keys), 'tensors,', len(l92), 'in layer 92, other:', other[:5])
assert len(l92) == len(keys), 'unexpected non-layer-92 tensors in mtp.safetensors'
clash = [k for k in keys if k in j['weight_map']]
assert not clash, f'layer-92 tensors already in the index: {clash[:3]}'
for k in keys: j['weight_map'][k] = 'mtp.safetensors'
json.dump(j, open(idx, 'w'))
print('index now', len(j['weight_map']), 'tensors')
PY
    MTP_FLAG=""
  fi
  log "convert -> q8_0 ${MTP_FLAG:-(with MTP layer)}"
  python3 convert_hf_to_gguf.py "$HF" --outtype q8_0 $MTP_FLAG --outfile "$Q8" > "$LOGS/convert-q8_0.log" 2>&1 \
    || { tail -20 "$LOGS/convert-q8_0.log"; exit 1; }
  tail -3 "$LOGS/convert-q8_0.log"; ls -la "$Q8"; disk
  python3 - "$Q8" <<'PY' || log "q8_0 inspection failed (non-fatal)"
import sys, gguf
r = gguf.GGUFReader(sys.argv[1])
names = [t.name for t in r.tensors]
kv = {f.name: f for f in r.fields.values()}
nl = [k for k in kv if k.endswith('.nextn_predict_layers')]
val = int(kv[nl[0]].parts[kv[nl[0]].data[0]][0]) if nl else None
print('q8_0 tensors', len(names), 'blk.92 tensors', sum(n.startswith('blk.92.') for n in names), 'nextn_predict_layers', val)
PY
  upload "$Q8"
  log "rm hf input"; rm -rf "$HF"; disk
fi

# --- imatrix: reuse from S3, resume a partial, or compute on the mixed calibration corpus ---------
if have "$DST_PREFIX/$(basename "$IMX")"; then
  log "imatrix found in S3, downloading"
  aws s3 cp "$DST_PREFIX/$(basename "$IMX")" "$IMX" --only-show-errors
else
  log "calibration corpus $CALIB_URI (waiting up to 2h for the calib job)"
  for _ in $(seq 1 120); do have "$CALIB_URI" && break; sleep 60; done
  aws s3 cp "$CALIB_URI" "$OUT/calib.jsonl" --only-show-errors
  # Deterministic text (same corpus, same quotas) so a resumed run can skip the chunks already done.
  IMATRIX_TOKENS="$IMATRIX_TOKENS" IMATRIX_SHARES="$IMATRIX_SHARES" IMATRIX_CTX="$IMATRIX_CTX" python3 - <<'PY'
import json, os, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/data/tok')
budget = int(os.environ['IMATRIX_TOKENS'])
shares = {k: float(v) for k, v in (kv.split('=') for kv in os.environ['IMATRIX_SHARES'].split())}
quota = {d: int(budget * s) for d, s in shares.items()}
used, rows_by = collections.Counter(), collections.Counter(); texts = []
for line in open('/data/out/calib.jsonl'):
    r = json.loads(line); d = r['domain']
    if d not in quota or used[d] >= quota[d]:
        continue
    used[d] += len(tok(r['text'], add_special_tokens=False).input_ids); rows_by[d] += 1; texts.append(r['text'])
open('/data/out/imatrix.txt', 'w').write('\n\n'.join(texts))
total = sum(used.values())
open('/data/out/imatrix.chunks', 'w').write(str(total // int(os.environ['IMATRIX_CTX'])))
print(f'imatrix text: {len(texts)} rows, {total} tokens (~{total // int(os.environ["IMATRIX_CTX"])} chunks)')
for d in quota:
    print(f'  {d:10s} {used[d]:7d} / {quota[d]:7d} tokens, {rows_by[d]} rows' + ('  SHORT' if used[d] < quota[d] * 0.9 else ''))
PY
  TOTAL_CHUNKS="$(cat "$OUT/imatrix.chunks")"
  RESUME=()
  if have "$PART_KEY"; then
    aws s3 cp "$PART_KEY" "$OUT/imatrix-prev.gguf" --only-show-errors
    DONE_CHUNKS="$(python3 - "$OUT/imatrix-prev.gguf" <<'PY' || echo 0
import sys, gguf
r = gguf.GGUFReader(sys.argv[1])
f = r.fields.get('imatrix.chunk_count')
print(int(f.parts[f.data[0]][0]) if f else 0)
PY
)"
    log "partial imatrix in S3: $DONE_CHUNKS of $TOTAL_CHUNKS chunks"
    if [ "$DONE_CHUNKS" -gt 0 ] && [ $(( DONE_CHUNKS + 2 )) -lt "$TOTAL_CHUNKS" ]; then
      RESUME=( --in-file "$OUT/imatrix-prev.gguf" --chunk "$DONE_CHUNKS" )
    elif [ "$DONE_CHUNKS" -gt 0 ]; then
      log "partial covers the text, using it as the final imatrix"; cp "$OUT/imatrix-prev.gguf" "$IMX"
    fi
  fi
  if [ ! -f "$IMX" ]; then
    NUMA=(); [ "${NUMA_NODES:-1}" -gt 1 ] && NUMA=( --numa distribute )
    # Snapshot the -o file to S3 every 10 min; a torn copy is rejected by the GGUF reader on resume.
    ( set +e +o pipefail
      while sleep 600; do
        if [ -f "$IMX" ] && cp "$IMX" "$OUT/imatrix-snap.gguf" \
           && python3 -c "import gguf,sys; gguf.GGUFReader(sys.argv[1])" "$OUT/imatrix-snap.gguf" 2>/dev/null; then
          aws s3 cp "$OUT/imatrix-snap.gguf" "$PART_KEY" --only-show-errors
        fi
        echo "[$(date -u +%FT%TZ)] imatrix progress: $(tail -c 300 "$LOGS/imatrix.log" 2>/dev/null | tr ',' '\n' | grep -E '^\[[0-9]+\]' | tail -1)"
      done ) &
    UPLOADER=$!
    log "llama-imatrix: ctx $IMATRIX_CTX, $CORES threads ${NUMA[*]} ${RESUME[*]}"
    # --no-repack (no_extra_bufts): with GGML_NATIVE on Xeon 6975P-C the AMX q8_0 kernel
    # (tinygemm_kernel_amx<block_q8_0>) dies on tileloadd with SIGILL 2 s into the first chunk.
    "$LLAMA/build/bin/llama-imatrix" -m "$Q8" -f "$OUT/imatrix.txt" -o "$IMX" -c "$IMATRIX_CTX" --no-repack \
      -t "$CORES" -tb "$CORES" "${NUMA[@]}" --parse-special --output-frequency 10 "${RESUME[@]}" \
      > "$LOGS/imatrix.log" 2>&1 || { tail -20 "$LOGS/imatrix.log"; exit 1; }
    kill "$UPLOADER" 2>/dev/null; UPLOADER=""
    grep -E "compute_imatrix:|save_imatrix:|Final estimate|storing only|partial data|no data" "$LOGS/imatrix.log" | tail -15
  fi
  ls -la "$IMX"
  upload "$IMX"
  aws s3 rm "$PART_KEY" --only-show-errors 2>/dev/null || true
fi

# --- quantize: first matching --tensor-type wins, so the MTP layer patterns go first --------------
F="$OUT/$OUT_NAME.gguf"
QARGS=( --imatrix "$IMX" --allow-requantize
        --token-embedding-type "$REST_TYPE" --output-tensor-type "$REST_TYPE"
        --tensor-type "^blk\.92\.ffn_(gate|up|down)_exps=$MTP_EXPERT_TYPE"
        --tensor-type "^blk\.92\.=$REST_TYPE"
        --tensor-type "ffn_(gate|up|down)_exps=$EXPERT_TYPE"
        --tensor-type "attn_(q|k|v|output)=$REST_TYPE"
        --tensor-type "ffn_(gate|up|down)_shexp=$REST_TYPE"
        --tensor-type "^blk\.[0-2]\.ffn_(gate|up|down)\.weight=$REST_TYPE" )
log "quantize dry-run"
"$LLAMA/build/bin/llama-quantize" --dry-run "${QARGS[@]}" "$Q8" "$F" "$BASE_FTYPE" "$THREADS" > "$LOGS/quantize-dryrun.log" 2>&1 || true
grep -E "model size|quant size" "$LOGS/quantize-dryrun.log" | tail -2 || true
# llama-quant prints the source type with %6s ("type =   q8_0"), hence " +" after "type =".
grep -oE "type = +[a-z0-9_A-Z]+, size = +[0-9.]+ MiB -> +[0-9.]+ MiB \([a-z0-9_A-Z]+\)" "$LOGS/quantize-dryrun.log" \
  | awk '{t[$NF]+=$(NF-2); n[$NF]++} END{for(k in t) printf "  planned %-10s %9.1f MiB  %d tensors\n", k, t[k], n[k]}' | sort -k3 -rn || true
log "quantize $BASE_FTYPE"
"$LLAMA/build/bin/llama-quantize" "${QARGS[@]}" "$Q8" "$F" "$BASE_FTYPE" "$THREADS" > "$LOGS/quantize.log" 2>&1 \
  || { tail -20 "$LOGS/quantize.log"; exit 1; }
grep -E "model size|quant size|total time" "$LOGS/quantize.log" | tail -4
echo "imatrix misses (expected for blk.92 only): $(grep -c 'did not find weights' "$LOGS/quantize.log" || true)"
grep 'did not find weights' "$LOGS/quantize.log" | grep -v 'blk\.92\.' | head -5 || true
ls -la "$F"; disk
upload "$F"

# --- manifest ----------------------------------------------------------------------
{
  echo "source: $SRC_PREFIX (trunk), $BASE_PREFIX/mtp.safetensors (MTP layer 92, WITH_MTP=$WITH_MTP)"
  echo "tokenizer_config.json: $BASE_PREFIX (converter transformers compat)"
  echo "llama.cpp: $LLAMA_COMMIT"
  echo "converter: convert_hf_to_gguf.py --outtype q8_0"
  echo "imatrix: llama-imatrix -c $IMATRIX_CTX --parse-special, $IMATRIX_TOKENS tokens of $CALIB_URI, shares: $IMATRIX_SHARES"
  echo "quantize: $BASE_FTYPE from q8_0; trunk experts $EXPERT_TYPE; blk.92 experts $MTP_EXPERT_TYPE; attn/shexp/dense/embd/output/blk.92 rest $REST_TYPE"
  echo "finished: $(date -u +%FT%TZ)"
} > "$OUT/MANIFEST.txt"
aws s3 cp "$OUT/MANIFEST.txt" "$DST_PREFIX/MANIFEST.txt" --only-show-errors
aws s3 cp "$OUT/MANIFEST.txt" "$DST_PREFIX/_COMPLETE" --only-show-errors
log "done"; aws s3 ls "$DST_PREFIX/"
