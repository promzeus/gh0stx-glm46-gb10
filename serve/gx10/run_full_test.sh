#!/bin/bash
# Runs ON gx10, detached. Serves a GGUF with llama-server, records memory and the context llama-server settled on,
# runs the loop test and/or the speed bench, stops the server. One run = one configuration; the env selects it.
#   M        model path (required)
#   TAG      run name; outputs go to ~/glm46_runs/$TAG, server log to /tmp/llama_$TAG.log
#   CTX      context size, or "auto" to let --fit pick the largest context that fits, leaving FIT_MARGIN MiB
#            (default 8192) free. The box has unified memory: --fit's default 1 GiB margin starved the OS on
#            2026-10-03 and the box stopped answering ping.
#   MIN_AVAIL_MIB  watchdog floor (default 3072): llama-server is killed when MemAvailable drops below it
#   CTK/CTV  KV cache types (q8_0, q4_0, f16, ...)
#   SPEC     extra speculative flags, e.g. "--spec-type draft-mtp --spec-draft-n-max 3"
#   EXTRA    other llama-server flags, e.g. "-ub 2048"
#   PHASES   comma list: loop, bench
#   LONG_TOKENS  approximate size of the long-context bench case (0 = skip)
set -u
M=${M:?model path}; TAG=${TAG:-run}; CTX=${CTX:-32768}; CTK=${CTK:-q8_0}; CTV=${CTV:-q8_0}; SPEC=${SPEC:-}; EXTRA=${EXTRA:-}
PORT=${PORT:-8011}; PHASES=${PHASES:-loop,bench}; LONG_TOKENS=${LONG_TOKENS:-0}
FIT_MARGIN=${FIT_MARGIN:-8192}; MIN_AVAIL_MIB=${MIN_AVAIL_MIB:-3072}
BIN=$HOME/llama.cpp/build/bin/llama-server
LOG=/tmp/llama_$TAG.log; OUTD=$HOME/glm46_runs/$TAG; mkdir -p "$OUTD"
CTXARG=(-c "$CTX"); [ "$CTX" = auto ] && CTXARG=(-fitt "$FIT_MARGIN")
echo "START $TAG $(date -u) model=$(basename "$M") ctx=$CTX k=$CTK v=$CTV spec='$SPEC' extra='$EXTRA'"
free -m | awk '/Mem:/{print "mem before load: used "$3" MiB, available "$7" MiB"}'
# shellcheck disable=SC2086
setsid "$BIN" -m "$M" --alias glm46 "${CTXARG[@]}" -ngl 999 -fa on -ctk "$CTK" -ctv "$CTV" -np 1 \
  --load-mode none --host 0.0.0.0 --port "$PORT" --jinja $SPEC $EXTRA > "$LOG" 2>&1 < /dev/null &
echo $! > /tmp/llama_$TAG.pid
# Watchdog: unified memory has no separate VRAM, an oversized KV or compute buffer eats the OS's memory.
( SPID=$(cat /tmp/llama_$TAG.pid)
  while kill -0 "$SPID" 2>/dev/null; do
    AV=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$AV" -lt "$MIN_AVAIL_MIB" ]; then
      echo "WATCHDOG_KILL $(date -u) MemAvailable ${AV} MiB < ${MIN_AVAIL_MIB}" | tee -a "$LOG"; kill -9 "$SPID"; break
    fi
    sleep 2
  done ) &
for i in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$(cat /tmp/llama_$TAG.pid)" 2>/dev/null || break; sleep 5; done
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "SERVER_NOT_UP after $((i*5))s"; tail -40 "$LOG"; kill "$(cat /tmp/llama_$TAG.pid)" 2>/dev/null
  echo "TESTDONE rc=3 $(date -u)"; exit 3
fi
echo "server up after $((i*5))s"
free -m | awk '/Mem:/{print "mem after load: used "$3" MiB, available "$7" MiB"}'
grep -iE "n_ctx|fit|kv|KV|buffer size|spec|draft|mtp|nextn" "$LOG" | grep -vE "slot|task" | tail -25 | cut -c1-200
curl -s "http://127.0.0.1:$PORT/props" | python3 -c "import sys,json; j=json.load(sys.stdin); \
  print('props: n_ctx', j.get('default_generation_settings',{}).get('n_ctx'), 'model', j.get('model_path','')[-60:])" 2>/dev/null
rc=0
case ",$PHASES," in *,loop,*)
  sed -e 's/timeout=600/timeout=3600/' "$HOME/loop_test.py" > /tmp/loop_test_$TAG.py
  LOOP_BASE=http://127.0.0.1:$PORT/v1 LOOP_MODEL=glm46 LOOP_OUT=$OUTD/loop.jsonl python3 /tmp/loop_test_$TAG.py; rc=$?
  echo "loop rc=$rc";;
esac
case ",$PHASES," in *,bench,*)
  [ -f /tmp/long_doc.txt ] || cat "$HOME"/llama.cpp/docs/*.md "$HOME"/llama.cpp/docs/*/*.md "$HOME"/llama.cpp/tools/*/README.md > /tmp/long_doc.txt 2>/dev/null
  BENCH_BASE=http://127.0.0.1:$PORT/v1 BENCH_OUT=$OUTD/bench.jsonl LONG_FILE=/tmp/long_doc.txt LONG_TOKENS=$LONG_TOKENS BENCH_SHORT=${BENCH_SHORT:-1} \
    python3 "$HOME/spec_bench.py"
  free -m | awk '/Mem:/{print "mem after bench: used "$3" MiB, available "$7" MiB"}';;
esac
grep -E "draft acceptance|accept|n_draft|spec" "$LOG" | tail -5 | cut -c1-200
kill "$(cat /tmp/llama_$TAG.pid)" 2>/dev/null; sleep 8
cp "$LOG" "$OUTD/server.log"
echo "TESTDONE rc=$rc $(date -u)"
