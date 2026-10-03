#!/bin/bash
# Runs ON gx10, detached. Stops the vLLM container (it holds the GPU), serves the Q5_K_S GGUF
# with llama-server (sm_121 build in ~/llama.cpp), runs the loop test, then restores vLLM.
set -u
M=$HOME/models/glm46-pruned-Q5_K_S.gguf
TEST=${1:-$HOME/w4a16_serve_test.py}        # top_p-only variant: the 7/12 baseline config
OUT=$HOME/glm46_q5ks_test.jsonl
LOG=/tmp/llama_q5ks.log
echo "START $(date -u)"
docker stop vllm-glm46-full >/dev/null 2>&1 && echo "stopped vllm-glm46-full"
sleep 5
setsid $HOME/llama.cpp/build/bin/llama-server -m "$M" --alias glm46 -c 32768 -ngl 999 -fa on \
  -ctk q8_0 -ctv q8_0 -np 1 --no-mmap --host 0.0.0.0 --port 8011 --jinja > $LOG 2>&1 < /dev/null &
echo $! > /tmp/llama_q5ks.pid
for i in $(seq 1 180); do curl -sf http://127.0.0.1:8011/health >/dev/null 2>&1 && break; sleep 5; done
if ! curl -sf http://127.0.0.1:8011/health >/dev/null 2>&1; then
  echo "SERVER_NOT_UP after $((i*5))s"; tail -30 $LOG; kill $(cat /tmp/llama_q5ks.pid) 2>/dev/null
  docker start vllm-glm46-full >/dev/null 2>&1; echo "TESTDONE rc=3 $(date -u)"; exit 3
fi
echo "server up after $((i*5))s"
grep -E "load_tensors:|KV self size|n_ctx |compute buffer|flash_attn" $LOG | tail -8
sed -e 's/timeout=600/timeout=1800/' "$TEST" > /tmp/loop_test.py
W4A16_BASE=http://127.0.0.1:8011/v1 W4A16_MODEL=glm46 W4A16_OUT=$OUT python3 /tmp/loop_test.py; rc=$?
echo "test rc=$rc"
kill $(cat /tmp/llama_q5ks.pid) 2>/dev/null; sleep 5
docker start vllm-glm46-full >/dev/null 2>&1 && echo "restarted vllm-glm46-full"
echo "TESTDONE rc=$rc $(date -u)"
