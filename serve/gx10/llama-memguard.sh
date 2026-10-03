#!/bin/bash
# Memory guard for llama-glm46 on gx10. GPU allocations on GB10 come out of the same RAM as the OS; when
# MemAvailable runs low the box can stop answering (NVRM OOM) and needs a physical reset. Killing the server
# instead costs one request: systemd restarts llama-glm46 (Restart=on-failure).
MIN_MIB=${MIN_MIB:-2048}
while sleep 2; do
  av=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$av" -lt "$MIN_MIB" ] && systemctl is-active --quiet llama-glm46; then
    logger -t llama-memguard "MemAvailable ${av} MiB < ${MIN_MIB}, killing llama-glm46"
    systemctl kill -s KILL llama-glm46
    sleep 60
  fi
done
