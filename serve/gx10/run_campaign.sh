#!/bin/bash
# Runs ON gx10, detached. Runs run_full_test.sh for each configuration line of a plan file, one after another,
# keeps vLLM stopped between runs and restores it after the last one. Plan line format (| separated):
#   TAG|CTX|CTK|CTV|PHASES|LONG_TOKENS|BENCH_SHORT|SPEC|EXTRA
# Lines starting with # are skipped. Summary of every run goes to ~/glm46_runs/campaign.log.
set -u
M=${M:?model path}; PLAN=${1:?plan file}
SUM=$HOME/glm46_runs/campaign.log; mkdir -p "$HOME/glm46_runs"
mapfile -t LINES < <(grep -vE '^\s*(#|$)' "$PLAN")
n=${#LINES[@]}; i=0
echo "CAMPAIGN START $(date -u) model=$(basename "$M") runs=$n" | tee -a "$SUM"
for line in "${LINES[@]}"; do
  i=$((i+1))
  IFS='|' read -r TAG CTX CTK CTV PHASES LONG BSHORT SPEC EXTRA <<< "$line"
  RV=0; [ "$i" -eq "$n" ] && RV=1
  M="$M" TAG="$TAG" CTX="$CTX" CTK="$CTK" CTV="$CTV" PHASES="$PHASES" LONG_TOKENS="${LONG:-0}" \
    BENCH_SHORT="${BSHORT:-1}" SPEC="$SPEC" EXTRA="$EXTRA" RESTORE_VLLM=$RV \
    bash "$HOME/run_full_test.sh" > "/tmp/test_$TAG.log" 2>&1
  {
    echo "== [$i/$n] $TAG ctx=$CTX k=$CTK v=$CTV spec='$SPEC' extra='$EXTRA'"
    grep -E "server up|SERVER_NOT_UP|WATCHDOG|mem after load|mem after bench|props:|\[test\] DONE|^\[[a-z0-9]+ +t=|mean decode|TESTDONE" "/tmp/test_$TAG.log" | cut -c1-200
  } >> "$SUM"
done
echo "CAMPAIGN DONE $(date -u)" | tee -a "$SUM"
