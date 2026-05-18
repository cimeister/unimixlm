#!/usr/bin/env bash
# _supervisor2.sh — drive the Phase-2 sweep to completion WITHOUT ever
# colliding with the live Phase-1 run.
#
# Gate: Phase-2 only starts once Phase-1 is fully over (its supervisor
# AND orchestrator processes are gone) and every GPU is free. Phase-1's
# orchestrator `wait`s on all workers incl. uploads before it exits, so
# "no _orchestrator.sh + no _supervisor.sh + no GPU compute procs" means
# Phase-1 has released the hardware. Then it (re)launches
# _orchestrator2.sh until all Phase-2 cells (TARGET) are completed.
# Resumable:
# completed / already-on-HF cells skip; bounded passes.
set -uo pipefail
cd /workspace/unimixlm
source ./_driver_env.sh
TARGET=2
MAX_PASSES="${MAX_PASSES:-6}"
POLL="${POLL:-30}"

p2_done() {
  grep -l '"phase":"2"' "$RUNS"/*.json 2>/dev/null \
    | xargs -r grep -l '"status":"completed"' 2>/dev/null | wc -l
}
gpus_free() {  # 0 = all free
  local n
  n="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)"
  [ "${n:-0}" -eq 0 ]
}
phase1_over() {
  ! pgrep -f '_orchestrator\.sh' >/dev/null 2>&1 \
    && ! pgrep -f '_supervisor\.sh' >/dev/null 2>&1
}

echo "[supervisor2][$(date -u +%T)] waiting for Phase-1 to release the GPUs..."
while true; do
  if phase1_over && gpus_free; then
    echo "[supervisor2][$(date -u +%T)] Phase-1 done + GPUs free -> starting Phase-2"
    break
  fi
  sleep "$POLL"
done

pass=0
while true; do
  done_n="$(p2_done)"
  echo "[supervisor2][$(date -u +%T)] phase-2 completed ${done_n}/${TARGET}, pass ${pass}/${MAX_PASSES}"
  if [ "$done_n" -ge "$TARGET" ]; then
    echo "[supervisor2] ALL ${TARGET} PHASE-2 CELLS COMPLETED"; break; fi
  if pgrep -f '_orchestrator2\.sh' >/dev/null 2>&1; then
    sleep 120; continue; fi
  if [ "$pass" -ge "$MAX_PASSES" ]; then
    echo "[supervisor2] MAX_PASSES reached; stopping. Inspect runs/ + logs."; break; fi
  pass=$((pass+1))
  echo "[supervisor2][$(date -u +%T)] launching orchestrator2 pass ${pass}"
  bash _orchestrator2.sh 2>&1 | tee -a "$SHM/logs/phase2.log"
done
for f in "$RUNS"/*.json; do [ -e "$f" ] && grep -q '"phase":"2"' "$f" && \
  python3 -c "import json;d=json.load(open('$f'));print(d['status'],d['tag'])"; done 2>/dev/null
