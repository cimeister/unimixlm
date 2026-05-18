#!/usr/bin/env bash
# _supervisor3.sh — drive the Phase-3 sweep to completion WITHOUT ever
# colliding with the live Phase-1 OR Phase-2 runs. Gate: Phase-3 only
# starts once BOTH prior phases are fully over (no orchestrator/
# supervisor/run_cell procs for phase 1 or 2) and every GPU is free.
# Then it (re)launches _orchestrator3.sh until all TARGET cells are
# completed. Resumable: completed / already-on-HF cells skip.
# Regexes use \. so they match _orchestrator.sh / _orchestrator2.sh but
# NOT this _supervisor3.sh / _run_cell3.sh / _orchestrator3.sh.
set -uo pipefail
cd /workspace/unimixlm
source ./_driver_env.sh
TARGET="${TARGET:-6}"
MAX_PASSES="${MAX_PASSES:-6}"
POLL="${POLL:-30}"

p3_done() {
  grep -l '"phase":"3"' "$RUNS"/*.json 2>/dev/null \
    | xargs -r grep -l '"status":"completed"' 2>/dev/null | wc -l
}
gpus_free() {
  local n
  n="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)"
  [ "${n:-0}" -eq 0 ]
}
prior_over() {
  ! pgrep -f '_orchestrator2?\.sh' >/dev/null 2>&1 \
    && ! pgrep -f '_supervisor2?\.sh' >/dev/null 2>&1 \
    && ! pgrep -f '_run_cell2?\.sh'   >/dev/null 2>&1 \
    && ! pgrep -f '_pack_handoff\.sh' >/dev/null 2>&1
}

echo "[supervisor3][$(date -u +%T)] waiting for Phase-1 + Phase-2 to release the GPUs..."
while true; do
  if prior_over && gpus_free; then
    echo "[supervisor3][$(date -u +%T)] prior phases done + GPUs free -> starting Phase-3"
    break
  fi
  sleep "$POLL"
done

pass=0
while true; do
  done_n="$(p3_done)"
  echo "[supervisor3][$(date -u +%T)] phase-3 completed ${done_n}/${TARGET}, pass ${pass}/${MAX_PASSES}"
  if [ "$done_n" -ge "$TARGET" ]; then
    echo "[supervisor3] ALL ${TARGET} PHASE-3 CELLS COMPLETED"; break; fi
  if pgrep -f '_orchestrator3\.sh' >/dev/null 2>&1; then
    sleep 120; continue; fi
  if [ "$pass" -ge "$MAX_PASSES" ]; then
    echo "[supervisor3] MAX_PASSES reached; stopping. Inspect runs/ + logs."; break; fi
  pass=$((pass+1))
  echo "[supervisor3][$(date -u +%T)] launching orchestrator3 pass ${pass}"
  bash _orchestrator3.sh 2>&1 | tee -a "$SHM/logs/phase3.log"
done
for f in "$RUNS"/*.json; do [ -e "$f" ] && grep -q '"phase":"3"' "$f" && \
  python3 -c "import json;d=json.load(open('$f'));print(d['status'],d['tag'])"; done 2>/dev/null
