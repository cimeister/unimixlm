#!/usr/bin/env bash
# _supervisor4.sh — drive Phase-4 (qwen0.5b base-vs-LangMAP) to
# completion WITHOUT colliding with the live Phase-3 run (or any
# earlier phase). Gate: Phase-4 starts only once Phase-3 is fully over
# (no _orchestrator3.sh / _supervisor3.sh / _run_cell3.sh, no
# run_finetune) AND every GPU is free. Then it (re)launches
# _orchestrator4.sh until all TARGET cells complete. Resumable:
# completed / already-on-HF cells skip; bounded passes.
# Regexes use '3\.sh' (literal dot) so they match the Phase-3 procs but
# NEVER this _supervisor4.sh / _run_cell4.sh / _orchestrator4.sh.
set -uo pipefail
cd /workspace/unimixlm
source ./_driver_env.sh
TARGET="${TARGET:-6}"
MAX_PASSES="${MAX_PASSES:-6}"
POLL="${POLL:-30}"

p4_done() {
  grep -l '"phase":"4"' "$RUNS"/*.json 2>/dev/null \
    | xargs -r grep -l '"status":"completed"' 2>/dev/null | wc -l
}
gpus_free() {
  local n
  n="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)"
  [ "${n:-0}" -eq 0 ]
}
prior_over() {  # Phase-3 (and, transitively, all earlier) fully finished
  ! pgrep -f '_orchestrator3\.sh' >/dev/null 2>&1 \
    && ! pgrep -f '_supervisor3\.sh' >/dev/null 2>&1 \
    && ! pgrep -f '_run_cell3\.sh'   >/dev/null 2>&1 \
    && ! pgrep -f 'run_finetune\.py' >/dev/null 2>&1
}

echo "[supervisor4][$(date -u +%T)] waiting for Phase-3 to release the GPUs..."
while true; do
  if prior_over && gpus_free; then
    echo "[supervisor4][$(date -u +%T)] Phase-3 done + GPUs free -> starting Phase-4"
    break
  fi
  sleep "$POLL"
done

pass=0
while true; do
  done_n="$(p4_done)"
  echo "[supervisor4][$(date -u +%T)] phase-4 completed ${done_n}/${TARGET}, pass ${pass}/${MAX_PASSES}"
  if [ "$done_n" -ge "$TARGET" ]; then
    echo "[supervisor4] ALL ${TARGET} PHASE-4 CELLS COMPLETED"; break; fi
  if pgrep -f '_orchestrator4\.sh' >/dev/null 2>&1; then
    sleep 120; continue; fi
  if [ "$pass" -ge "$MAX_PASSES" ]; then
    echo "[supervisor4] MAX_PASSES reached; stopping. Inspect runs/ + logs."; break; fi
  pass=$((pass+1))
  echo "[supervisor4][$(date -u +%T)] launching orchestrator4 pass ${pass}"
  bash _orchestrator4.sh 2>&1 | tee -a "$SHM/logs/phase4.log"
done
for f in "$RUNS"/*.json; do [ -e "$f" ] && grep -q '"phase":"4"' "$f" && \
  python3 -c "import json;d=json.load(open('$f'));print(d['status'],d['tag'])"; done 2>/dev/null
