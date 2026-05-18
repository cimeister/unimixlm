#!/usr/bin/env bash
# _supervisor.sh — keep the 9-cell sweep progressing to 100% completion.
# Waits out the running orchestrator; whenever none is running and not
# all 9 cells are 'completed', relaunches it (resumable: completed/
# on-HF cells skip, failed cells — e.g. the patched gemma3 — rerun).
# Bounded attempts so a persistently-broken cell can't loop forever.
set -uo pipefail
cd /workspace/unimixlm
source ./_driver_env.sh
TARGET=9
MAX_PASSES="${MAX_PASSES:-6}"
pass=0
while true; do
  done_n=$(grep -l '"status":"completed"' "$RUNS"/*.json 2>/dev/null | wc -l)
  echo "[supervisor][$(date -u +%T)] completed ${done_n}/${TARGET}, pass ${pass}/${MAX_PASSES}"
  if [ "$done_n" -ge "$TARGET" ]; then
    echo "[supervisor] ALL ${TARGET} CELLS COMPLETED"; break; fi
  if pgrep -f '_orchestrator.sh' >/dev/null 2>&1; then
    sleep 120; continue; fi
  if [ "$pass" -ge "$MAX_PASSES" ]; then
    echo "[supervisor] MAX_PASSES reached; stopping. Inspect runs/ + logs."; break; fi
  pass=$((pass+1))
  echo "[supervisor][$(date -u +%T)] launching orchestrator pass ${pass}"
  bash _orchestrator.sh 2>&1 | tee -a "$SHM/logs/pipeline.log"
done
# final summary
for f in "$RUNS"/*.json; do [ -e "$f" ] && \
  python3 -c "import json;d=json.load(open('$f'));print(d['status'],d['tag'])"; done 2>/dev/null
