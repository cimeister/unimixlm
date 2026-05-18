#!/usr/bin/env bash
# _pack_handoff.sh — one-shot: wait until Phase-1 fully releases GPU0,1,
# then launch Phase-2 cell #1 (r64-k128) there as 2-GPU DDP so it runs
# CONCURRENTLY with the early-packed cell #0 (r16-k32) on GPU2,3.
# Gate: no Phase-1 orch/supervisor procs AND GPU0+GPU1 effectively empty
# (cell #0 lives on GPU2,3, so GPU0/1 memory reflects only Phase-1).
set -uo pipefail
cd /workspace/unimixlm
source ./_driver_env.sh
POLL="${POLL:-30}"
echo "[handoff][$(date -u +%T)] waiting for Phase-1 to release GPU0,1..."
while true; do
  p1=0
  pgrep -f '_orchestrator\.sh' >/dev/null 2>&1 && p1=1
  pgrep -f '_supervisor\.sh'   >/dev/null 2>&1 && p1=1
  m0="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -dc 0-9)"
  m1="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr -dc 0-9)"
  if [ "$p1" -eq 0 ] && [ "${m0:-9999}" -lt 2000 ] && [ "${m1:-9999}" -lt 2000 ]; then
    echo "[handoff][$(date -u +%T)] Phase-1 done, GPU0,1 free (m0=${m0} m1=${m1}) -> launching cell #1 r64-k128 on 0,1"
    break
  fi
  sleep "$POLL"
done
export NUM_CKPT=10
exec bash _run_cell2.sh "0,1" qwen2_5-0_5b tur 500M r64-k128
