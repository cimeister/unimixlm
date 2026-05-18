#!/usr/bin/env bash
# _orchestrator2.sh — Phase-2 sweep: qwen2_5-0_5b, original arm,
# embeddings correctly saved. Cheapest LoRA sweep r16-k32 vs r64-k128,
# token axis 100M & 500M. Longest cells (500M ~9.5h) first so the short
# 100M cells backfill the tail and all 4 H200s stay busy.
#   cell = MODEL:LANG:SIZE:LORA   (LORA = hydra group name)
# Existing Phase-1 r16/100M repos are deliberately NOT re-run (user:
# only new LoRA points get 100M).
set -uo pipefail
cd /workspace/unimixlm
source ./_driver_env.sh

QUEUE=(
  "qwen2_5-0_5b:tur:500M:r16-k32"
  "qwen2_5-0_5b:fin:500M:r16-k32"
  "qwen2_5-0_5b:hun:500M:r16-k32"
  "qwen2_5-0_5b:tur:500M:r64-k128"
  "qwen2_5-0_5b:fin:500M:r64-k128"
  "qwen2_5-0_5b:hun:500M:r64-k128"
  "qwen2_5-0_5b:tur:100M:r64-k128"
  "qwen2_5-0_5b:fin:100M:r64-k128"
  "qwen2_5-0_5b:hun:100M:r64-k128"
)
printf '%s\n' "${QUEUE[@]}" > "$SHM/queue2.txt"
IDX="$SHM/queue2.idx"; echo 0 > "$IDX"
LOCK="$SHM/queue2.lock"

worker() {
  local gpu="$1"
  while true; do
    local i cell
    exec 9>"$LOCK"; flock 9
    i="$(cat "$IDX")"
    if [[ "$i" -ge "${#QUEUE[@]}" ]]; then flock -u 9; exec 9>&-; break; fi
    echo $((i+1)) > "$IDX"
    flock -u 9; exec 9>&-
    cell="${QUEUE[$i]}"
    IFS=':' read -r model lang size lora <<< "$cell"
    echo "[orch2][$(date -u +%T)] GPU$gpu -> cell #$i ${model}/${lang}/${size}/${lora}"
    bash _run_cell2.sh "$gpu" "$model" "$lang" "$size" "$lora" || true
  done
  echo "[orch2][$(date -u +%T)] GPU$gpu drained, exiting"
}

for g in 0 1 2 3; do worker "$g" & done
wait
echo "[orch2][$(date -u +%T)] ALL PHASE-2 CELLS PROCESSED"
