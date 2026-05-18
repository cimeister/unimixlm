#!/usr/bin/env bash
# _orchestrator2.sh — Phase-2 sweep (revised): qwen2_5-0_5b, original
# arm, embeddings correctly saved. ONE language (tur), 500M only, LoRA
# rank sweep r16-k32 vs r64-k128. 10 checkpoints over the 500M run =
# one every ~50M tokens (NUM_CKPT below; finetune.py spaces evenly).
#   cell = MODEL:LANG:SIZE:LORA   (LORA = hydra group name)
set -uo pipefail
cd /workspace/unimixlm
source ./_driver_env.sh

# 10 checkpoints across a single-epoch 500M run ≈ every 50M tokens.
export NUM_CKPT=10

QUEUE=(
  "qwen2_5-0_5b:tur:500M:r16-k32"
  "qwen2_5-0_5b:tur:500M:r64-k128"
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

# 2 workers, each owning a 2-GPU pair → both cells run as 2-GPU DDP
# concurrently, all 4 H200s busy (no idle GPUs).
for pair in "0,1" "2,3"; do worker "$pair" & done
wait
echo "[orch2][$(date -u +%T)] ALL PHASE-2 CELLS PROCESSED"
