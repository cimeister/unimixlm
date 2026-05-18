#!/usr/bin/env bash
# _orchestrator4.sh — Phase-4: qwen2_5-0_5b base-vs-LangMAP.
# qwen2_5-0_5b × {tur,fin,hun} × {original,langspec}, 100M, r16-k32,
# save_embeddings, 20 weights-only ckpts (+final-full). 6 cells.
# Same machinery as Phase-3: 2 workers each a 2-GPU pair ("0,1","2,3")
# -> 2 concurrent 2-GPU-DDP cells, all 4 H200s busy, 6/2 = 3 even
# waves, no mid-run idle. Order pairs each lang's original+langspec
# adjacently so a complete base-vs-LangMAP pair lands as early as
# possible. Identical training setup to Phase-1/2/3 baselines (only
# las_mode/las_tokenizer_path differ across the two arms). The 0.5b
# langspec toks are byte-identical to the 1.5B ones (Qwen2.5 shares one
# tokenizer across sizes; LAS overlay is size-independent) -> reused,
# no retraining needed.
set -uo pipefail
cd /workspace/unimixlm
source ./_driver_env.sh
export NUM_CKPT=20
MODEL=qwen2_5-0_5b; SIZE=100M; LORA=r16-k32

QUEUE=(
  "$MODEL:tur:$SIZE:$LORA:original"
  "$MODEL:tur:$SIZE:$LORA:langspec"
  "$MODEL:fin:$SIZE:$LORA:original"
  "$MODEL:fin:$SIZE:$LORA:langspec"
  "$MODEL:hun:$SIZE:$LORA:original"
  "$MODEL:hun:$SIZE:$LORA:langspec"
)
printf '%s\n' "${QUEUE[@]}" > "$SHM/queue4.txt"
IDX="$SHM/queue4.idx"; echo 0 > "$IDX"
LOCK="$SHM/queue4.lock"

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
    IFS=':' read -r model lang size lora arm <<< "$cell"
    echo "[orch4][$(date -u +%T)] GPU$gpu -> cell #$i ${model}/${lang}/${arm}"
    bash _run_cell4.sh "$gpu" "$model" "$lang" "$size" "$lora" "$arm" || true
  done
  echo "[orch4][$(date -u +%T)] GPU$gpu drained, exiting"
}

for pair in "0,1" "2,3"; do worker "$pair" & done
wait
echo "[orch4][$(date -u +%T)] ALL PHASE-4 CELLS PROCESSED"
