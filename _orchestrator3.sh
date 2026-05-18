#!/usr/bin/env bash
# _orchestrator3.sh — Phase-3 LangMAP finetuning sweep.
# {qwen2_5-1_5b, gemma2-2b} × {tur,fin,hun} × langspec, 100M, r16-k32,
# save_embeddings, 10 ckpts. 6 cells. GUARANTEED FULL GPU UTILISATION:
# 2 workers each owning a 2-GPU pair ("0,1","2,3") run 2-GPU DDP cells
# concurrently -> all 4 H200s busy; 6 cells / 2 = 3 even waves, no
# mid-run idle (vs single-GPU x4 which would idle 2 GPUs for the whole
# 2nd wave). DDP preserves the global eff-batch-48 invariant so
# training stays identical to the Phase-1/2 baselines. Longest-job-
# first (slower gemma2-2b cells ahead of qwen) so any end straggler is
# a short qwen cell -> minimal tail idle. Full HF-Trainer checkpoints
# (optimizer/scheduler/rng + embeddings) pushed -> continual pretrain
# 100M->500M resumable. (original/placebo arms not in this phase.)
set -uo pipefail
cd /workspace/unimixlm
source ./_driver_env.sh
# ~20 weights-only checkpoints/run ≈ one per ~5M tokens for downstream
# learning-dynamics analysis (+ a single final-full resumable bundle).
export NUM_CKPT=20
SIZE=100M; LORA=r16-k32

# Longest-job-first (LPT): the slower gemma2-2b cells first so the final
# straggler is a fast qwen cell -> minimal end-of-run idle.
QUEUE=(
  "gemma2-2b:tur:$SIZE:$LORA:langspec"
  "gemma2-2b:fin:$SIZE:$LORA:langspec"
  "gemma2-2b:hun:$SIZE:$LORA:langspec"
  "qwen2_5-1_5b:tur:$SIZE:$LORA:langspec"
  "qwen2_5-1_5b:fin:$SIZE:$LORA:langspec"
  "qwen2_5-1_5b:hun:$SIZE:$LORA:langspec"
)
printf '%s\n' "${QUEUE[@]}" > "$SHM/queue3.txt"
IDX="$SHM/queue3.idx"; echo 0 > "$IDX"
LOCK="$SHM/queue3.lock"

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
    echo "[orch3][$(date -u +%T)] GPU$gpu -> cell #$i ${model}/${lang}/${arm}"
    bash _run_cell3.sh "$gpu" "$model" "$lang" "$size" "$lora" "$arm" || true
  done
  echo "[orch3][$(date -u +%T)] GPU$gpu drained, exiting"
}

# 2 workers, each owning a 2-GPU pair -> 2 concurrent 2-GPU DDP cells,
# all 4 H200s busy; workers steal the next queued cell the instant they
# free, so GPUs stay saturated until the queue is empty.
for pair in "0,1" "2,3"; do worker "$pair" & done
wait
echo "[orch3][$(date -u +%T)] ALL PHASE-3 CELLS PROCESSED"
