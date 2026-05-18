#!/usr/bin/env bash
# _orchestrator.sh — fill all 4 H200s with original-arm LangMAP cells.
# Priority: qwen2_5-1_5b/tur first. Then longest model first (mistral 7B ->
# gemma 2.6B -> qwen 1.5B) so big jobs start early and GPUs stay balanced.
# Each worker is pinned to one GPU and pulls the next cell atomically.
set -uo pipefail
cd /workspace/unimixlm
source ./_driver_env.sh

# 7 models x {tur,fin,hun} = 21 cells, original arm, 100M tokens.
# qwen2_5-1_5b/tur forced first (user priority); then longest model first
# so the fast 1B/0.5B cells backfill the tail and all 4 H200s stay busy
# until the very end (near-100% steady-state utilisation).
# Small-models-only matrix (user choice): 3 models x {tur,fin,hun} = 9
# cells, original arm, 100M tokens. ~19 GPU-h ~ 5h on 4xH200.
# qwen2_5-1_5b/tur forced first (priority); longest-first otherwise.
QUEUE=(
  "qwen2_5-1_5b:tur"        # <-- explicit priority, starts immediately
  "qwen2_5-1_5b:fin" "qwen2_5-1_5b:hun"                            # ~1.5B
  "gemma3-1b:tur" "gemma3-1b:fin" "gemma3-1b:hun"                  # ~1B
  "qwen2_5-0_5b:tur" "qwen2_5-0_5b:fin" "qwen2_5-0_5b:hun"         # ~0.5B
)
printf '%s\n' "${QUEUE[@]}" > "$SHM/queue.txt"
IDX="$SHM/queue.idx"; echo 0 > "$IDX"
LOCK="$SHM/queue.lock"

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
    local model="${cell%%:*}" lang="${cell##*:}"
    echo "[orch][$(date -u +%T)] GPU$gpu -> cell #$i ${model}/${lang}"
    bash _run_cell.sh "$gpu" "$model" "$lang" || true
  done
  echo "[orch][$(date -u +%T)] GPU$gpu drained, exiting"
}

for g in 0 1 2 3; do worker "$g" & done
wait
echo "[orch][$(date -u +%T)] ALL CELLS PROCESSED"
python3 - "$RUNS" <<'PY'
import json,glob,os,sys
rows=[]
for f in sorted(glob.glob(os.path.join(sys.argv[1],"*.json"))):
    d=json.load(open(f)); rows.append((d.get("tag",os.path.basename(f)),d.get("status","?")))
for t,s in rows: print(f"  {s:<14} {t}")
print(f"\n  {sum(1 for _,s in rows if s=='completed')}/{len(rows)} completed")
PY
