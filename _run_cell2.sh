#!/usr/bin/env bash
# _run_cell2.sh GPU MODEL LANG SIZE LORA
# Phase-2 cell: LoRA-rank/token-budget sweep, original arm, embeddings
# CORRECTLY saved (save_embeddings=true). Fully isolated from the live
# Phase-1 daemons:
#   - output base  $SHM/ft_models2  (Phase-1 pusher only scans ft_models)
#   - per-(lora) subtree so same (lang,size) different rank never collide
#   - HF repos carry a -<loratag> suffix so they never clobber the
#     existing unsuffixed Phase-1 100M r16 repos
#   - manifest tag carries the loratag so the Phase-2 supervisor counts
#     only Phase-2 cells
# Resumable: skips if manifest=completed or the HF repo already has files.
set -uo pipefail
cd /workspace/unimixlm
source ./_driver_env.sh
source .venv/bin/activate

GPU="$1"; MODEL="$2"; LANG="$3"; SIZE="$4"; LORA="$5"   # LORA = hydra group, e.g. r64-k128
LORATAG="${LORA//-/}"                                    # r64-k128 -> r64k128
TAG="${MODEL}_${LANG}_original_${SIZE}_${LORATAG}_s42"
MAN="$RUNS/${TAG}.json"
HF_REPO="${HF_ORG}/ft-langmap-${MODEL}-fineweb${SIZE}-${LANG}-original-${LORATAG}"
LOG="$SHM/logs/${TAG}.log"
OUT2_BASE="$SHM/ft_models2"
OUTDIR="$OUT2_BASE/$MODEL/$LORATAG"     # run_finetune.py appends ft_<lang>-eng_<size>
mkdir -p "$OUTDIR" "$RUNS"

manifest() {
  local st="$1"; shift
  { printf '{"tag":"%s","phase":"2","model":"%s","lang":"%s","gpu":%s,"size":"%s","lora":"%s","hf_repo":"%s","status":"%s","ts":%s' \
      "$TAG" "$MODEL" "$LANG" "$GPU" "$SIZE" "$LORATAG" "$HF_REPO" "$st" "$(date +%s)"
    while [ "$#" -ge 2 ]; do printf ',"%s":"%s"' "$1" "$2"; shift 2; done
    printf '}\n'; } > "$MAN"
}

if [[ -f "$MAN" ]] && grep -q '"status":"completed"' "$MAN"; then
  echo "[GPU$GPU][skip] $TAG already completed"; exit 0; fi
if python3 - "$HF_REPO" <<'PY'
import sys,os
from huggingface_hub import HfApi
api=HfApi(token=os.environ["HF_TOKEN"])
try:
    f=api.list_repo_files(sys.argv[1],repo_type="model")
    sys.exit(0 if any(x.endswith((".safetensors",".bin")) or "adapter_config.json" in x for x in f) else 1)
except Exception:
    sys.exit(1)
PY
then
  echo "[GPU$GPU][skip] $TAG already on HF ($HF_REPO)"
  manifest completed note "preexisting on hub"; exit 0; fi

echo "[GPU$GPU][$(date -u +%T)] START $TAG -> $HF_REPO"
manifest started

CUDA_VISIBLE_DEVICES="$GPU" python3 run_finetune.py \
  model="$MODEL" lora="$LORA" \
  las_mode=original las_tokenizer_path=null \
  model_tokenizer_path="./_modeltok_${MODEL}" \
  langs="{${LANG}:1.0,eng:0.2}" \
  dataset.size="$SIZE" dataset.max_seq_len=1024 \
  num_checkpoints="${NUM_CKPT:-12}" \
  unfreeze_embeddings=true save_embeddings=true gradient_checkpointing=false \
  epochs=1 seed=42 logger.enabled=false \
  output_dir="$OUTDIR" > "$LOG" 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then
  echo "[GPU$GPU][FAIL rc=$rc] $TAG (see $LOG)"
  manifest failed rc "$rc" log "$LOG"; exit $rc; fi

RUN_DIR="$(ls -d "$OUTDIR"/ft_*"${LANG}"*_"${SIZE}" 2>/dev/null | head -1)"
[[ -d "$RUN_DIR" ]] || RUN_DIR="$(ls -dt "$OUTDIR"/*/ 2>/dev/null | head -1)"
if [[ -z "${RUN_DIR:-}" || ! -d "$RUN_DIR" ]]; then
  echo "[GPU$GPU][FAIL] $TAG produced no output dir under $OUTDIR"
  manifest failed reason "no output dir"; exit 1; fi

echo "[GPU$GPU][$(date -u +%T)] UPLOAD $RUN_DIR -> $HF_REPO"
hf repo create "$HF_REPO" --repo-type model -y >/dev/null 2>&1 || true
if hf upload "$HF_REPO" "$RUN_DIR" --repo-type model >>"$LOG" 2>&1; then
  SZ="$(du -sh "$RUN_DIR" 2>/dev/null | cut -f1)"
  rm -rf "$RUN_DIR"
  echo "[GPU$GPU][$(date -u +%T)] OK $TAG (freed $SZ)"
  manifest completed freed "$SZ"
else
  echo "[GPU$GPU][FAIL] $TAG upload failed; keeping $RUN_DIR"
  manifest upload_failed run_dir "$RUN_DIR"; exit 1
fi
