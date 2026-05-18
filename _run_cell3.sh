#!/usr/bin/env bash
# _run_cell3.sh GPU MODEL LANG SIZE LORA ARM
# Phase-3: base-vs-LangMAP study. IDENTICAL training setup to the
# Phase-2 baseline runner (_run_cell2.sh) — the ARGS array below is
# verbatim the same; the ONLY per-cell differences are:
#     ARM=original  -> las_mode=original   las_tokenizer_path=null
#     ARM=langspec  -> las_mode=langspec   las_tokenizer_path=<lang dir>
#     ARM=placebo   -> las_mode=placebo    las_tokenizer_path=<lang dir>
# Single-GPU on purpose: WORLD_SIZE=1 keeps every DDP-gated branch in
# finetune.py inert, so training is byte-identical to the single-GPU
# baseline path (global eff-batch 48, lr/val/etc. inherited from the
# same config defaults — NOT overridden here). save_embeddings=true is
# mandatory for the overlay arms (and used by all 3 so the control is
# clean). Fully isolated from live Phase-1/2 daemons:
#   - output base $SHM/ft_models3 (P1/P2 pushers never scan it)
#   - HF repo carries -<arm>-<loratag> so it never clobbers P1/P2 repos
#   - manifest carries "phase":"3" + arm
# Resumable: skips if manifest=completed or the HF repo already has files.
set -uo pipefail
cd /workspace/unimixlm
source ./_driver_env.sh
source .venv/bin/activate

GPUS="$1"; MODEL="$2"; LANG="$3"; SIZE="$4"; LORA="$5"; ARM="$6"  # GPUS CSV e.g. "0,1"
GPU="$GPUS"                                            # used in labels/manifest
NPROC="$(awk -F, '{print NF}' <<< "$GPUS")"            # GPUs in this cell
FIRST_GPU="${GPUS%%,*}"
MASTER_PORT=$(( 29600 + FIRST_GPU ))                   # unique per GPU-pair (P2-free)
LORATAG="${LORA//-/}"                                  # r16-k32 -> r16k32
TAG="${MODEL}_${LANG}_${ARM}_${SIZE}_${LORATAG}_s42"
MAN="$RUNS/${TAG}.json"
HF_REPO="${HF_ORG}/ft-langmap-${MODEL}-fineweb${SIZE}-${LANG}-${ARM}-${LORATAG}"
LOG="$SHM/logs/${TAG}.log"
OUTDIR="$SHM/ft_models3/$MODEL/$ARM/$LORATAG"   # run_finetune.py appends ft_<lang>-eng_<size>
LAS_DIR="$REPO/_las_tok3/$MODEL/$LANG"          # holds {langspec,placebo}_<lang>_Latn.json
mkdir -p "$OUTDIR" "$RUNS"

if [[ "$ARM" == "original" ]]; then
  LAS_MODE=original; LAS_PATH=null
elif [[ "$ARM" == "langspec" || "$ARM" == "placebo" ]]; then
  LAS_MODE="$ARM"; LAS_PATH="$LAS_DIR"
  f="$LAS_DIR/${ARM}_${LANG}_Latn.json"
  if [[ ! -f "$f" ]]; then
    echo "[GPU$GPU][FAIL] $TAG missing tokenizer $f (run _prep_tok3.sh)"
    exit 1; fi
else
  echo "[GPU$GPU][FAIL] $TAG unknown ARM '$ARM'"; exit 1; fi

manifest() {
  local st="$1"; shift
  { printf '{"tag":"%s","phase":"3","model":"%s","lang":"%s","arm":"%s","gpu":"%s","size":"%s","lora":"%s","hf_repo":"%s","status":"%s","ts":%s' \
      "$TAG" "$MODEL" "$LANG" "$ARM" "$GPU" "$SIZE" "$LORATAG" "$HF_REPO" "$st" "$(date +%s)"
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

echo "[GPU$GPU][$(date -u +%T)] START $TAG ($ARM) -> $HF_REPO"
manifest started

# ---- VERBATIM the Phase-2 baseline ARGS; only las_mode/las_tokenizer_path vary ----
ARGS=(
  model="$MODEL" lora="$LORA"
  las_mode="$LAS_MODE" las_tokenizer_path="$LAS_PATH"
  model_tokenizer_path="./_modeltok_${MODEL}"
  langs="{${LANG}:1.0,eng:0.2}"
  dataset.size="$SIZE" dataset.max_seq_len=1024
  num_checkpoints="${NUM_CKPT:-20}"
  unfreeze_embeddings=true save_embeddings=true gradient_checkpointing=false
  save_only_model=true save_final_full_state=true
  epochs=1 seed=42 logger.enabled=false
  output_dir="$OUTDIR"
  hydra.job.chdir=False hydra.output_subdir=null
  hydra/job_logging=disabled hydra/hydra_logging=disabled
  hydra.run.dir="/tmp/hyd_${TAG}"
)
if [[ "$NPROC" -gt 1 ]]; then
  echo "[GPU$GPU][$(date -u +%T)] DDP torchrun nproc=$NPROC port=$MASTER_PORT"
  CUDA_VISIBLE_DEVICES="$GPUS" python3 -m torch.distributed.run \
    --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    run_finetune.py "${ARGS[@]}" > "$LOG" 2>&1
else
  CUDA_VISIBLE_DEVICES="$GPUS" python3 run_finetune.py "${ARGS[@]}" > "$LOG" 2>&1
fi
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
