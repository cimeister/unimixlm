#!/usr/bin/env bash
# run_langmap_qwen_initial.sh — ASAP initial LangMAP experiment.
#
#   Model : Qwen2.5-1.5B   (config/finetune/model/qwen2_5-1_5b.yaml)
#   Langs : tur (priority) then fin, hun     [override via LANGS=...]
#   Regime: unfrozen embeddings + LoRA, 500M tokens, 1 epoch,
#           5 checkpoints (~one per 100M tokens), seq 1024.
#   Arms  : original (runs now). langspec runs ONLY if an overlay
#           tokenizer dir exists for the model (see scripts/LANGMAP_README.md);
#           otherwise that arm is skipped with a clear message — no GPU spend.
#
# Resumable: a per-cell JSON manifest under $RUNS; completed cells skip.
set -euo pipefail
cd "$(dirname "$0")/.."

LANGS="${LANGS:-tur fin hun}"           # space-separated; tur first by priority
SIZE="${SIZE:-500M}"
SEQ="${SEQ:-1024}"
LORA="${LORA:-r16-k32}"
NUM_CKPT="${NUM_CKPT:-5}"               # 500M / 5 ≈ checkpoint per ~100M tok
UNFREEZE="${UNFREEZE:-true}"            # unfrozen embeddings (LangMAP regime)
GRAD_CKPT="${GRAD_CKPT:-false}"         # 1.5B fits on H200/A100-80G w/o it
ARMS="${ARMS:-original langspec}"       # langspec auto-skips if no overlay
SEED="${SEED:-42}"
OUT="${OUT:-./ft_models}"
RUNS="${RUNS:-./runs}"
# Overlay dir holding base_tokenizer.json + langspec_{lang}_Latn.json +
# model_tokenizer/  (only needed for the langspec arm).
LAS_QWEN="${LAS_QWEN:-./las_qwen2_5_1_5b}"
mkdir -p "$RUNS" "$OUT"

mkstatus() { python3 - "$@" <<'PY'
import json, sys, time
path, status, extra = sys.argv[1], sys.argv[2], sys.argv[3]
try: d = json.load(open(path))
except Exception: d = {}
d.update(json.loads(extra)); d["status"] = status; d["ts"] = time.time()
json.dump(d, open(path, "w"), indent=2)
PY
}

for lang in $LANGS; do
  for arm in $ARMS; do
    tag="qwen2_5-1_5b_${lang}_${arm}_${SIZE}_s${SEED}"
    man="${RUNS}/${tag}.json"
    if [[ -f "$man" ]] && grep -q '"status": "completed"' "$man"; then
      echo "[skip] $tag already completed"; continue
    fi

    las_args=()
    if [[ "$arm" == "original" ]]; then
      # original tokenizer still needs a model_tokenizer dir; the model's
      # own HF tokenizer is used. Point model_tokenizer_path at the overlay
      # if present, else fall back to a snapshot of the base model tokenizer.
      mtp="${LAS_QWEN}/model_tokenizer"
      if [[ ! -d "$mtp" ]]; then mtp="./_modeltok_qwen2_5-1_5b"; fi
      las_args=( las_mode=original las_tokenizer_path=null
                 model_tokenizer_path="$mtp" )
    else
      if [[ ! -d "$LAS_QWEN" ]]; then
        echo "[gate] $tag — langspec overlay '$LAS_QWEN' not built yet; SKIP."
        mkstatus "$man" gated '{"reason":"no langspec overlay tokenizer"}'
        continue
      fi
      las_args=( las_mode=langspec las_tokenizer_path="$LAS_QWEN"
                 model_tokenizer_path="${LAS_QWEN}/model_tokenizer" )
    fi

    mkstatus "$man" started "$(printf '{"tag":"%s","model":"qwen2_5-1_5b","lang":"%s","arm":"%s","size":"%s","num_ckpt":%s,"unfreeze":%s}' \
        "$tag" "$lang" "$arm" "$SIZE" "$NUM_CKPT" "$UNFREEZE")"
    echo "=== RUN $tag ==="
    if python3 run_finetune.py \
          model=qwen2_5-1_5b \
          lora="$LORA" \
          "${las_args[@]}" \
          langs="{${lang}:1.0,eng:0.2}" \
          dataset.size="$SIZE" \
          dataset.max_seq_len="$SEQ" \
          num_checkpoints="$NUM_CKPT" \
          unfreeze_embeddings="$UNFREEZE" \
          gradient_checkpointing="$GRAD_CKPT" \
          epochs=1 \
          seed="$SEED" \
          logger.enabled=false \
          output_dir="$OUT" ; then
      mkstatus "$man" completed '{}'; echo "[ok] $tag"
    else
      mkstatus "$man" failed '{}'; echo "[FAIL] $tag — continuing"
    fi
  done
done
echo; echo "Done. Manifests in $RUNS, checkpoints under $OUT/ft*_${SIZE}/"
