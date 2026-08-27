#!/usr/bin/env bash
# run_langmap_ft_sweep.sh — the LangMAP finetuning sweep.
#
#   Models: qwen2_5-1_5b (1.5B), gemma2-2b (2.6B), mistral-7b-v0_3 (7.2B)
#   Langs : tur, fin, hun
#   Arms  : original, langspec   (langspec auto-skips per model until its
#                                 overlay tokenizer dir exists)
#   Regime: unfrozen embeddings + LoRA, 500M tokens, 1 epoch,
#           5 checkpoints (~one per 100M tokens), seq 1024.
#
#   => 3 models x 3 langs x 2 arms = 18 cells (langspec cells gate-skip
#      until overlays are built; original cells = 9 run immediately).
#
# Run ONE process per GPU (the launcher does not do DDP) — for a 4-GPU
# pod, start this script 4x pinned to CUDA_VISIBLE_DEVICES=0..3 over
# disjoint slices, or just run it once and let cells run serially.
# Resumable via per-cell JSON manifest under $RUNS.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${MODELS:-qwen2_5-1_5b gemma2-2b mistral-7b-v0_3}"
LANGS="${LANGS:-tur fin hun}"
ARMS="${ARMS:-original langspec}"
SIZE="${SIZE:-500M}"
SEQ="${SEQ:-1024}"
LORA="${LORA:-r16-k32}"
NUM_CKPT="${NUM_CKPT:-5}"
UNFREEZE="${UNFREEZE:-true}"
GRAD_CKPT="${GRAD_CKPT:-false}"        # set true if 7B OOMs on smaller cards
SEED="${SEED:-42}"
OUT="${OUT:-./ft_models}"
RUNS="${RUNS:-./runs}"
mkdir -p "$RUNS" "$OUT"

# Per-model overlay tokenizer dir (langspec arm only). Override via env.
overlay_dir() { case "$1" in
  qwen2_5-1_5b)    echo "${LAS_QWEN:-./las_qwen2_5_1_5b}";;
  gemma2-2b)       echo "${LAS_GEMMA:-./las_gemma2_2b}";;
  mistral-7b-v0_3) echo "${LAS_MISTRAL:-./las_mistral_7b_v0_3}";;
  *) echo "";; esac; }

mkstatus() { python3 - "$@" <<'PY'
import json, sys, time
path, status, extra = sys.argv[1], sys.argv[2], sys.argv[3]
try: d = json.load(open(path))
except Exception: d = {}
d.update(json.loads(extra)); d["status"] = status; d["ts"] = time.time()
json.dump(d, open(path, "w"), indent=2)
PY
}

for model in $MODELS; do
  ovl="$(overlay_dir "$model")"
  for lang in $LANGS; do
    for arm in $ARMS; do
      tag="${model}_${lang}_${arm}_${SIZE}_s${SEED}"
      man="${RUNS}/${tag}.json"
      if [[ -f "$man" ]] && grep -q '"status": "completed"' "$man"; then
        echo "[skip] $tag already completed"; continue
      fi

      if [[ "$arm" == "langspec" ]]; then
        if [[ -z "$ovl" || ! -d "$ovl" ]]; then
          echo "[gate] $tag — langspec overlay '$ovl' not built; SKIP."
          mkstatus "$man" gated '{"reason":"no langspec overlay tokenizer"}'
          continue
        fi
        las_args=( las_mode=langspec las_tokenizer_path="$ovl"
                   model_tokenizer_path="${ovl}/model_tokenizer" )
      else
        mtp="${ovl}/model_tokenizer"
        [[ -d "$mtp" ]] || mtp="./_modeltok_${model}"
        las_args=( las_mode=original las_tokenizer_path=null
                   model_tokenizer_path="$mtp" )
      fi

      mkstatus "$man" started "$(printf '{"tag":"%s","model":"%s","lang":"%s","arm":"%s","size":"%s","num_ckpt":%s,"unfreeze":%s}' \
          "$tag" "$model" "$lang" "$arm" "$SIZE" "$NUM_CKPT" "$UNFREEZE")"
      echo "=== RUN $tag ==="
      if python3 run_finetune.py \
            model="$model" \
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
done

echo; echo "=== sweep_status ==="
python3 - "$RUNS" <<'PY'
import json, glob, os, sys
rows=[]
for f in sorted(glob.glob(os.path.join(sys.argv[1],"*.json"))):
    d=json.load(open(f)); rows.append((d.get("tag",os.path.basename(f)),d.get("status","?")))
for t,s in rows: print(f"  {s:<10} {t}")
done=sum(1 for _,s in rows if s=="completed")
print(f"\n  {done}/{len(rows)} completed")
PY
