#!/usr/bin/env bash
# run_core_sweep.sh — the cheapest high-effect core sweep, with a
# resumable run-status manifest so partial matrices can never be
# silently consumed downstream.
#
# Cells (chosen from intrinsic headroom — the largest, cheapest wins):
#   1. gemma2-2b x fin  — THE bet (Δbpb_fin≈0.296, max headroom cell)
#                          arms: original, base, langspec, placebo
#                          + 2nd seed on base & langspec (anti-luck)
#   2. gemma2-2b x hun  — 2nd-strongest language          arms: base, langspec
#   3. phi3-mini x fin  — paradigm-different robustness    arms: base, langspec
#
# Regime: LoRA r16, FROZEN embeddings (cheapest), seq 1024, 100M tokens,
# 1 epoch. Override via env vars below.
#
# Prereqs: overlay tokenizers built
# for each model, HF model + FineWeb2 access, placebo files generated.
set -euo pipefail
cd "$(dirname "$0")/.."

SIZE="${SIZE:-100M}"
SEQ="${SEQ:-1024}"
LORA="${LORA:-r16-k32}"
UNFREEZE="${UNFREEZE:-false}"          # cheapest = false (frozen embeddings)
GRAD_CKPT="${GRAD_CKPT:-false}"        # 2-4B on A100-80G fits without it
OUT="${OUT:-./ft_models}"
RUNS="${RUNS:-./runs}"                 # status-manifest dir
LAS_GEMMA="${LAS_GEMMA:-./las_gemma2_2b}"   # overlay dir for gemma2-2b
LAS_PHI="${LAS_PHI:-./las_phi3_mini}"       # overlay dir for phi3-mini
mkdir -p "$RUNS" "$OUT"

# cell spec: "model|model_cfg|lang|las_dir|arm|seed"
CELLS=(
  "gemma2-2b|gemma2-2b|fin|${LAS_GEMMA}|original|42"
  "gemma2-2b|gemma2-2b|fin|${LAS_GEMMA}|placebo|42"
  "gemma2-2b|gemma2-2b|fin|${LAS_GEMMA}|base|42"
  "gemma2-2b|gemma2-2b|fin|${LAS_GEMMA}|base|1234"
  "gemma2-2b|gemma2-2b|fin|${LAS_GEMMA}|langspec|42"
  "gemma2-2b|gemma2-2b|fin|${LAS_GEMMA}|langspec|1234"
  "gemma2-2b|gemma2-2b|hun|${LAS_GEMMA}|base|42"
  "gemma2-2b|gemma2-2b|hun|${LAS_GEMMA}|langspec|42"
  "phi3-mini|phi3-mini|fin|${LAS_PHI}|base|42"
  "phi3-mini|phi3-mini|fin|${LAS_PHI}|langspec|42"
)

mkstatus() { python3 - "$@" <<'PY'
import json, sys, time
path, status, extra = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(path))
except Exception:
    d = {}
d.update(json.loads(extra)); d["status"] = status; d["ts"] = time.time()
json.dump(d, open(path, "w"), indent=2)
PY
}

for spec in "${CELLS[@]}"; do
  IFS='|' read -r model mcfg lang lasdir arm seed <<<"$spec"
  tag="${model}_${lang}_${arm}_${SIZE}_s${seed}"
  man="${RUNS}/${tag}.json"

  if [[ -f "$man" ]] && grep -q '"status": "completed"' "$man"; then
    echo "[skip] $tag already completed"; continue
  fi

  mkstatus "$man" started "$(printf '{"tag":"%s","model":"%s","lang":"%s","arm":"%s","seed":%s,"size":"%s","lora":"%s","unfreeze":%s}' \
      "$tag" "$model" "$lang" "$arm" "$seed" "$SIZE" "$LORA" "$UNFREEZE")"

  echo "=== RUN $tag ==="
  if python3 run_finetune.py \
        model="$mcfg" \
        lora="$LORA" \
        las_mode="$arm" \
        las_tokenizer_path="$lasdir" \
        model_tokenizer_path="${lasdir}/model_tokenizer" \
        langs="{${lang}:1.0,eng:0.2}" \
        dataset.size="$SIZE" \
        dataset.max_seq_len="$SEQ" \
        unfreeze_embeddings="$UNFREEZE" \
        gradient_checkpointing="$GRAD_CKPT" \
        epochs=1 \
        seed="$seed" \
        logger.enabled=false \
        output_dir="$OUT" ; then
    bpb=$(grep -ho 'eval_bpb[^,]*' "${OUT}"/*"${lang}"*/trainer_state.json 2>/dev/null | tail -1 || true)
    mkstatus "$man" completed "$(printf '{"eval_bpb_raw":"%s"}' "${bpb:-NA}")"
    echo "[ok] $tag"
  else
    mkstatus "$man" failed '{}'
    echo "[FAIL] $tag — leaving manifest=failed, continuing"
  fi
done

echo; echo "=== matrix_status ==="
python3 - "$RUNS" <<'PY'
import json, glob, os, sys
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    d = json.load(open(f))
    rows.append((d.get("tag", os.path.basename(f)), d.get("status", "?")))
for tag, st in rows:
    print(f"  {st:<10} {tag}")
done = sum(1 for _, s in rows if s == "completed")
print(f"\n  {done}/{len(rows)} completed")
PY
