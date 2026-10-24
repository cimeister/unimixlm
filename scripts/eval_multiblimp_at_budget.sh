#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Evaluate the fine-tuning checkpoints on MultiBLiMP — BPE (original) vs LAS
# (langspec) only — across the 3 anchor models × 2 modes × {fin, tur}, at a
# CHOSEN token-budget observation checkpoint. Generalizes
# scripts/eval_100m_multiblimp.sh (now superseded) to any budget.
#
# Budget → checkpoint-step mapping (per the project's observation schedule):
#   10M=checkpoint-77, 50M=checkpoint-382, 100M=checkpoint-763, 500M=checkpoint-3815.
# Output dir per budget:
#   10M  -> eval_results/p4d_downstream_10M/
#   50M  -> eval_results/p4d_downstream_50M/
#   100M -> eval_results/p4d_downstream_100M/
#   500M -> eval_results/p4d_downstream/        (the existing F12 dir — REFUSED
#           here to avoid clobbering F12; eval 500M via the SLURM submitter.)
#
# Reuses the same engine (run_eval.py --config-name p4d) as F12 so deltas are
# apples-to-apples; only the checkpoint step differs. Runs on the LOGIN NODE
# under `uenv run`, serial, one GPU. Idempotent (skips groups already done).
#
# Usage:
#   bash scripts/eval_multiblimp_at_budget.sh --budget 50M --dry-run
#   GPU=1 bash scripts/eval_multiblimp_at_budget.sh --budget 50M --only llama3.2-1b:unfrozen:tur
#   bash scripts/eval_multiblimp_at_budget.sh --budget 50M     # all 12 groups
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
set -uo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"
source slurm/env.sh

CKPT_ROOT="${SCRATCH_DIR:?set SCRATCH_DIR to your cluster scratch root}/unimix_p4d_phase1"
LOGD="${REPO_DIR}/logs"
LAS="${REPO_DIR}/las_tokenizers"
GPU="${GPU:-0}"

BUDGET=""; DRY_RUN=false; ONLY=""
while [ $# -gt 0 ]; do case "$1" in
  --budget)  BUDGET="$2"; shift 2 ;;
  --dry-run) DRY_RUN=true; shift ;;
  --only)    ONLY="$2"; shift 2 ;;   # model:mode:lang
  *) echo "Unknown arg: $1"; exit 2 ;;
esac; done

# Budget → (checkpoint step, output subdir). Fail loud on unknown / 500M.
case "$BUDGET" in
  10M)  CKPT_STEP=77;   OUT_SUB="p4d_downstream_10M" ;;
  50M)  CKPT_STEP=382;  OUT_SUB="p4d_downstream_50M" ;;
  100M) CKPT_STEP=763;  OUT_SUB="p4d_downstream_100M" ;;
  500M) echo "[ABORT] 500M is the F12 dir (eval_results/p4d_downstream); refusing"
        echo "        to risk clobbering it from this login-node script. Use the"
        echo "        SLURM submitter slurm/submit_p4d_downstream_eval.sh for 500M."
        exit 3 ;;
  *) echo "[ABORT] --budget must be one of {10M,50M,100M} (got '${BUDGET}')"; exit 2 ;;
esac
OUT_ROOT="${REPO_DIR}/eval_results/${OUT_SUB}"
mkdir -p "$OUT_ROOT" "$LOGD"

declare -A BASE_MODEL
BASE_MODEL[qwen3-0.6b]="Qwen/Qwen3-0.6B"
BASE_MODEL[gemma3-1b]="google/gemma-3-1b-pt"
BASE_MODEL[llama3.2-1b]="meta-llama/Llama-3.2-1B"
declare -A TOK_MORPH
TOK_MORPH[qwen3-0.6b]="${LAS}/qwen3-0.6b_v151k_fw2sample_50klines_fin-hun-tur-swh-kor-arb_softem_20260417"
TOK_MORPH[gemma3-1b]="${LAS}/gemma3-1b_v262k_fw2sample_50klines_fin-hun-tur-swh-kor-arb_softem_20260417"
TOK_MORPH[llama3.2-1b]="${LAS}/llama3.2-1b_v128k_fw2sample_50klines_fin-hun-tur-swh-kor-arb_softem_20260515"

MODELS="qwen3-0.6b gemma3-1b llama3.2-1b"
MODES="embedonly unfrozen"
LANGS="fin tur"

ck_step() {  # $1=cell -> echoes checkpoint-${CKPT_STEP} path or returns 1
  local cell="$1"
  local inner; inner=$(find "${CKPT_ROOT}/${cell}" -maxdepth 1 -type d -name 'ft*' 2>/dev/null | head -1)
  [ -n "$inner" ] || { echo "  [no-inner] $cell" >&2; return 1; }
  local st; st=$(python3 -c "import json;print(json.load(open('${inner}/run_metadata.json')).get('status','?'))" 2>/dev/null)
  [ "$st" = completed ] || { echo "  [status=$st] $cell" >&2; return 1; }
  local ck="${inner}/checkpoint-${CKPT_STEP}"
  [ -d "$ck" ] || { echo "  [no-checkpoint-${CKPT_STEP}] $cell" >&2; return 1; }
  echo "$ck"
}

echo "=== MultiBLiMP eval @ budget=${BUDGET} (checkpoint-${CKPT_STEP}) -> ${OUT_SUB} ==="
run_count=0; skip_count=0; fail_count=0
for model in $MODELS; do
  for mode in $MODES; do
    for lang in $LANGS; do
      tag="${model}:${mode}:${lang}"
      [ -n "$ONLY" ] && [ "$ONLY" != "$tag" ] && continue
      LASDIR="${TOK_MORPH[$model]}"
      gtag="p4d-${model}-morph-${mode}-${lang}"
      OUT="${OUT_ROOT}/${gtag}"
      if ls "${OUT}"/eval_results_ft_2*.json >/dev/null 2>&1 \
         && ls "${OUT}"/eval_results_ft_las_*.json >/dev/null 2>&1; then
        echo "[SKIP existing] ${gtag}"; skip_count=$((skip_count+1)); continue
      fi
      CK_ORIG=$(ck_step "p4d-${model}-morph-${mode}-original-${lang}") || {
        echo "[SKIP unresolved-BPE] ${gtag}"; skip_count=$((skip_count+1)); continue; }
      CK_LAS=$(ck_step  "p4d-${model}-morph-${mode}-langspec-${lang}") || {
        echo "[SKIP unresolved-LAS] ${gtag}"; skip_count=$((skip_count+1)); continue; }
      if [ "$DRY_RUN" = true ]; then
        echo "[DRY] ${gtag}  GPU=${GPU}  step=${CKPT_STEP}  langs=[eng,${lang}] task=multiblimp"
        run_count=$((run_count+1)); continue
      fi
      mkdir -p "$OUT"
      LOG="${LOGD}/eval${BUDGET}_${gtag}.log"
      echo "[RUN] ${gtag} -> ${LOG} (GPU ${GPU})"
      CUDA_VISIBLE_DEVICES="${GPU}" uenv run pytorch/v2.8.0:v1 --view=default -- \
        env PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" HF_DATASETS_TRUST_REMOTE_CODE=1 \
        "${PYTHON}" run_eval.py --config-name p4d \
          output_path="$OUT" \
          model_args.pretrained="${BASE_MODEL[$model]}" \
          model_args.peft_original_tok="$CK_ORIG" \
          model_args.peft_base_tok="$CK_ORIG" \
          model_args.peft_pathpiece_tok="$CK_ORIG" \
          model_args.peft_las_tok="$CK_LAS" \
          eval_ft_base.enabled=false \
          eval_ft_pathpiece.enabled=false \
          eval_ft_base.base_tokenizer_path="${LASDIR}/base_tokenizer.json" \
          eval_ft_pathpiece.pathpiece_tokenizer_path="${LASDIR}/pathpiece_tokenizer.json" \
          custom_tokenizer.las_tokenizer_main_path="$LASDIR" \
          custom_tokenizer.las_lang="$lang" \
          custom_tokenizer.lang_suffix=_Latn \
          batch_size=16 \
          "eval_langs=[eng,${lang}]" \
          "tasks=[multiblimp]" \
        > "$LOG" 2>&1
      rc=$?
      if [ "$rc" -eq 0 ]; then echo "  [OK] ${gtag}"; run_count=$((run_count+1))
      else echo "  [FAIL rc=${rc}] ${gtag} — see ${LOG}"; fail_count=$((fail_count+1)); fi
    done
  done
done
echo ""
echo "=== budget=${BUDGET} summary: ran/dry=${run_count} skipped=${skip_count} failed=${fail_count} ==="
