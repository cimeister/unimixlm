#!/usr/bin/env bash
# launch_f15_wave1.sh — F15 wave 1: Qwen2.5-0.5B new cells (A1/A2/A3/A5/A4).
#
# See the experiment plan ("Arms" table,
# "Execution order" step 2: "Wave 1: Qwen2.5 new cells"). Pre-launch review
# finding S1 (2026-08-17) found the tur/fin/hun langspec overlays previously
# in use were PR21-era HF artifacts, not fit from THIS family's F15
# base_tokenizer.json, which made the A2-vs-A5 control unclean for those 3
# languages (A2=A5+language-specific-EM only holds when both descend from
# the identical base vocab -- see scripts/prep_f15_tokenizers.py FAMILIES
# comment). Resolution: tur/fin/hun overlays are refit from the F15 base
# (scripts/prep_f15_tokenizers.py, run by a separate concurrent
# process/session), and this launcher now trains FRESH A1/A2 tur/fin/hun
# cells here rather than reusing the F14c (S18) tur/fin/hun A1/A2 cells --
# those remain a valid but PROTOCOL-DIFFERENT comparison point (see the
# EVAL_TOKEN_BUDGETS note below), used only in the mechanism analysis, not
# folded into the wave-1 headline numbers.
#
# Grid: seeds {42,43,44} x
#   A1 original     x {tur,fin,hun,arb,hin}   (5/seed)
#   A2 langspec     x {tur,fin,hun,arb,hin}   (5/seed)
#   A5 base         x {tur,fin,hun,arb,hin}   (5/seed)
#   A3 placebo      x {arb,hin}               (2/seed)
#   A4 vocab_extend x {tur,fin,hun,arb,hin}   (5/seed)
# = 22 cells/seed x 3 seeds = 66 cells.
#
# Protocol (F15 plan "Arms" row, EXPERIMENTAL_SETUP.md S19): true_tie
# throughout, LoRA r16-k32 + fully-trained tied embedding, lr 2e-4, eff.
# batch 48 (per_device_batch=6 x grad_accum=8, single GPU/cell — matches
# config/finetune/model/qwen2_5-0_5b.yaml), 100M target-lang tokens (design
# target; see EVAL_TOKEN_BUDGETS below for the actual padded-position
# mechanism) + eng:0.2, observation checkpoints calibrated to ~10M/50M/100M
# real target-lang tokens. Derived from scripts/launch_pr21_truetie.sh (job
# packing, preflight, sbatch conventions) with per-arm/per-lang
# generalization for the 5 F15 arms.
#
# EVAL_TOKEN_BUDGETS AND THE STOP POINT (read before changing this script):
# EVAL_TOKEN_BUDGETS is defined in PADDED TRAINING POSITIONS, not real
# (non-padding) target-language tokens -- every sequence in every batch is
# padded/packed to max_seq_len=1024 regardless of how much real content it
# holds, so "positions processed" and "real tokens learned from" are
# different quantities that diverge by each arm's own padding efficiency
# (itself a function of that arm's tokenizer: langspec/base compress
# differently than the model's original BPE, vocab_extend differently
# again). Running every arm to the SAME max_steps (identical optimizer step
# count, hence identical position count) is deliberate: it is the fairness
# mechanism across arms -- a real-token-budget stop would let a
# more-efficient tokenizer's arm see MORE optimizer steps for the "same"
# budget, confounding tokenizer efficiency with training duration. The
# budgets below are 1.75x the original round-number targets
# (10M/50M/100M), calibrated so the LARGEST budget still lands close to the
# original ~100M-real-target-lang-tokens design target for the MEDIAN arm
# (not every arm identically, since padding efficiency differs per arm) --
# the actual real (non-padding) per-arm token count trained is NOT
# estimated here; it is read after the fact from each run's own
# run_metadata.json (tokens.total_posttrunc_trained / data_sources[].
# target_tokens), never hardcoded.
#
# tokens_per_step = per_device_batch_size(6) x world_size(1, single GPU/cell,
# no torch.distributed) x gradient_accumulation_steps(8) x max_seq_len(1024)
# = 49,152 tokens/step (las_experiment/finetune.py, eval_token_budgets block;
# unchanged from F14c/PR21 -- only the budget values below changed).
# With EVAL_TOKEN_BUDGETS=[17500000,87500000,175000000], observation_steps =
# [ceil(17.5e6/49152), ceil(87.5e6/49152), ceil(175e6/49152)] = [357, 1781,
# 3561] (math.ceil, as finetune.py actually computes it -- a round()-based
# back-of-envelope gives [356,1780,3560], off by one step per budget from
# what the code will actually run; noted here rather than silently
# "corrected" since it's a 0.03%, walltime-immaterial difference either way).
# max_steps is fixed to 3561 (the LAST budget), i.e. training stops at
# 3561*49152 ~= 175.03M padded positions REGARDLESS of epoch count or corpus
# size — finetune.py sets num_train_epochs=1_000_000 and lets max_steps cap
# it, so the Trainer loops the dataset across epoch boundaries as needed.
# The F15 languages' dataset.size=100M hub corpora are each ~100M
# whitespace-tokens (188k-260k documents per the HF dataset cards) —
# comfortably enough documents that this run finishes within roughly one
# lang-corpus pass; the eng:0.2 slice is a document-count subsample of the
# eng corpus, not a separate token budget.
#
# WAVE-1 vs F14c COMPARABILITY: wave-1 A1/A2 tur/fin/hun cells (this
# launcher) and F14c's (S18) tur/fin/hun A1/A2 cells are BOTH true_tie, but
# use DIFFERENT stopping protocols -- F14c used epochs=1 with NO
# eval_token_budgets (single checkpoint at end of one epoch, real-token
# count = whatever one epoch of that corpus contains); wave-1 uses the
# padded-position max_steps schedule above (identical optimizer steps across
# every wave-1 arm/lang/seed). Wave-1 A1/A2 tur/fin/hun cells ARE
# budget-matched against every other wave-1 cell (IN-wave comparisons are
# valid); F14c cells are NOT budget-matched against wave-1 and are used only
# for the stopping-protocol mechanism analysis, never averaged into a
# wave-1 headline number.
#
# Usage: bash scripts/launch_f15_wave1.sh [--dry-run]
set -euo pipefail

REPO_DIR="/home/USER/unimixlm"
PYTHON="/path/to/scratch/USER/unimixlm-venv-aug26/bin/python"
TOK_ROOT="${REPO_DIR}/las_tokenizers/f15/qwen2_5-0_5b"
OUT_ROOT="/path/to/scratch/USER/f15_wave1"
# Verified complete (2026-08-17): all 5 F15 lang repos, the eng repo, and
# the Qwen2.5-0.5B snapshot are cached here (shared with the PR#21 rerun --
# same hub repos, see finetune.py resolve_fineweb2_lang).
HF_CACHE="/path/to/scratch/USER/hf_cache_pr21"
LOG_DIR="${REPO_DIR}/logs/f15_wave1"
JID_FILE="${REPO_DIR}/.f15_hunplacebo_jids"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

DRY_RUN="${1:-}"

SEEDS=(42 43 44 45 46)

# Per-arm language lists (F15 plan "Arms"/"Seeds and cell count" sections).
declare -A ARM_LANGS
ARM_LANGS[original]=""
ARM_LANGS[langspec]=""
ARM_LANGS[base]=""
ARM_LANGS[placebo]="hun"
ARM_LANGS[vocab_extend]=""
ARMS_ORDER=(original langspec base placebo vocab_extend)

# Script-suffix per language (ISO 15924; matches
# scripts/eval_f14_multiblimp.py's LANG_SUFFIX and the mono32k/langspec/
# vocabext filenames under TOK_ROOT).
declare -A LANG_SUFFIX
LANG_SUFFIX[tur]="_Latn"; LANG_SUFFIX[fin]="_Latn"; LANG_SUFFIX[hun]="_Latn"
LANG_SUFFIX[arb]="_Arab"; LANG_SUFFIX[hin]="_Deva"

# Padded training positions, not real tokens -- see the EVAL_TOKEN_BUDGETS
# header note above for the fairness-mechanism rationale and the
# steps=[357,1781,3561] this yields via tokens_per_step=49152.
EVAL_TOKEN_BUDGETS="[17500000,87500000,175000000]"

# Flat cell list: lang:arm:seed
CELLS=()
for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS_ORDER[@]}"; do
    for lang in ${ARM_LANGS[$arm]}; do
      CELLS+=("${lang}:${arm}:${seed}")
    done
  done
done
echo "Total cells: ${#CELLS[@]}"

# ---- Preflight: tokenizer artifacts must exist for every arm that needs one.
# The tur/fin/hun langspec overlays are being refit by a concurrent process
# (scripts/prep_f15_tokenizers.py, a separate agent/session -- see the S1
# note in the header) at the time this launcher may be exercised, so
# --dry-run tolerates a missing artifact as a WARNING (useful for sanity-
# checking cell/job packing while that fit is still in flight). A REAL
# submission still hard-fails on ANY missing artifact -- no cell is ever
# silently launched against a tokenizer that does not exist yet. ----
missing_artifact() {
  local f="$1" hint="$2"
  if [[ "${DRY_RUN}" == "--dry-run" ]]; then
    echo "WARN (dry-run): missing ${f} (${hint}) -- would FATAL on a real submission."
  else
    echo "FATAL: missing ${f} (${hint})"
    exit 1
  fi
}
[[ -f "${TOK_ROOT}/manifest.json" ]] || missing_artifact "${TOK_ROOT}/manifest.json" "run scripts/prep_f15_tokenizers.py"
[[ -f "${TOK_ROOT}/base_tokenizer.json" ]] || missing_artifact "${TOK_ROOT}/base_tokenizer.json" "A5 needs it"
for lang in ${ARM_LANGS[langspec]}; do
  f="${TOK_ROOT}/${lang}/langspec_${lang}${LANG_SUFFIX[$lang]}.json"
  [[ -f "${f}" ]] || missing_artifact "${f}" "run scripts/prep_f15_tokenizers.py"
done
for lang in ${ARM_LANGS[placebo]}; do
  f="${TOK_ROOT}/${lang}/placebo_${lang}${LANG_SUFFIX[$lang]}.json"
  [[ -f "${f}" ]] || missing_artifact "${f}" "run scripts/prep_f15_tokenizers.py"
done
for lang in ${ARM_LANGS[vocab_extend]}; do
  f="${TOK_ROOT}/${lang}/vocabext_${lang}${LANG_SUFFIX[$lang]}.json"
  [[ -f "${f}" ]] || missing_artifact "${f}" "run scripts/build_f15_vocabext.py"
done

# ---- Preflight (fail-loud, no silent skips): refuse to launch any cell whose
# derived run dir already exists. resume_policy=never would abort anyway, but
# only after tokenization + dataset load; catching it here is free. ----
declare -A MODE_PREFIX=(
  [original]="ft"
  [langspec]="ft-las-langspec"
  [placebo]="ft-las-placebo"
  [base]="ft-las-base"
  [vocab_extend]="ft-vocabext"
)
preflight_fail=0
for cell in "${CELLS[@]}"; do
  IFS=':' read -r lang arm seed <<< "${cell}"
  tag="qwen2_5-0_5b_${lang}_${arm}_truetie_s${seed}"
  derived="${OUT_ROOT}/${tag}/${MODE_PREFIX[$arm]}_${lang}-eng_100M_lr2e-04_r16_uf_truetie"
  if [[ -d "${derived}" ]]; then
    echo "PREFLIGHT: ${derived} already exists — investigate/clean before relaunching this cell."
    preflight_fail=1
  fi
done
[[ ${preflight_fail} -eq 0 ]] || exit 1

# Pack 4 cells per node job (one cell per GPU).
i=0
job_idx=0
: > "${JID_FILE}.tmp"
while [[ $i -lt ${#CELLS[@]} ]]; do
  batch=("${CELLS[@]:$i:4}")
  i=$((i+4)); job_idx=$((job_idx+1))
  name="f15hp_${job_idx}"

  inner=""
  gpu=0
  for cell in "${batch[@]}"; do
    IFS=':' read -r lang arm seed <<< "${cell}"
    tag="qwen2_5-0_5b_${lang}_${arm}_truetie_s${seed}"
    cell_out="${OUT_ROOT}/${tag}"
    suffix="${LANG_SUFFIX[$lang]}"
    case "${arm}" in
      original)
        las_args="las_mode=original las_tokenizer_path=null"
        ;;
      base)
        # base_tokenizer.json lives at the family root, not a per-lang subdir.
        las_args="las_mode=base las_tokenizer_path=${TOK_ROOT}"
        ;;
      langspec|placebo|vocab_extend)
        las_args="las_mode=${arm} las_tokenizer_path=${TOK_ROOT}/${lang}"
        ;;
      *)
        echo "FATAL: unhandled arm ${arm}"; exit 1
        ;;
    esac
    inner+="
CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} ${REPO_DIR}/run_finetune.py \\
  model=qwen2_5-0_5b lora=r16-k32 ${las_args} \\
  langs=\"{${lang}:1.0,eng:0.2}\" \\
  dataset.size=100M dataset.max_seq_len=1024 dataset.local_dir=null \\
  dataset.lang_suffix=${suffix} \\
  seed=${seed} epochs=1 learning_rate=2e-4 val_fraction=0.05 \\
  use_lora=true unfreeze_embeddings=true true_tie=true resume_policy=never \\
  logger.enabled=false \\
  \"eval_token_budgets=${EVAL_TOKEN_BUDGETS}\" \\
  output_dir=${cell_out} \\
  hydra.run.dir=${cell_out}/hydra \\
  > ${LOG_DIR}/${tag}_\${SLURM_JOB_ID}.log 2>&1 &
PIDS+=(\$!); TAGS+=(${tag})
"
    gpu=$((gpu+1))
  done

  wrap="
set -uo pipefail
cd ${REPO_DIR} || exit 1
export HF_HOME=${HF_CACHE}
export HF_HUB_OFFLINE=1
export PYTHONPATH=${REPO_DIR}:\${PYTHONPATH:-}
# 4 cells share one node: cap per-cell tokenization/BLAS threads.
export RAYON_NUM_THREADS=32
export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=true
PIDS=(); TAGS=()
${inner}
rc=0
for k in \${!PIDS[@]}; do
  if ! wait \${PIDS[\$k]}; then echo \"CELL FAILED: \${TAGS[\$k]}\"; rc=1; fi
done
exit \$rc
"
  # Persist the exact generated batch payload for provenance.
  printf '%s\n' "${wrap}" > "${LOG_DIR}/${name}.sbatch"
  if [[ "${DRY_RUN}" == "--dry-run" ]]; then
    echo "--- job ${name}: ${batch[*]} (payload: ${LOG_DIR}/${name}.sbatch)"
    continue
  fi
  # Walltime unchanged at 8h despite the larger budget: 3561 steps/cell is
  # ~0.61x F14c's per-epoch step cost (F14c ran epochs=1 with no max_steps
  # cap), so the new position budgets still fit comfortably inside the prior
  # 8h allowance -- not a coincidence, just headroom that was already there.
  jid=$(sbatch --parsable \
    --job-name="${name}" \
    --account=ACCOUNT --partition=normal \
    --nodes=1 --ntasks=1 --cpus-per-task=288 --mem=450G \
    --time=08:00:00 \
    --output="${LOG_DIR}/${name}_%j.out" --error="${LOG_DIR}/${name}_%j.err" \
    --uenv=pytorch/v2.8.0:v1 --view=default \
    --wrap="${wrap}")
  echo "${jid} ${name} ${batch[*]}" | tee -a "${JID_FILE}.tmp"
done
if [[ "${DRY_RUN}" != "--dry-run" ]]; then
  mv "${JID_FILE}.tmp" "${JID_FILE}"
  echo "Job IDs recorded in ${JID_FILE}"
fi
