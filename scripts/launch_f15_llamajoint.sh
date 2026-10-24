#!/usr/bin/env bash
# launch_f15_llamajoint.sh — F15: the 6 Llama-3.2-1B joint-multilingual cells
# (arms original/langspec x seeds 42-44) against the 5-overlay pool. Derived
# from launch_f15_wave2.sh; single-language cells removed (LLAMA_LANGS empty).
# Per the F15 experiment plan, sections "Models"
# ("Llama-3.2-1B (tied): arms A0/A1/A2 x 3 langs (tur, arb, hin) x 3 seeds"),
# "Joint-multilingual arm (C6)" and "Execution order" step 3 ("Wave 2: Llama
# cells + joint-multilingual").
#
# Grid:
#   (A) Llama-3.2-1B single-language:
#         arms {original, langspec} x langs {tur, arb, hin} x seeds {42,43,44}
#         = 18 cells.
#       A5 (base-Unigram) is NOT run here. The plan makes Llama A5 adaptive:
#       base ~= langspec is already established on OLMo-3, Qwen3 and
#       from-scratch, and wave 1 measures it once more true-tie on Qwen2.5;
#       a third family only runs if that Qwen A5-vs-A2 contrast is nonzero
#       beyond the corrected CI. A3 (placebo) and A4 (vocab_extend) are
#       Qwen-only by the same budget decision. A0 (no-adapt) is eval-only and
#       needs no training cell.
#   (B) Joint-multilingual (C6):
#         families {qwen2_5-0_5b, llama3_2-1b} x arms {original, langspec}
#         x seeds {42,43,44} = 12 cells, of which the 6 Llama cells are
#         currently BLOCKED (see the INCLUDE_LLAMA_JOINT block below).
#       One cell trains all 5 target languages at once,
#       langs={tur:0.2,fin:0.2,hun:0.2,arb:0.2,hin:0.2,eng:0.2}: each language
#       contributes a 20% document subsample of its own 100M-token corpus, so
#       the 5 target languages sum to roughly the same ~100M target-language
#       tokens a wave-1 single-language cell sees, split five ways, plus the
#       same eng:0.2 slice. Inference on these cells uses real routing over
#       the 5-overlay pool (LasTokenizerHFLM), which is the deployment setting
#       the paper's abstract describes.
#
# Protocol (identical to wave 1 except the batch factorization, below):
# true_tie, LoRA r16-k32, lr 2e-4, val_fraction 0.05, epochs=1,
# dataset.size=100M (HF hub, local_dir=null), max_seq_len=1024,
# resume_policy=never, logger disabled, HF cache offline.
#
# BATCH FACTORIZATION (the one deliberate difference from wave 1's command
# lines): effective batch is 48 for every wave-1 and wave-2 cell, but the
# factorization is per-model. config/finetune/model/llama3.2-1b.yaml declares
# per_device_batch_size=2 x gradient_accumulation_steps=16, which was written
# for the OLD 4-GPU protocol (2 x 4 GPUs x 16 = 128 effective) and yields only
# 2 x 1 x 16 = 32 on the single-GPU-per-cell layout used here. This launcher
# overrides the model group's two keys to 3 x 16 = 48 rather than editing that
# config file, so the 4-GPU protocol's file stays intact for any run that still
# uses it. Overriding `model.*` (not the top-level keys) keeps
# cfg.per_device_batch_size, which interpolates ${model.per_device_batch_size},
# in agreement with cfg.model.* inside run_metadata.json's cfg_snapshot.
#   Llama:  3 x 1 GPU x 16 x 1024 = 49,152 tokens/step
#   Qwen:   6 x 1 GPU x  8 x 1024 = 49,152 tokens/step  (model config, unchanged)
# The two families therefore run the SAME number of optimizer steps for the
# same EVAL_TOKEN_BUDGETS, which is what makes the budgets comparable across
# families as well as across arms.
#
# VRAM (GH200, 97,871 MiB/GPU, one cell per GPU), MEASURED on a login-node
# GPU rather than estimated, 2026-08-18: a Llama-3.2-1B langspec cell at
# per_device_batch_size=3, grad_accum=16, max_seq_len=1024 peaks at
# 21,473 MiB with a 2,055 MiB pre-existing baseline on that GPU, i.e. ~19.4 GB
# for the cell. The same cell at per_device_batch_size=2 peaks at ~14.2 GB.
# 19.4 GB against 96 GB leaves the 2 x 24 factorization contemplated during
# planning unnecessary. Training throughput measured in the same probe:
# 22.5k tokens/s at 3 x 16, so the 3561-step budget (175M padded positions)
# takes roughly 2.2 h of compute plus corpus load and tokenization. The 12 h
# walltime below therefore has a large margin, which is deliberate: login-GPU
# rates are measured under contention and compute-node rates may differ.
#
# EVAL_TOKEN_BUDGETS: padded training POSITIONS, not real target-language
# tokens — identical values and identical rationale as wave 1; see the long
# header note in scripts/launch_f15_wave1.sh before changing them. With
# tokens_per_step=49,152 the observation steps are
# [ceil(17.5e6/49152), ceil(87.5e6/49152), ceil(175e6/49152)] = [357,1781,3561]
# and max_steps=3561, for BOTH families.
#
# Usage: bash scripts/launch_f15_llamajoint.sh [--dry-run]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH_DIR="${SCRATCH_DIR:?set SCRATCH_DIR to your cluster scratch root}"
PYTHON="${VENV_DIR:?set VENV_DIR to the project virtualenv}/bin/python"
TOK_ROOT="${REPO_DIR}/las_tokenizers/f15"
OUT_ROOT="${SCRATCH_DIR}/f15_wave2"
# Same offline cache as wave 1. Llama-3.2-1B's weights were added to it on
# 2026-08-18 (prep_f15_tokenizers.py had cached only the tokenizer files);
# the preflight below re-checks that model.safetensors is present, because
# HF_HUB_OFFLINE=1 turns a cache miss into a mid-job failure.
HF_CACHE="${SCRATCH_DIR}/hf_cache_pr21"
LOG_DIR="${REPO_DIR}/logs/f15_wave2"
JID_FILE="${REPO_DIR}/.f15_llamajoint_jids"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

DRY_RUN="${1:-}"

SEEDS=(42 43 44)
ARMS=(original langspec)
LLAMA_LANGS=()
# Joint-multilingual language mix. Equal fifths of the 5 target languages
# (each 0.2 of its own 100M corpus) plus the same eng:0.2 slice every wave-1
# cell trains on.
JOINT_LANGS="{tur:0.2,fin:0.2,hun:0.2,arb:0.2,hin:0.2,eng:0.2}"
JOINT_LANG_DIRNAME="tur-fin-hun-arb-hin-eng"   # run_finetune.py's '-'.join(langs)

# ---- Llama joint-multilingual cells: UNBLOCKED 2026-08-24 ------------------
# All five langspec overlays and the pool dir now exist (fin/hun fit 2026-08-18,
# pool built+validated 2026-08-23); the flag below is 1 and the historical
# blocking rationale is preserved in launch_f15_wave2.sh.
INCLUDE_LLAMA_JOINT=1

# Script-suffix per language (ISO 15924). Single-language cells must pass the
# matching dataset.lang_suffix — las_experiment/finetune.py's
# resolve_lang_suffixes still hard-fails a single-target-language run whose
# declared suffix contradicts FINEWEB2_LANG_SUFFIX.
declare -A LANG_SUFFIX
LANG_SUFFIX[tur]="_Latn"; LANG_SUFFIX[fin]="_Latn"; LANG_SUFFIX[hun]="_Latn"
LANG_SUFFIX[arb]="_Arab"; LANG_SUFFIX[hin]="_Deva"

EVAL_TOKEN_BUDGETS="[17500000,87500000,175000000]"

declare -A MODE_PREFIX=(
  [original]="ft"
  [langspec]="ft-las-langspec"
)

# Hydra model-group name (file stem under config/finetune/model/) vs the tag
# token. They differ for Llama: the config file is llama3.2-1b.yaml while the
# tag uses llama3_2-1b, which is the key scripts/eval_f14_multiblimp.py's
# MODEL_MAP (and hence TAG_RE) matches on.
declare -A MODEL_GROUP=(
  [qwen2_5-0_5b]="qwen2_5-0_5b"
  [llama3_2-1b]="llama3.2-1b"
)
# Per-family batch overrides, applied to the model group so the interpolated
# top-level keys follow. Empty string = use the model config as-is.
declare -A BATCH_OVERRIDE=(
  [qwen2_5-0_5b]=""
  [llama3_2-1b]="model.per_device_batch_size=3 model.gradient_accumulation_steps=16"
)
# Per-family tokenizer roots.
declare -A FAM_TOK_ROOT=(
  [qwen2_5-0_5b]="${TOK_ROOT}/qwen2_5-0_5b"
  [llama3_2-1b]="${TOK_ROOT}/llama3.2-1b"
)
# Per-family HF model repo, for the offline-cache preflight.
declare -A FAM_HF_REPO=(
  [qwen2_5-0_5b]="models--Qwen--Qwen2.5-0.5B"
  [llama3_2-1b]="models--meta-llama--Llama-3.2-1B"
)
# Walltime: Llama-3.2-1B measures ~2x the per-step cost of Qwen2.5-0.5B
# (plan's rate table: ~6.5 vs ~3.5 GPU-h/cell), so its 3561 steps get the
# 12h maximum while the Qwen joint cells keep wave 1's 8h.
declare -A FAM_WALLTIME=(
  [qwen2_5-0_5b]="08:00:00"
  [llama3_2-1b]="12:00:00"
)

# ---- Cell list. Format: family:kind:lang:arm:seed  (lang="joint5" for kind=joint)
CELLS=()
for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do
    for lang in "${LLAMA_LANGS[@]}"; do
      CELLS+=("llama3_2-1b:single:${lang}:${arm}:${seed}")
    done
  done
done
JOINT_FAMILIES=()
if [[ "${INCLUDE_LLAMA_JOINT}" == "1" ]]; then
  JOINT_FAMILIES+=(llama3_2-1b)
fi
for seed in "${SEEDS[@]}"; do
  for fam in "${JOINT_FAMILIES[@]}"; do
    for arm in "${ARMS[@]}"; do
      CELLS+=("${fam}:joint:joint5:${arm}:${seed}")
    done
  done
done
echo "Total cells: ${#CELLS[@]}"
if [[ "${INCLUDE_LLAMA_JOINT}" != "1" ]]; then
  echo "BLOCKED: 6 llama3_2-1b joint cells excluded — las_tokenizers/f15/llama3.2-1b/{fin,hun}/langspec_*.json do not exist (see INCLUDE_LLAMA_JOINT above)."
fi

# ---- Preflight: artifacts. A real submission hard-fails on any miss; a
# --dry-run downgrades to a warning so cell/job packing can be checked while
# the tokenizer-prep pipeline is still producing overlays. ----
missing_artifact() {
  local f="$1" hint="$2"
  if [[ "${DRY_RUN}" == "--dry-run" ]]; then
    echo "WARN (dry-run): missing ${f} (${hint}) -- would FATAL on a real submission."
  else
    echo "FATAL: missing ${f} (${hint})"
    exit 1
  fi
}

# Models must be in the offline cache: the jobs run with HF_HUB_OFFLINE=1.
declare -A FAMS_USED=()
for cell in "${CELLS[@]}"; do
  IFS=':' read -r fam _ _ _ _ <<< "${cell}"
  FAMS_USED[$fam]=1
done
for fam in "${!FAMS_USED[@]}"; do
  repo_dir="${HF_CACHE}/hub/${FAM_HF_REPO[$fam]}"
  if ! compgen -G "${repo_dir}/snapshots/*/model*.safetensors" > /dev/null; then
    missing_artifact "${repo_dir}/snapshots/*/model*.safetensors" \
      "pre-download the model into HF_HOME=${HF_CACHE} while online"
  fi
  if ! compgen -G "${repo_dir}/snapshots/*/tokenizer.json" > /dev/null; then
    missing_artifact "${repo_dir}/snapshots/*/tokenizer.json" \
      "pre-download the tokenizer into HF_HOME=${HF_CACHE} while online"
  fi
done

# Llama single-language langspec overlays.
for lang in "${LLAMA_LANGS[@]}"; do
  f="${FAM_TOK_ROOT[llama3_2-1b]}/${lang}/langspec_${lang}${LANG_SUFFIX[$lang]}.json"
  [[ -f "${f}" ]] || missing_artifact "${f}" "run scripts/prep_f15_tokenizers.py"
done

# Joint-arm routing pools: base_tokenizer.json + all 5 langspec overlays in ONE
# directory, which is both what las_mode=langspec reads for a 5-language run
# and what LasTokenizerHFLM globs for routed eval.
for fam in "${JOINT_FAMILIES[@]}"; do
  pool="${FAM_TOK_ROOT[$fam]}/pool"
  [[ -f "${pool}/base_tokenizer.json" ]] || missing_artifact "${pool}/base_tokenizer.json" "build the family's pool dir"
  for lang in tur fin hun arb hin; do
    f="${pool}/langspec_${lang}${LANG_SUFFIX[$lang]}.json"
    [[ -f "${f}" ]] || missing_artifact "${f}" "build the family's pool dir"
  done
done
# Report (do not act on) overlays that would unblock the Llama joint cells.
if [[ "${INCLUDE_LLAMA_JOINT}" != "1" ]]; then
  llama_fin="${FAM_TOK_ROOT[llama3_2-1b]}/fin/langspec_fin_Latn.json"
  llama_hun="${FAM_TOK_ROOT[llama3_2-1b]}/hun/langspec_hun_Latn.json"
  if [[ -f "${llama_fin}" && -f "${llama_hun}" ]]; then
    echo "NOTE: llama3.2-1b fin+hun overlays now exist. Build ${FAM_TOK_ROOT[llama3_2-1b]}/pool/ and set INCLUDE_LLAMA_JOINT=1 to add the 6 Llama joint cells."
  fi
fi

# ---- Preflight: refuse to launch any cell whose derived run dir exists.
# resume_policy=never would abort anyway, but only after tokenization. ----
derived_dir_for() {   # family kind lang arm seed -> path
  local fam="$1" kind="$2" lang="$3" arm="$4" seed="$5"
  local tag="${fam}_${lang}_${arm}_truetie_s${seed}"
  local langdir
  if [[ "${kind}" == "joint" ]]; then langdir="${JOINT_LANG_DIRNAME}"; else langdir="${lang}-eng"; fi
  echo "${OUT_ROOT}/${tag}/${MODE_PREFIX[$arm]}_${langdir}_100M_lr2e-04_r16_uf_truetie"
}
preflight_fail=0
for cell in "${CELLS[@]}"; do
  IFS=':' read -r fam kind lang arm seed <<< "${cell}"
  derived="$(derived_dir_for "${fam}" "${kind}" "${lang}" "${arm}" "${seed}")"
  if [[ -d "${derived}" ]]; then
    echo "PREFLIGHT: ${derived} already exists — investigate/clean before relaunching this cell."
    preflight_fail=1
  fi
done
[[ ${preflight_fail} -eq 0 ]] || exit 1

# ---- Pack 4 cells per node job (one cell per GPU), never mixing families:
# walltime and per-cell cost differ between the 0.5B and 1B families, and a
# node job is only released when its slowest cell finishes. ----
ORDERED=()
for fam in llama3_2-1b qwen2_5-0_5b; do
  for cell in "${CELLS[@]}"; do
    # `if`, not `[[ ]] && append`: under `set -e` a false test as the loop's
    # last command would abort the script (it does not today only because the
    # final CELLS entry happens to match the final family, which stops being
    # true as soon as INCLUDE_LLAMA_JOINT=1 adds Llama joint cells last).
    if [[ "${cell}" == "${fam}:"* ]]; then
      ORDERED+=("${cell}")
    fi
  done
done
if [[ ${#ORDERED[@]} -ne ${#CELLS[@]} ]]; then
  echo "FATAL: family ordering dropped cells (${#ORDERED[@]} of ${#CELLS[@]}); a cell's family is not in the ordering list."
  exit 1
fi

job_idx=0
# Only a real submission touches the job-ID record; a dry run leaves no trace.
[[ "${DRY_RUN}" == "--dry-run" ]] || : > "${JID_FILE}.tmp"
i=0
while [[ $i -lt ${#ORDERED[@]} ]]; do
  # Take up to 4 consecutive cells of the SAME family.
  IFS=':' read -r batch_fam _ _ _ _ <<< "${ORDERED[$i]}"
  batch=()
  while [[ ${#batch[@]} -lt 4 && $i -lt ${#ORDERED[@]} ]]; do
    IFS=':' read -r cand_fam _ _ _ _ <<< "${ORDERED[$i]}"
    [[ "${cand_fam}" == "${batch_fam}" ]] || break
    batch+=("${ORDERED[$i]}")
    i=$((i+1))
  done
  job_idx=$((job_idx+1))
  name="f15lj_${job_idx}"

  inner=""
  gpu=0
  for cell in "${batch[@]}"; do
    IFS=':' read -r fam kind lang arm seed <<< "${cell}"
    tag="${fam}_${lang}_${arm}_truetie_s${seed}"
    cell_out="${OUT_ROOT}/${tag}"
    fam_tok="${FAM_TOK_ROOT[$fam]}"
    if [[ "${kind}" == "joint" ]]; then
      langs_arg="${JOINT_LANGS}"
      # Multi-script run: finetune.py resolves _Latn/_Arab/_Deva per language
      # from FINEWEB2_LANG_SUFFIX. dataset.lang_suffix is consulted only for
      # languages absent from that map, i.e. eng — whose hub slice is resolved
      # by its own branch and whose langspec-mode tokenizer is the model's own,
      # so this value is inert for these cells. Declared anyway because the
      # config key has no null-safe default.
      suffix="_Latn"
      las_path="${fam_tok}/pool"
    else
      langs_arg="{${lang}:1.0,eng:0.2}"
      suffix="${LANG_SUFFIX[$lang]}"
      las_path="${fam_tok}/${lang}"
    fi
    case "${arm}" in
      original)  las_args="las_mode=original las_tokenizer_path=null" ;;
      langspec)  las_args="las_mode=langspec las_tokenizer_path=${las_path}" ;;
      *) echo "FATAL: unhandled arm ${arm}"; exit 1 ;;
    esac
    inner+="
CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} ${REPO_DIR}/run_finetune.py \\
  model=${MODEL_GROUP[$fam]} lora=r16-k32 ${BATCH_OVERRIDE[$fam]} ${las_args} \\
  langs=\"${langs_arg}\" \\
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
  printf '%s\n' "${wrap}" > "${LOG_DIR}/${name}.sbatch"
  if [[ "${DRY_RUN}" == "--dry-run" ]]; then
    echo "--- job ${name} [${batch_fam}, ${FAM_WALLTIME[$batch_fam]}]: ${batch[*]} (payload: ${LOG_DIR}/${name}.sbatch)"
    continue
  fi
  jid=$(sbatch --parsable \
    --job-name="${name}" \
    --account="${SLURM_ACCOUNT:?set SLURM_ACCOUNT to your cluster account}" --partition=normal \
    --nodes=1 --ntasks=1 --cpus-per-task=288 --mem=450G \
    --time="${FAM_WALLTIME[$batch_fam]}" \
    --output="${LOG_DIR}/${name}_%j.out" --error="${LOG_DIR}/${name}_%j.err" \
    --uenv=pytorch/v2.8.0:v1 --view=default \
    --wrap="${wrap}")
  echo "${jid} ${name} ${batch[*]}" | tee -a "${JID_FILE}.tmp"
done
if [[ "${DRY_RUN}" != "--dry-run" ]]; then
  mv "${JID_FILE}.tmp" "${JID_FILE}"
  echo "Job IDs recorded in ${JID_FILE}"
fi

# TAG NOTE (eval side, not this launcher's to fix): the single-language tags
# llama3_2-1b_<lang>_<arm>_truetie_s<seed> already match
# scripts/eval_f14_multiblimp.py's TAG_RE (MODEL_MAP has llama3_2-1b, LANG_SUFFIX
# has tur/arb/hin). The joint tags <family>_joint5_<arm>_truetie_s<seed> do NOT:
# TAG_RE's language alternation is built from that script's 5-language
# LANG_SUFFIX map, so "joint5" fails to match and those cells would be skipped
# by the eval driver until it grows a joint/routed mode (plan work item 5).
