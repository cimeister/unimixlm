#!/usr/bin/env bash
# launch_pr21_truetie.sh — F14c: true-tie pretrained baseline.
# The intended protocol (fully trainable tied embedding, peft
# ensure_weight_tying) with pretrained embeddings. Serves as (i) the
# wiring-matched baseline for the F14b re-init gap and (ii) the first
# replication of tab:extrinsic under the protocol the paper intends.
#
# Grid: {tur,fin,hun} x {original,langspec} x seeds {42,43,44} = 18 cells.
# Protocol matches suchir-code-tok _run_cell4.sh except:
#   - finetune.py is the mainline tie-preserving version (bug-4 fixed):
#     tied Qwen2.5 embeddings stay tied; modules_to_save=[embed_tokens] only.
#   - checkpointing: final model per epoch only (no 10-step cadence).
#   - placebo arm added (never actually ran on the branch).
# Everything else identical: data repos, tokenizers (prep_manifest.json),
# lr 2e-4, LoRA r16-k32 + unfrozen embeddings, eff. batch 48, 1 epoch,
# max_seq_len 1024, val_fraction 0.05, langs {lang:1.0, eng:0.2}.
#
# Usage: bash scripts/launch_pr21_rerun.sh [--dry-run]
set -euo pipefail

REPO_DIR="/home/USER/unimixlm"
PYTHON="/path/to/scratch/USER/unimixlm-venv-aug26/bin/python"
TOK_ROOT="${REPO_DIR}/las_tokenizers/qwen2_5-0_5b_pr21"
OUT_ROOT="/path/to/scratch/USER/pr21_rerun"
HF_CACHE="/path/to/scratch/USER/hf_cache_pr21"
LOG_DIR="${REPO_DIR}/logs/pr21_rerun"
JID_FILE="${REPO_DIR}/.pr21_truetie_jids"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

DRY_RUN="${1:-}"

LANGS=(tur fin hun)
ARMS=(original langspec)
SEEDS=(42 43 44)

# Flat cell list: lang:arm:seed
CELLS=()
for seed in "${SEEDS[@]}"; do
  for lang in "${LANGS[@]}"; do
    for arm in "${ARMS[@]}"; do
      CELLS+=("${lang}:${arm}:${seed}")
    done
  done
done
echo "Total cells: ${#CELLS[@]}"

# Preflight: tokenizer artifacts must exist for langspec/placebo arms.
for lang in "${LANGS[@]}"; do
  for f in "langspec_${lang}_Latn.json" "placebo_${lang}_Latn.json"; do
    [[ -f "${TOK_ROOT}/${lang}/${f}" ]] || {
      echo "FATAL: missing ${TOK_ROOT}/${lang}/${f} (run prep_qwen25_las_rerun_tokenizers.py)"; exit 1; }
  done
done
[[ -f "${TOK_ROOT}/prep_manifest.json" ]] || { echo "FATAL: missing prep_manifest.json"; exit 1; }
grep -q '"placebo_mode": "bucketed"' "${TOK_ROOT}/prep_manifest.json" || {
  echo "FATAL: prep_manifest.json does not record bucketed placebos — regenerate with current prep script"; exit 1; }

# Preflight (fail-loud, no silent skips): refuse to launch any cell whose
# derived run dir already exists. resume_policy=never would abort anyway,
# but only after ~30 min of tokenization; catching it here is free.
declare -A MODE_PREFIX=( [original]="ft" [langspec]="ft-las-langspec" [placebo]="ft-las-placebo" )
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

# Pack 4 cells per node job.
i=0
job_idx=0
: > "${JID_FILE}.tmp"
while [[ $i -lt ${#CELLS[@]} ]]; do
  batch=("${CELLS[@]:$i:4}")
  i=$((i+4)); job_idx=$((job_idx+1))
  name="pr21tt_${job_idx}"

  # Build the per-node script: one cell per GPU, in parallel.
  inner=""
  gpu=0
  for cell in "${batch[@]}"; do
    IFS=':' read -r lang arm seed <<< "${cell}"
    tag="qwen2_5-0_5b_${lang}_${arm}_truetie_s${seed}"
    cell_out="${OUT_ROOT}/${tag}"
    if [[ "${arm}" == "original" ]]; then
      las_args="las_mode=original las_tokenizer_path=null"
    else
      las_args="las_mode=${arm} las_tokenizer_path=${TOK_ROOT}/${lang}"
    fi
    inner+="
CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} ${REPO_DIR}/run_finetune.py \\
  model=qwen2_5-0_5b lora=r16-k32 ${las_args} \\
  langs=\"{${lang}:1.0,eng:0.2}\" \\
  dataset.size=100M dataset.max_seq_len=1024 dataset.local_dir=null \\
  seed=${seed} epochs=1 learning_rate=2e-4 val_fraction=0.05 \\
  use_lora=true unfreeze_embeddings=true true_tie=true resume_policy=never \\
  logger.enabled=false \\
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
