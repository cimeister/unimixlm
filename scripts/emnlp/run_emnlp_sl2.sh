#!/bin/bash
# ============================================================================
# run_emnlp_sl2.sh — Wilkes3 SL2 wrapper for EMNLP finetuning / eval jobs.
#
# SL2 limits respected:
#   - <= 64 GPUs in use at any one time (this job uses 4 — 1 ampere node)
#   - <= 36 h wallclock per job
#
# Per-job shape:
#   1 ampere node, 4 A100s. Used for any Core-suite job and any cell that
#   does not fit in the SL3 12 h cap (chiefly Olmo3-7B / unfrozen / 500M /
#   5 epochs, ~30 h).
#
# Usage:
#   sbatch --export=ALL,EXP_KIND=ft,EXP_DOMAIN=code,EXP_MODEL=qwen2_5-0_5b,\
#       EXP_LANG=python,EXP_LAS_MODE=langspec,EXP_VOCAB=256k,\
#       EXP_UNFREEZE=1,EXP_DATA_SIZE=100M,EXP_LR=2e-4,\
#       EXP_LORA_PRESET=r64-k128,EXP_EPOCHS=5,HF_TOKEN="$HF_TOKEN" \
#       scripts/emnlp/run_emnlp_sl2.sh
# ============================================================================
#SBATCH -J emnlp-sl2
#SBATCH -A BUTTERY-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=36:00:00
#SBATCH --mail-type=FAIL
#SBATCH --no-requeue
#SBATCH -p ampere
#SBATCH -o logs/emnlp/%x-sl2-%j.out
#SBATCH -e logs/emnlp/%x-sl2-%j.err

set -euo pipefail

# ── Modules ─────────────────────────────────────────────────────────────────
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

JOBID=${SLURM_JOB_ID:-local}
echo "JobID:  $JOBID"
echo "Time:   $(date)"
echo "Host:   $(hostname)"
echo "Node:   ${SLURM_JOB_NODELIST:-n/a}"

# ── Project root ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
echo "PWD:    $(pwd)"

mkdir -p logs/emnlp

# ── venv: prefer repo convention, fall back to user template layout ─────────
if   [ -f venvs/demo/bin/activate ]; then
    # shellcheck disable=SC1091
    source venvs/demo/bin/activate
elif [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "ERROR: no venv found (expected venvs/demo or .venv)" >&2
    exit 1
fi
echo "Python: $(which python)  ($(python --version 2>&1))"

# ── Required env ────────────────────────────────────────────────────────────
: "${HF_TOKEN:?HF_TOKEN must be set (sbatch --export=ALL,HF_TOKEN=...)}"
: "${EXP_KIND:?EXP_KIND must be 'ft' or 'eval'}"
: "${EXP_MODEL:?EXP_MODEL must be set (qwen2_5-0_5b|gemma3-1b|olmo3-1025-7b)}"
: "${EXP_DOMAIN:?EXP_DOMAIN must be 'code' or 'nl'}"

# ── Pipeline env ────────────────────────────────────────────────────────────
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"

# 35 h cumulative budget for the in-job timeout wrapper.
export WALLCLOCK_BUDGET_SEC="${WALLCLOCK_BUDGET_SEC:-126000}"

# Experiment axes (defaults safe; override via --export=ALL,EXP_*=...).
export EXP_LANG="${EXP_LANG:-python}"
export EXP_REHEARSAL="${EXP_REHEARSAL:-eng}"
export EXP_LAS_MODE="${EXP_LAS_MODE:-langspec}"   # original | langspec
export EXP_VOCAB="${EXP_VOCAB:-256k}"             # 100k | 150k | 256k
export EXP_UNFREEZE="${EXP_UNFREEZE:-1}"          # 0 | 1
export EXP_LR="${EXP_LR:-2e-4}"
export EXP_DATA_SIZE="${EXP_DATA_SIZE:-100M}"     # 10M|50M|100M|500M
export EXP_LORA_PRESET="${EXP_LORA_PRESET:-r64-k128}"
export EXP_EPOCHS="${EXP_EPOCHS:-5}"

echo "EXP: kind=$EXP_KIND domain=$EXP_DOMAIN model=$EXP_MODEL lang=$EXP_LANG"
echo "     rehearsal=$EXP_REHEARSAL las_mode=$EXP_LAS_MODE vocab=$EXP_VOCAB"
echo "     unfreeze=$EXP_UNFREEZE lr=$EXP_LR data=$EXP_DATA_SIZE"
echo "     lora=$EXP_LORA_PRESET epochs=$EXP_EPOCHS"
echo "WALLCLOCK_BUDGET_SEC=${WALLCLOCK_BUDGET_SEC}"

# ── In-job supervisor: retry on transient HF / network failures ─────────────
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
RESTART_BACKOFF_SEC="${RESTART_BACKOFF_SEC:-30}"
PAYLOAD="scripts/emnlp/_payload_${EXP_KIND}.sh"

if [ ! -f "${PAYLOAD}" ]; then
    echo "ERROR: payload script not found: ${PAYLOAD}" >&2
    exit 2
fi

attempt=0
while (( attempt < MAX_ATTEMPTS )); do
    attempt=$(( attempt + 1 ))
    echo "================ attempt ${attempt}/${MAX_ATTEMPTS} ================"
    set +e
    timeout "${WALLCLOCK_BUDGET_SEC}" bash "${PAYLOAD}"
    rc=$?
    set -e
    if (( rc == 0 )); then
        echo "${PAYLOAD} succeeded on attempt ${attempt}"
        exit 0
    fi
    echo "${PAYLOAD} failed (rc=${rc}); sleeping ${RESTART_BACKOFF_SEC}s before retry"
    sleep "${RESTART_BACKOFF_SEC}"
    # On retry, instruct payload to resume from the latest HF / trainer ckpt.
    export RESUME_FROM_CHECKPOINT=1
done

echo "FATAL: gave up after ${MAX_ATTEMPTS} attempts" >&2
exit 1
