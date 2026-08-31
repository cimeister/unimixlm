#!/bin/bash
# Build code-A val caches (purged from scratch) on a COMPUTE node (login-node
# cgroup OOMs on the StarCoder parquet reads) and compute per-language eval-mode
# validation BpB for the 4 code-A SET-A models (python/js/java; eng excluded by
# user choice). Caches are reconstructed deterministically via --build_missing
# (byte-identical to training; NOT a data substitution).
#
# No default SLURM account is set below (a submission-time literal can't be
# made fail-loud the way a shell variable can): submit with
#   sbatch --account=<your account> scripts/run_code_val_slurm.sh
#SBATCH --job-name=fs-codeval
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=220G
#SBATCH --time=03:00:00
#SBATCH --uenv=pytorch/v2.8.0:v1
#SBATCH --view=default
#SBATCH --output=logs/fs-codeval_%j.out
#SBATCH --error=logs/fs-codeval_%j.err
set -euo pipefail

REPO="${REPO_DIR:?set REPO_DIR to the unimixlm repo root (dirname \$0 is not reliable under sbatch)}"
VENV="${VENV_DIR:?set VENV_DIR to the project virtualenv}"
MROOT="${SCRATCH_DIR:?set SCRATCH_DIR to your cluster scratch root}/unimix_fromscratch_morph"
OUT=${REPO}/eval_results/fromscratch_val_perlang
PY=${VENV}/bin/python
cd ${REPO}

for tok in bpe base pathpiece langspec; do
  echo "=== code_A_${tok} (build+eval) ==="
  srun -ul --export=ALL \
    env PYTHONPATH="${REPO}" \
    ${PY} scripts/eval_fromscratch_val_perlang.py \
      --model_dir ${MROOT}/fs_code_A_v50k_${tok}_seed42 \
      --langs python,javascript,java --build_missing \
      --output_json ${OUT}/code_A_${tok}.json
done
echo "ALL_CODE_VAL_DONE"
