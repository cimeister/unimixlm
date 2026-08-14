#!/usr/bin/env bash
# run_f14_eval_login.sh — MultiBLiMP eval of F14/F14b cells on login-node GPUs.
# 4 parallel shards (one per GPU).
# Wiring (2026-08-14 user decision): training wiring only — frozen-original
# output head, the configuration the optimizer actually saw.
# Usage: bash scripts/run_f14_eval_login.sh [family] [wiring] [cells_root] [out_dir]
set -euo pipefail

REPO_DIR="/home/USER/unimixlm"
FAMILY="${1:-f14}"
WIRING="${2:-training}"
CELLS_ROOT="${3:-/path/to/scratch/USER/pr21_rerun}"
OUT_DIR="${4:-${REPO_DIR}/eval_results/f14_multiblimp}"
VENV=/path/to/scratch/USER/unimixlm-eval-venv-aug26
LOG_DIR="${REPO_DIR}/logs/pr21_rerun"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# Online HF (task datasets may not be cached); shared cache dir.
for gpu in 0 1 2 3; do
  nohup uenv run pytorch/v2.8.0:v1 --view=default -- env \
    HF_HOME=/path/to/scratch/USER/hf_cache_pr21 \
    PYTHONPATH="${REPO_DIR}" \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    "${VENV}/bin/python" "${REPO_DIR}/scripts/eval_f14_multiblimp.py" \
      --cells-root "${CELLS_ROOT}" --out "${OUT_DIR}" --family "${FAMILY}" \
      --head-wiring "${WIRING}" --shard "${gpu}/4" --batch-size 16 \
    > "${LOG_DIR}/f14eval_${FAMILY}_shard${gpu}.log" 2>&1 &
  echo "shard ${gpu}/4 pid $!"
done
echo "Logs: ${LOG_DIR}/f14eval_${FAMILY}_shard{0..3}.log ; results: ${OUT_DIR}"
