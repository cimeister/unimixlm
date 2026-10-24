#!/usr/bin/env bash
# run_f14_eval_login.sh — MultiBLiMP/Belebele-eng eval of F14/F14b/F15 cells
# on login-node GPUs. 4 parallel shards (one per GPU).
# Wiring (2026-08-14 user decision): training wiring only — frozen-original
# output head, the configuration the optimizer actually saw. (F15 note: for
# true-tie cells 'training' and 'reshared' are numerically identical, see
# eval_f14_multiblimp.py --head-wiring help.)
# Usage: bash scripts/run_f14_eval_login.sh [family] [wiring] [cells_root] \
#            [out_dir] [tasks] [model] [expect_cells]
#   family:       f14 | f14b | f15 | all   (default f14)
#   tasks:        comma list, e.g. multiblimp,belebele_eng   (default multiblimp)
#   model:        qwen2_5-0_5b | llama3_2-1b | all           (default all)
#   expect_cells: total cell count expected under cells_root after the
#                 family/model filter (pre-shard -- every shard checks the
#                 same total). Passed through to eval_f14_multiblimp.py's
#                 --expect-cells, which aborts the shard rather than silently
#                 evaluating a wrong/partial cell set. Optional: omit (leave
#                 unset/empty) for ad hoc invocations where the expected
#                 count isn't known up front.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH_DIR="${SCRATCH_DIR:?set SCRATCH_DIR to your cluster scratch root}"
FAMILY="${1:-f14}"
WIRING="${2:-training}"
CELLS_ROOT="${3:-${SCRATCH_DIR}/pr21_rerun}"
OUT_DIR="${4:-${REPO_DIR}/eval_results/f14_multiblimp}"
TASKS="${5:-multiblimp}"
MODEL="${6:-all}"
EXPECT_CELLS="${7:-}"
VENV="${VENV_DIR:?set VENV_DIR to the project virtualenv}"
LOG_DIR="${REPO_DIR}/logs/pr21_rerun"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

EXPECT_ARGS=()
if [[ -n "${EXPECT_CELLS}" ]]; then
  EXPECT_ARGS=(--expect-cells "${EXPECT_CELLS}")
fi

# Online HF (task datasets may not be cached); shared cache dir.
for gpu in 0 1 2 3; do
  nohup uenv run pytorch/v2.8.0:v1 --view=default -- env \
    HF_HOME="${SCRATCH_DIR}/hf_cache_pr21" \
    PYTHONPATH="${REPO_DIR}" \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    "${VENV}/bin/python" "${REPO_DIR}/scripts/eval_f14_multiblimp.py" \
      --cells-root "${CELLS_ROOT}" --out "${OUT_DIR}" --family "${FAMILY}" \
      --head-wiring "${WIRING}" --tasks "${TASKS}" --model "${MODEL}" \
      --shard "${gpu}/4" --batch-size 16 "${EXPECT_ARGS[@]}" \
    > "${LOG_DIR}/f14eval_${FAMILY}_shard${gpu}.log" 2>&1 &
  echo "shard ${gpu}/4 pid $!"
done
echo "Logs: ${LOG_DIR}/f14eval_${FAMILY}_shard{0..3}.log ; results: ${OUT_DIR}"
