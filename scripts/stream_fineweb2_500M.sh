#!/usr/bin/env bash
# =============================================================================
# Stream 500M tokens/language of Finnish, Hungarian, Turkish from FineWeb-2
# and push one HF dataset repo per language under MultilingualUnigramLM.
#
# Memory-bounded: batched tokenization + incremental sharded parquet writes
# (see fineweb2_finetune_dataset.py --per-language-repos).
#
# Designed to run unattended inside tmux:
#   tmux new -d -s fw2_500m 'bash scripts/stream_fineweb2_500M.sh'
#   tmux attach -t fw2_500m
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# HF auth lives in the shared .hf token store on this box.
export HF_HOME="${HF_HOME:-/local/scratch/sas245/.hf}"

PYTHON="${PYTHON:-/local/scratch/sas245/unimixlm-sct/.venv/bin/python}"

TARGET_TOKENS="${TARGET_TOKENS:-500000000}"   # per language
LANGUAGES="${LANGUAGES:-fin_Latn hun_Latn tur_Latn}"
REPO_TEMPLATE="${REPO_TEMPLATE:-MultilingualUnigramLM/FineWeb2-500M-{lang}}"
TMP_DIR="${TMP_DIR:-tmp_fineweb2_500M}"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/stream_fineweb2_500M_$(date +%Y%m%d_%H%M%S).log"

echo "[stream_fineweb2_500M] python=$PYTHON" | tee -a "$LOG_FILE"
echo "[stream_fineweb2_500M] HF_HOME=$HF_HOME" | tee -a "$LOG_FILE"
echo "[stream_fineweb2_500M] languages=$LANGUAGES target=$TARGET_TOKENS/lang" | tee -a "$LOG_FILE"
echo "[stream_fineweb2_500M] repo_template=$REPO_TEMPLATE" | tee -a "$LOG_FILE"
echo "[stream_fineweb2_500M] log=$LOG_FILE" | tee -a "$LOG_FILE"

# shellcheck disable=SC2086
"$PYTHON" -u fineweb2_finetune_dataset.py \
  --per-language-repos \
  --hf-repo-template "$REPO_TEMPLATE" \
  --target-tokens "$TARGET_TOKENS" \
  --languages $LANGUAGES \
  --tmp-dir "$TMP_DIR" \
  2>&1 | tee -a "$LOG_FILE"

echo "[stream_fineweb2_500M] DONE rc=${PIPESTATUS[0]} at $(date)" | tee -a "$LOG_FILE"
