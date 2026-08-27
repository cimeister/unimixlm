#!/usr/bin/env bash
# =============================================================================
# Single-language code finetuning with LAS langspec tokenizers.
#
# Runs ONE language at a time in the FOREGROUND so you can see output/errors.
# For batch runs across 8 GPUs, use run_code_finetuning.sh instead.
#
# Usage:
#   bash scripts/run_single_code_finetune.sh rust
#   bash scripts/run_single_code_finetune.sh rust langspec   # LAS tokenizer (default)
#   bash scripts/run_single_code_finetune.sh rust original   # BPE baseline
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_code_finetune.sh python
# =============================================================================

set -euo pipefail

LANG="${1:?Usage: $0 <language> [langspec|original]}"
MODE="${2:-langspec}"

# Rehearsal language: Python by default, JavaScript when target is Python.
if [ "$LANG" = "python" ]; then REHEARSAL="javascript"; else REHEARSAL="python"; fi

echo "========================================"
echo "Code finetuning: ${LANG} (${MODE})"
echo "  Rehearsal: ${REHEARSAL}"
echo "  Mix: 80% ${LANG}, 20% ${REHEARSAL}"
echo "========================================"
hostname
date

# Verify tokenizers exist
TOK_DIR="./olmo3_las_tokenizers/las-code-tokenizers-OLMo-3-1025-7B"
MODEL_TOK_DIR="./olmo3_las_tokenizers/model_tokenizer"

if [ "$MODE" = "langspec" ]; then
    TOK_FILE="${TOK_DIR}/langspec_${LANG}.json"
    if [ ! -f "$TOK_FILE" ]; then
        echo "ERROR: Tokenizer not found: ${TOK_FILE}"
        echo "Available tokenizers:"
        ls "${TOK_DIR}"/langspec_*.json 2>/dev/null | head -10
        exit 1
    fi
    echo "  Tokenizer: ${TOK_FILE} ($(du -h "$TOK_FILE" | cut -f1))"
fi

if [ ! -d "$MODEL_TOK_DIR" ]; then
    echo "ERROR: Model tokenizer directory not found: ${MODEL_TOK_DIR}"
    echo "Run: bash scripts/download_code_tokenizers.sh"
    exit 1
fi
echo "  Model tokenizer: ${MODEL_TOK_DIR}"

# Verify dataset exists on HF
echo "  Dataset: MultilingualUnigramLM/LangMap-TheStack-${LANG}-100M"

# Reduce CUDA fragmentation (recommended by PyTorch for large models)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ""
echo "Starting finetuning..."
echo ""

python3 run_finetune.py \
    --config-name=code_default \
    "+langs={${LANG}: 0.8, ${REHEARSAL}: 0.2}" \
    "las_tokenizer_path=${TOK_DIR}" \
    "model_tokenizer_path=${MODEL_TOK_DIR}" \
    "dataset.size=100M" \
    "las_mode=${MODE}"

# ─── Upload to HuggingFace ────────────────────────────────────────────────────

if [ "$MODE" = "langspec" ]; then
    OUT_DIR="./finetuned_model/ft-code-las-langspec_${LANG}-${REHEARSAL}_100M"
    HF_REPO="MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-${LANG}"
else
    OUT_DIR="./finetuned_model/ft-code_${LANG}-${REHEARSAL}_100M"
    HF_REPO="MultilingualUnigramLM/ft-code-olmo3-7b-thestack100M-${LANG}"
fi

echo ""
echo "Uploading to HuggingFace: ${HF_REPO}"
hf repo create "${HF_REPO}" 2>/dev/null || true
hf upload "${HF_REPO}" "${OUT_DIR}" --repo-type model

echo ""
echo "========================================"
echo "Finetuning complete for ${LANG} (${MODE})"
echo "  Output: ${OUT_DIR}"
echo "  HF repo: https://huggingface.co/${HF_REPO}"
date
echo "========================================"
