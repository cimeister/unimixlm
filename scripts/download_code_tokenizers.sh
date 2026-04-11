#!/usr/bin/env bash
# Download code LAS tokenizers and the OLMo-2 model tokenizer from HuggingFace.
# Run this before code finetuning to set up the local tokenizer directory.

set -euo pipefail

DEST_DIR="./code_las_tokenizers"
mkdir -p "$DEST_DIR"

echo "=== Downloading OLMo-2-1124-7B model tokenizer ==="
huggingface-cli download allenai/OLMo-2-1124-7B \
    --include "tokenizer*" "special_tokens_map*" \
    --local-dir "$DEST_DIR/model_tokenizer"

echo "=== Downloading code LAS tokenizers ==="
huggingface-cli download MultilingualUnigramLM/las-code-tokenizers-OLMo-2-1124-7B-ass_bat_c_plus27 \
    --local-dir "$DEST_DIR/las-code-tokenizers-OLMo-2-1124-7B"

echo "=== Verifying downloads ==="
echo "Model tokenizer files:"
ls -la "$DEST_DIR/model_tokenizer/"

echo ""
echo "Code LAS tokenizer files:"
ls -la "$DEST_DIR/las-code-tokenizers-OLMo-2-1124-7B/"

# Quick sanity check
N_LANGSPEC=$(ls "$DEST_DIR/las-code-tokenizers-OLMo-2-1124-7B"/langspec_*.json 2>/dev/null | wc -l)
echo ""
echo "Found $N_LANGSPEC language-specific tokenizers."
if [ "$N_LANGSPEC" -ge 20 ]; then
    echo "OK: Expected ~30 langspec tokenizers."
else
    echo "WARNING: Expected ~30 langspec tokenizers, found only $N_LANGSPEC."
fi

echo ""
echo "=== Done. Tokenizers ready at: $DEST_DIR ==="
