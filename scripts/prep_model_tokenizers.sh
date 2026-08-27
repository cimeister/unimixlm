#!/usr/bin/env bash
# prep_model_tokenizers.sh — snapshot each base model's HF tokenizer to
# ./_modeltok_<config>  so the `original`-arm runs (which load
# model_tokenizer_path as a directory) work WITHOUT any LAS overlay.
# Run once on the pod after `huggingface-cli login`. Cheap, no GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

declare -A MODELS=(
  [qwen2_5-1_5b]="Qwen/Qwen2.5-1.5B"
  [gemma2-2b]="google/gemma-2-2b"
  [mistral-7b-v0_3]="mistralai/Mistral-7B-v0.3"
)
for cfg in "${!MODELS[@]}"; do
  out="./_modeltok_${cfg}"
  if [[ -f "${out}/tokenizer.json" || -f "${out}/tokenizer_config.json" ]]; then
    echo "[skip] $out exists"; continue
  fi
  echo "[snap] ${MODELS[$cfg]} -> $out"
  python3 - "$cfg" "${MODELS[$cfg]}" "$out" <<'PY'
import sys
from transformers import AutoTokenizer
cfg, name, out = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
tok.save_pretrained(out)
print(f"saved {name} tokenizer -> {out}")
PY
done
echo "Done. original-arm runs can now use model_tokenizer_path=./_modeltok_<config>"
