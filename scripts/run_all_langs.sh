#!/usr/bin/env bash
set -euo pipefail

LANG_FILE="$(dirname "$0")/../config/tokenizer_training/languages/all.yaml"

grep -oP '^\s*-\s+\K\S+' "$LANG_FILE" | while read -r lang; do
    echo "=== Running for $lang ==="
    python3 run_las_tok_training.py "languages.langs=[$lang]" hf.push=true
done
