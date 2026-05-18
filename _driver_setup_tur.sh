#!/usr/bin/env bash
# Stage 1: environment setup -> tokenizer snapshots -> Turkish Qwen (priority).
set -euo pipefail
cd /workspace/unimixlm
source ./_driver_env.sh

echo "==== [$(date -u)] STAGE: setup.sh --create-venv ===="
bash setup.sh --create-venv 2>&1
source .venv/bin/activate

echo "==== [$(date -u)] STAGE: prep_model_tokenizers ===="
bash scripts/prep_model_tokenizers.sh 2>&1

echo "==== [$(date -u)] STAGE: Turkish Qwen2.5-1.5B (original arm) ===="
LANGS="tur" ARMS="original" bash scripts/run_langmap_qwen_initial.sh 2>&1

echo "==== [$(date -u)] STAGE 1 COMPLETE ===="
