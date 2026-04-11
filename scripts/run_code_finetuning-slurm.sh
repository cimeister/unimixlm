#!/usr/bin/env bash
#SBATCH --job-name=ft-code-las
#SBATCH --output=logs/%x_%a.out
#SBATCH --error=logs/%x_%a.err
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-7
#
# Fine-tune OLMo-2-7B on code with LAS langspec tokenizers (SLURM array).
# Each array task handles one programming language.
# Tier 1 languages prioritised by The Stack data size.

set -euo pipefail

# Tier 1 languages (>100GB in The Stack)
code_langs=(python javascript java c cpp csharp go typescript)

lang="${code_langs[$SLURM_ARRAY_TASK_ID]}"

hostname
date
mkdir -p logs

echo "Running LAS code finetuning for lang=${lang} (task ${SLURM_ARRAY_TASK_ID})"

# ─── LAS langspec mode ───────────────────────────────────────────────────────

python3 run_finetune.py \
    --config-name=code_default \
    "+langs={${lang}: 0.8, python: 0.2}" \
    "las_tokenizer_path=./code_las_tokenizers/las-code-tokenizers-OLMo-2-1124-7B" \
    "dataset.size=100M" \
    "las_mode=langspec" \
    &> "logs/${lang}_code_las_100M.txt"

# Upload LAS results
hf repo create "MultilingualUnigramLM/ft-code-las-olmo2-7b-thestack100M-${lang}" 2>/dev/null || true
hf upload "MultilingualUnigramLM/ft-code-las-olmo2-7b-thestack100M-${lang}" \
    "./finetuned_model/ft-code-las-langspec_${lang}-python_100M" --repo-type model

# ─── Original tokenizer baseline ─────────────────────────────────────────────

python3 run_finetune.py \
    --config-name=code_default \
    "+langs={${lang}: 0.8, python: 0.2}" \
    "las_mode=original" \
    "dataset.size=100M" \
    &> "logs/${lang}_code_orig_100M.txt"

# Upload original results
hf repo create "MultilingualUnigramLM/ft-code-olmo2-7b-thestack100M-${lang}" 2>/dev/null || true
hf upload "MultilingualUnigramLM/ft-code-olmo2-7b-thestack100M-${lang}" \
    "./finetuned_model/ft-code_${lang}-python_100M" --repo-type model

date
