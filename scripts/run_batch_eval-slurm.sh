#!/usr/bin/env bash
#SBATCH --job-name=eval-las
#SBATCH --output=logs/%x_%a.out
#SBATCH --error=logs/%x_%a.err
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-7

langs=(ibo fil lit swh zsm ron ind yor)

lang="${langs[$SLURM_ARRAY_TASK_ID]}"

hostname
date

mkdir -p logs

echo "Running evaluation for lang=${lang} (task ${SLURM_ARRAY_TASK_ID})"

PYTHONUNBUFFERED=1 \
    python3 run_eval.py \
        "model_lang=${lang}" \
    &> "logs/eval_full_${lang}.txt"

date
