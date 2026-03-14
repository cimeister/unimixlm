#!/usr/bin/env bash
#SBATCH --job-name=ft-las
#SBATCH --output=logs/%x_%a.out
#SBATCH --error=logs/%x_%a.err
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-7

langs=(ibo fil lit swh zsm ron ind yor)

lang="${langs[$SLURM_ARRAY_TASK_ID]}"

hostname
date

mkdir -p logs

echo "Running LAS finetuning for lang=${lang} (task ${SLURM_ARRAY_TASK_ID})"

python3 run_finetune.py \
    '~langs' '+langs={'"${lang}"': 1.0, eng: 0.2}' \
    "las_tokenizer_path=/home/as3623/unimixlm/olmo3_las_tokenizers/las-tokenizers-Olmo-3-1025-7B-${lang}" \
    las_mode="langspec" \
    "dataset.size=100M" \
    &> "logs/${lang}_las_10M.txt"

date
