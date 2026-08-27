#!/usr/bin/env bash
#SBATCH --job-name=ft-orig
#SBATCH --output=logs/%x_%a.out
#SBATCH --error=logs/%x_%a.err
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-7

langs=(ibo fil lit swh zsm ron ind yor)

lang="${langs[$SLURM_ARRAY_TASK_ID]}"

hostname
date

mkdir -p logs

echo "Running finetuning for lang=${lang} (task ${SLURM_ARRAY_TASK_ID})"

python3 run_finetune.py \
    '+langs={'"${lang}"': 0.8, eng: 0.2}' \
    "las_mode=original" \
    "dataset.size=100M" \
    &> "logs/${lang}_orig_100M.txt"

hf repo create MultilingualUnigramLM/ft-olmo3-7b-fineweb100M-${lang}
hf upload MultilingualUnigramLM/ft-olmo3-7b-fineweb100M-${lang} ./finetuned_model/ft_${lang}-eng_100M --repo-type model

date
