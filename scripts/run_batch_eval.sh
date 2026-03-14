#!/usr/bin/env bash

hostname
date

mkdir -p logs

langs=(ibo fil lit swh zsm ron ind yor)

for lang in "${langs[@]}"; do
	CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
        python3 run_eval.py \
            "model_lang=${lang}" \
            &> "logs/eval_full_${lang}.txt"

done

date
