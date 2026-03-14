#!/usr/bin/env bash

hostname
date

mkdir -p logs

langs=(yor ibo fil lit swh zsm ron ind yor)

for lang in "${langs[@]}"; do
	CUDA_VISIBLE_DEVICES=1 \
        python3 run_finetune.py \
            '~langs' '+langs={'"${lang}"': 1.0, eng: 0.2}' \
            "las_mode=original" \
            "dataset.size=10M" \
        &> "logs/${lang}_orig_10M.txt"
done

date
