#!/usr/bin/env bash

hostname
date

mkdir -p logs

langs=(yor ibo fil lit swh zsm ron ind) 

for lang in "${langs[@]}"; do
	CUDA_VISIBLE_DEVICES=1 \
        python3 las_finetune.py \
            "langs=[${lang}]" \
            "dataset.size=10M" \
        &> "logs/${lang}_orig_10M.txt"
done

date