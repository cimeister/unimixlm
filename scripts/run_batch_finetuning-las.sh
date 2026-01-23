#!/usr/bin/env bash

hostname
date

mkdir -p logs

langs=(ibo fil lit swh zsm ron ind)  # yor

for lang in "${langs[@]}"; do
	CUDA_VISIBLE_DEVICES=0 \
        python3 run_finetune.py \
            "langs=[${lang}]" \
            "las_tokenizer_path=/home/as3623/unimixlm/olmo3_las_tokenizers_new/${lang}_Latn" \
            "dataset.size=10M" \
        &> "logs/${lang}_las_10M.txt"
done

date