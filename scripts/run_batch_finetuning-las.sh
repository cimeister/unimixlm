#!/usr/bin/env bash

hostname
date

mkdir -p logs

langs=(ibo fil lit swh zsm ron ind yor)

for lang in "${langs[@]}"; do
	CUDA_VISIBLE_DEVICES=0 \
        python3 run_finetune.py \
            '~langs' '+langs={'"${lang}"': 1.0, eng: 0.2}' \
            "las_tokenizer_path=/home/as3623/unimixlm/olmo3_las_tokenizers/las-tokenizers-Olmo-3-1025-7B-${lang}" \
            "dataset.size=10M" \
        &> "logs/${lang}_las_10M.txt"
done

date
