#!/usr/bin/env bash

hostname
date

mkdir -p logs

langs=(ibo fil lit swh zsm ron ind yor)

for lang in "${langs[@]}"; do
	CUDA_VISIBLE_DEVICES=0 \
        python3 run_finetune.py \
            '+langs={'"${lang}"': 0.8, eng: 0.2}' \
            "las_mode=original" \
            "dataset.size=100M" \
        &> "logs/${lang}_orig_100M.txt"

    hf repo create MultilingualUnigramLM/ft-olmo3-7b-fineweb100M-${lang}
    hf upload MultilingualUnigramLM/ft-olmo3-7b-fineweb100M-${lang} ./finetuned_model/ft_${lang}-eng_100M --repo-type model

done

date
