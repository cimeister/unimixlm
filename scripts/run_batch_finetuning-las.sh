#!/usr/bin/env bash

hostname
date

mkdir -p logs

langs=(ibo fil lit swh zsm ron ind yor)

for lang in "${langs[@]}"; do
	CUDA_VISIBLE_DEVICES=0 \
        python3 run_finetune.py \
            '+langs={'"${lang}"': 0.8, eng: 0.2}' \
            "las_tokenizer_path=/home/as3623/unimixlm/olmo3_las_tokenizers/las-tokenizers-Olmo-3-1025-7B-${lang}" \
            "dataset.size=100M" \
            "las_mode=langspec" \
        &> "logs/${lang}_las_100M.txt"

    hf repo create MultilingualUnigramLM/ft-las-olmo3-7b-fineweb100M-${lang}
    hf upload MultilingualUnigramLM/ft-las-olmo3-7b-fineweb100M-${lang} /home/as3623/unimixlm/finetuned_model/ft-las-langspec_${lang}-eng_100M --repo-type model
done

date
