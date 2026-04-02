#!/usr/bin/env bash

hostname
date

mkdir -p logs

langs=(ibo fil lit swh zsm ron ind yor)

# Launch all 8 languages in parallel, one per GPU
for i in "${!langs[@]}"; do
    lang="${langs[$i]}"
    CUDA_VISIBLE_DEVICES=$i \
        python3 run_finetune.py \
            '+langs={'"${lang}"': 0.8, eng: 0.2}' \
            "las_tokenizer_path=./olmo3_las_tokenizers/las-tokenizers-Olmo-3-1025-7B-${lang}" \
            "dataset.size=100M" \
            "las_mode=langspec" \
        &> "logs/${lang}_las_100M.txt" &
done

echo "All training jobs launched, waiting for completion..."
wait
echo "All training jobs finished."

# Upload results
for lang in "${langs[@]}"; do
    hf repo create MultilingualUnigramLM/ft-las-olmo3-7b-fineweb100M-${lang}
    hf upload MultilingualUnigramLM/ft-las-olmo3-7b-fineweb100M-${lang} ./finetuned_model/ft-las-langspec_${lang}-eng_100M --repo-type model
done

date
