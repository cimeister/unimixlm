#!/usr/bin/env bash

hostname
date

mkdir -p logs

langs=(ibo fil lit swh zsm ron ind yor)

# Launch all 8 evals in parallel, one per GPU
for i in "${!langs[@]}"; do
    lang="${langs[$i]}"
    CUDA_VISIBLE_DEVICES=$i \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONUNBUFFERED=1 \
        python3 run_eval.py \
            "model_lang=${lang}" \
        &> "logs/eval_full_${lang}.txt" &
done

echo "All eval jobs launched, waiting for completion..."
wait
echo "All eval jobs finished."

date
