#!/usr/bin/env bash
# Fine-tune OLMo-2-7B on code with LAS langspec tokenizers.
# Launches one language per GPU (8 GPUs, Tier 1 languages prioritised by data size).
#
# Runs both "original" (BPE baseline) and "langspec" (LAS code tokenizer) modes
# for direct comparison, mirroring the NL finetuning scripts.

set -euo pipefail
hostname
date

mkdir -p logs

# Tier 1 languages (>100GB in The Stack, 100M tokens trivially feasible)
# Ordered by The Stack size descending.
code_langs=(python javascript java c cpp csharp go typescript)

# ─── LAS langspec mode ───────────────────────────────────────────────────────

echo "=== Starting LAS langspec code finetuning ==="

for i in "${!code_langs[@]}"; do
    lang="${code_langs[$i]}"
    CUDA_VISIBLE_DEVICES=$i \
        python3 run_finetune.py \
            --config-name=code_default \
            "+langs={${lang}: 0.8, python: 0.2}" \
            "las_tokenizer_path=./code_las_tokenizers/las-code-tokenizers-OLMo-2-1124-7B" \
            "dataset.size=100M" \
            "las_mode=langspec" \
        &> "logs/${lang}_code_las_100M.txt" &
done

echo "All LAS langspec jobs launched, waiting for completion..."
wait
echo "All LAS langspec jobs finished."

# ─── Original tokenizer baseline ─────────────────────────────────────────────

echo "=== Starting original tokenizer code finetuning ==="

for i in "${!code_langs[@]}"; do
    lang="${code_langs[$i]}"
    CUDA_VISIBLE_DEVICES=$i \
        python3 run_finetune.py \
            --config-name=code_default \
            "+langs={${lang}: 0.8, python: 0.2}" \
            "las_mode=original" \
            "dataset.size=100M" \
        &> "logs/${lang}_code_orig_100M.txt" &
done

echo "All original tokenizer jobs launched, waiting for completion..."
wait
echo "All original tokenizer jobs finished."

# ─── Upload results ──────────────────────────────────────────────────────────

echo "=== Uploading results to HuggingFace ==="

for lang in "${code_langs[@]}"; do
    # LAS langspec
    hf repo create "MultilingualUnigramLM/ft-code-las-olmo2-7b-thestack100M-${lang}" 2>/dev/null || true
    hf upload "MultilingualUnigramLM/ft-code-las-olmo2-7b-thestack100M-${lang}" \
        "./finetuned_model/ft-code-las-langspec_${lang}-python_100M" --repo-type model

    # Original
    hf repo create "MultilingualUnigramLM/ft-code-olmo2-7b-thestack100M-${lang}" 2>/dev/null || true
    hf upload "MultilingualUnigramLM/ft-code-olmo2-7b-thestack100M-${lang}" \
        "./finetuned_model/ft-code_${lang}-python_100M" --repo-type model
done

date
echo "=== All done ==="
