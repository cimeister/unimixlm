#!/usr/bin/env bash
# Fine-tune OLMo-3-7B on code with LAS langspec tokenizers.
# Launches one language per GPU (8 GPUs, Tier 1 languages prioritised by data size).
#
# Runs both "original" (BPE baseline) and "langspec" (LAS code tokenizer) modes
# for direct comparison, mirroring the NL finetuning scripts.
#
# Language mixing: 80% target language, 20% rehearsal (Python by default,
# JavaScript for Python itself to avoid duplicate keys).

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
hostname
date

mkdir -p logs

# Tier 1 languages (>100GB in The Stack, 100M tokens trivially feasible)
# Ordered by The Stack size descending.
code_langs=(python javascript java c cpp csharp go typescript)

# Default rehearsal language is Python; for Python itself use JavaScript.
get_rehearsal() {
    if [ "$1" = "python" ]; then echo "javascript"; else echo "python"; fi
}

# Compute the output dir name that run_finetune.py generates:
#   {base}/ft-code-las-langspec_{lang}-{rehearsal}_{size}
#   {base}/ft-code_{lang}-{rehearsal}_{size}
get_las_outdir() {
    local lang="$1" rehearsal
    rehearsal=$(get_rehearsal "$lang")
    echo "./finetuned_model/ft-code-las-langspec_${lang}-${rehearsal}_100M"
}
get_orig_outdir() {
    local lang="$1" rehearsal
    rehearsal=$(get_rehearsal "$lang")
    echo "./finetuned_model/ft-code_${lang}-${rehearsal}_100M"
}

# ─── LAS langspec mode ───────────────────────────────────────────────────────

echo "=== Starting LAS langspec code finetuning ==="

for i in "${!code_langs[@]}"; do
    lang="${code_langs[$i]}"
    rehearsal=$(get_rehearsal "$lang")
    CUDA_VISIBLE_DEVICES=$i \
        python3 run_finetune.py \
            --config-name=code_default \
            "+langs={${lang}: 0.8, ${rehearsal}: 0.2}" \
            "las_tokenizer_path=./olmo3_las_tokenizers/las-code-tokenizers-OLMo-3-1025-7B" \
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
    rehearsal=$(get_rehearsal "$lang")
    CUDA_VISIBLE_DEVICES=$i \
        python3 run_finetune.py \
            --config-name=code_default \
            "+langs={${lang}: 0.8, ${rehearsal}: 0.2}" \
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
    las_dir=$(get_las_outdir "$lang")
    orig_dir=$(get_orig_outdir "$lang")

    # LAS langspec
    hf repo create "MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-${lang}" 2>/dev/null || true
    hf upload "MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-${lang}" \
        "$las_dir" --repo-type model

    # Original
    hf repo create "MultilingualUnigramLM/ft-code-olmo3-7b-thestack100M-${lang}" 2>/dev/null || true
    hf upload "MultilingualUnigramLM/ft-code-olmo3-7b-thestack100M-${lang}" \
        "$orig_dir" --repo-type model
done

date
echo "=== All done ==="
