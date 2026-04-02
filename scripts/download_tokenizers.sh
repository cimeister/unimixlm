#!/bin/bash

langs=(ibo fil lit swh zsm ron ind yor)

echo "Downloading model tokenizer..."
hf download allenai/Olmo-3-1025-7B --include="tokenizer*" \
    --local-dir=./olmo3_las_tokenizers/model_tokenizer

for lang in "${langs[@]}"; do
    echo "Downloading tokenizer for ${lang}..."
    hf download MultilingualUnigramLM/las-tokenizers-Olmo-3-1025-7B-"${lang}" \
        --local-dir=./olmo3_las_tokenizers/las-tokenizers-Olmo-3-1025-7B-"${lang}"
done
