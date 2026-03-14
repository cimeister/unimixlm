#!/bin/bash

langs=(ibo fil lit swh zsm ron ind yor)

for lang in "${langs[@]}"; do
    echo "Downloading tokenizer for ${lang}..."
    uv run hf download MultilingualUnigramLM/las-tokenizers-Olmo-3-1025-7B-"${lang}" \
        --local-dir=./olmo3_las_tokenizers/las-tokenizers-Olmo-3-1025-7B-"${lang}"
done
