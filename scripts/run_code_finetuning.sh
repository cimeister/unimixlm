#!/usr/bin/env bash
set -euo pipefail

echo "=== Running single code finetunes for all languages ==="

langs=(
  #python
  javascript
  java
  c
  cpp
  csharp
  html
  php
  css
  markdown
  typescript
  go
)

for lang in "${langs[@]}"; do
    echo "--------------------------------------------"
    echo "Starting finetune for: $lang"
    echo "--------------------------------------------"

    bash scripts/run_single_code_finetune.sh "$lang"

    echo "Finished: $lang"
done

echo "=== All single-language finetunes completed ==="