# UniMixLM - Language-Adaptive Segmentation Tokenizers

Training pipeline for Language-Adaptive Segmentation (LAS) tokenizers and finetuning LLMs with language-specific tokenizers for both natural languages and programming languages.

## Setup

```bash
# Clone with submodules
git clone https://github.com/cimeister/unimixlm.git
cd unimixlm
git switch suchir-code-tok

# OR if already cloned:
git submodule update --init --recursive

# Run setup (installs deps, builds tokenizers + sentencepiece)
bash setup.sh

# Login to HuggingFace (needed for dataset access and model upload)
huggingface-cli login
```

`setup.sh` handles everything: Python dependencies, Rust tokenizers build, and SentencePiece C++ build with warm-start support. See comments in the script for details.

## Overview

The pipeline has four stages:

1. **Dataset Creation** - Stream and curate training data from HuggingFace
2. **Tokenizer Training** - Train LAS tokenizers (shared base + language-specific)
3. **Finetuning** - LoRA fine-tune LLMs with original or LAS tokenizers
4. **Evaluation** - Benchmark with lm-eval-harness (BpB + downstream tasks)

Each stage supports both **natural language** and **code** domains.

---

## 1. Dataset Creation

### Natural Language (FineWeb2)

```bash
# Create a 100M-token-per-language dataset from FineWeb2
python fineweb2_finetune_dataset.py \
    --hf-repo MultilingualUnigramLM/FineWeb2-100M \
    --target-tokens 100000000

# Verify
python fineweb2_finetune_dataset.py \
    --hf-repo MultilingualUnigramLM/FineWeb2-100M --verify fil_Latn
```

Existing datasets on HuggingFace:
- `MultilingualUnigramLM/FineWeb2-5M` (5M tokens/lang)
- `MultilingualUnigramLM/FineWeb2-10M` (10M tokens/lang)
- `MultilingualUnigramLM/FineWeb2-100M-olmo3-7b-toks` (100M tokens/lang)

### Code (The Stack)

Creates **one HF repo per language**: `{user}/LangMap-TheStack-{lang}-{size}-olmo2`.

```bash
# Create 100M-token datasets for Tier 1 languages
python thestack_finetune_dataset.py \
    --hf-user MultilingualUnigramLM \
    --target-tokens 100000000 \
    --languages python javascript java c cpp csharp go typescript

# Smaller dataset for quick experiments
python thestack_finetune_dataset.py \
    --hf-user MultilingualUnigramLM \
    --target-tokens 5000000 \
    --languages rust
```

**Language data availability** (The Stack sizes, for 100M token feasibility):

| Tier | Languages | The Stack Size |
|------|-----------|---------------|
| Tier 1 (>100GB) | python, javascript, java, c, cpp, csharp, html, php, css, markdown, typescript, go | Trivially feasible |
| Tier 2 (5-100GB) | rust, ruby, sql, scala, shell, haskell, lua, perl, makefile, tex | Feasible |
| Tier 3 (<5GB) | fortran, julia, powershell, visual_basic, assembly, cmake, dockerfile, batchfile | May not reach 100M |

---

## 2. Tokenizer Training

### Natural Language

```bash
# Train tokenizers for one language
python run_las_tok_training.py languages.langs=[yor_Latn]

# Train for multiple languages (shared base, per-language re-estimation)
python run_las_tok_training.py languages.langs=[yor_Latn,fil_Latn,swh_Latn]

# Push to HuggingFace
python run_las_tok_training.py languages=yor hf.push=true
```

Config: `config/tokenizer_training/default.yaml`

### Code

```bash
# Train code tokenizers for all 30 programming languages
python run_las_code_tok.py

# Train for specific languages only
python run_las_code_tok.py languages.langs=[python,rust,javascript]
```

Config: `config/tokenizer_training/default.yaml`

Published code tokenizers: [`MultilingualUnigramLM/las-code-tokenizers-OLMo-2-1124-7B-ass_bat_c_plus27`](https://huggingface.co/MultilingualUnigramLM/las-code-tokenizers-OLMo-2-1124-7B-ass_bat_c_plus27)

### How It Works

1. **Base Tokenizer Training**: Initialises vocabulary from the base model tokenizer (e.g., OLMo BPE), trains a shared Unigram tokenizer on combined corpora using EM
2. **Language-Specific Re-estimation**: Freezes the shared vocabulary, re-estimates token probabilities per language using EM on language-specific corpora
3. **Validation**: Verifies pre/post-tokenizer consistency and token order across all tokenizers

### Output Structure

```
# Natural language
las-tokenizers-{model}-{langs}/
  base_tokenizer.json
  langspec_yor_Latn.json
  langspec_fil_Latn.json

# Code (note: no _Latn suffix)
las-code-tokenizers-{model}-{langs}/
  base_tokenizer.json
  langspec_python.json
  langspec_rust.json
```

---

## 3. Finetuning

### Prerequisites

Download tokenizers before finetuning:

```bash
# For code finetuning
bash scripts/download_code_tokenizers.sh

# For NL finetuning (manual download)
# Place model tokenizer at ./olmo3_las_tokenizers/model_tokenizer
# Place LAS tokenizers at ./olmo3_las_tokenizers/las-tokenizers-Olmo-3-1025-7B-{lang}
```

### Natural Language Finetuning

```bash
# Single language with LAS tokenizer (80% target, 20% English)
python run_finetune.py \
    '+langs={yor: 0.8, eng: 0.2}' \
    las_tokenizer_path=./olmo3_las_tokenizers/las-tokenizers-Olmo-3-1025-7B-yor \
    las_mode=langspec \
    dataset.size=100M

# Original tokenizer baseline
python run_finetune.py \
    '+langs={yor: 0.8, eng: 0.2}' \
    las_mode=original \
    dataset.size=100M

# Batch: all languages on 8 GPUs (SLURM)
sbatch scripts/run_batch_finetuning-las-slurm.sh
```

Config: `config/finetune/default.yaml`

### Code Finetuning

```bash
# Single language with LAS code tokenizer (80% target, 20% Python rehearsal)
python run_finetune.py \
    --config-name=code_default \
    '+langs={rust: 0.8, python: 0.2}' \
    las_mode=langspec \
    dataset.size=100M

# Original tokenizer baseline
python run_finetune.py \
    --config-name=code_default \
    '+langs={rust: 0.8, python: 0.2}' \
    las_mode=original

# Multilingual code (language cluster)
python run_finetune.py \
    --config-name=code_default \
    '+langs=[java, scala]' \
    las_mode=langspec

# With unfrozen embeddings (experimental, best BpB results)
python run_finetune.py \
    --config-name=code_default \
    '+langs={rust: 0.8, python: 0.2}' \
    lora=r64-k128 \
    unfreeze_embeddings=true \
    epochs=5

# Batch: Tier 1 languages on 8 GPUs
bash scripts/run_code_finetuning.sh            # local multi-GPU
sbatch scripts/run_code_finetuning-slurm.sh    # SLURM
```

Config: `config/finetune/code_default.yaml`

### Finetuning Configuration

| Parameter | NL Default | Code Default | Notes |
|-----------|-----------|-------------|-------|
| Model | OLMo-3-1025-7B | OLMo-2-1124-7B | |
| LoRA | r=16, alpha=32 | r=16, alpha=32 | Override: `lora=r64-k128` |
| Batch size | 48 (6x8) | 48 (6x8) | per_device x grad_accum |
| Learning rate | 2e-4 | 2e-4 | |
| Epochs | 1 | 1 | 5 for unfrozen embeddings |
| Dataset | 100M tokens | 100M tokens | Per language |
| Embeddings | Frozen | Frozen | `unfreeze_embeddings=true` to unfreeze |

### Embedding Unfreezing

Based on experiments, unfreezing `embed_tokens` and `lm_head` alongside LoRA adapters improves BpB for language-specific tokenizers. Use with `lora=r64-k128` and `epochs=5`:

```bash
python run_finetune.py \
    --config-name=code_default \
    '+langs={python: 0.8, javascript: 0.2}' \
    lora=r64-k128 \
    unfreeze_embeddings=true \
    epochs=5
```

### Language Mixing

**Language-specific** (recommended starting point): 80% target language, 20% rehearsal. For NL the rehearsal is English; for code the default is Python (configurable).

```yaml
langs:
  rust: 0.8
  python: 0.2    # rehearsal language
```

Modern practitioners sometimes use a 90/10 split for code (90% target, 10% rehearsal). Adjust proportions as needed.

**Multilingual** (language clusters): Fine-tune on related languages simultaneously. Pre-defined clusters in `config/finetune/code_langs/`:

| Cluster | Config | Languages | Rationale |
|---------|--------|-----------|-----------|
| Systems | `code_langs/systems` | c, cpp, rust | Similar AST, memory management |
| JVM | `code_langs/jvm` | java, scala | JVM ecosystem, shared libraries |
| Web | `code_langs/web` | javascript, typescript, html, css | Web stack, syntax similarity |
| Scripting | `code_langs/scripting` | python, ruby, perl, lua, shell | Dynamic typing |

Additional suggested clusters:
- **Functional**: haskell, scala, julia
- **ML/Scientific**: python, julia, fortran, rust
- **C-family**: c, cpp, csharp, java, javascript, typescript

Choosing a curated subset (e.g., JVM languages for a Scala model) is the recommended approach for cross-language transfer.

### GPU Hour Estimates (1x A100 80GB, bf16)

Based on ~4,500 tok/s throughput for LoRA r=16, ~2,750 tok/s for LoRA r=64 with unfrozen embeddings. Training steps per epoch: 100M / (48 * 1024) = 2,034 steps.

| Scenario | Tokens | Throughput | Time per Language |
|----------|--------|------------|-------------------|
| Frozen emb, LoRA r=16, 1 epoch | 100M | ~4,500 tok/s | **~6-7 hours** |
| Unfrozen emb, LoRA r=64, 5 epochs | 500M | ~2,750 tok/s | **~50-55 hours** |
| Dataset creation (streaming) | 100M tok | I/O bound | **~0.5-1 hour** |

**Per-model estimates** (all 7B-class, similar architectures):

| Model | LoRA r=16, 1 epoch | LoRA r=64, 5 epochs (unfrozen) |
|-------|-------------------|-------------------------------|
| OLMo-2-7B | ~6-7 hrs | ~50-55 hrs |
| CodeLlama-7B | ~6-7 hrs | ~50-55 hrs |
| StarCoder2-7B (GQA) | ~5.5-6.5 hrs | ~45-50 hrs |
| DeepSeek-Coder-6.7B | ~5.5-6.5 hrs | ~45-50 hrs |
| Qwen2.5-Coder-7B (GQA) | ~5.5-6.5 hrs | ~45-50 hrs |

**Total cost for 8 languages x 2 modes** (langspec + original baseline):
- Frozen embeddings: ~112 GPU-hours (~14 wall-clock hours on 8 GPUs)
- Unfrozen embeddings: ~880 GPU-hours (~110 wall-clock hours on 8 GPUs)

*GQA (grouped-query attention) models are ~5-10% faster. Unfrozen embeddings add ~1B trainable parameters (embed_tokens + lm_head), which is the dominant cost driver.*

---

## 4. Evaluation

### Natural Language

```bash
python run_eval.py model_lang=yor
```

Tasks:
- **Global MMLU** - multilingual general knowledge MCQ (random guess: 25%)
- **Belebele** - reading comprehension (random guess: 25%)
- **PIQA** - question answering (random guess: 50%)

### Code

```bash
python run_eval.py --config-name=code_default model_lang=python
```

Tasks:
- **HumanEval** - code generation benchmark
- **MBPP** - Mostly Basic Python Problems
- **MultiPL-E** - multilingual code generation across programming languages

### BpB Metric

All finetuning runs report bits-per-byte (BpB) during training and evaluation. BpB is tokenizer-agnostic, enabling fair comparison between original and LAS tokenizers:

```
BpB = total_NLL / (total_bytes * ln(2))
```

---

## Project Structure

```
unimixlm/
├── run_finetune.py                  # Finetuning entry point
├── run_las_tok_training.py          # NL tokenizer training entry point
├── run_las_code_tok.py              # Code tokenizer training entry point
├── run_eval.py                      # Evaluation entry point
├── fineweb2_finetune_dataset.py     # NL dataset creation
├── thestack_finetune_dataset.py     # Code dataset creation
├── setup.sh                         # Environment setup
├── requirements.txt                 # Python dependencies
│
├── config/
│   ├── finetune/
│   │   ├── default.yaml             # NL finetuning config
│   │   ├── code_default.yaml        # Code finetuning config
│   │   ├── model/                   # Model configs (olmo3, olmo2)
│   │   ├── lora/                    # LoRA configs (r16-k32, r32-k64, r64-k128)
│   │   ├── code_langs/              # Language cluster presets
│   │   └── logger/                  # WandB configs
│   ├── eval/
│   │   ├── default.yaml             # NL evaluation config
│   │   └── code_default.yaml        # Code evaluation config
│   └── tokenizer_training/
│       └── default.yaml             # Tokenizer training config
│
├── las_experiment/
│   ├── finetune.py                  # Core finetuning logic
│   ├── downstream_eval.py           # Evaluation with LAS tokenizers
│   └── tokenizer_training/
│       ├── training.py              # NL tokenizer training
│       └── code_tokenizer_training.py  # Code tokenizer training
│
├── code/                            # Core tokenizer implementations
│   ├── standard_unigramlm.py
│   ├── language_specific_unigramlm.py
│   ├── em_unigram_trainer.py
│   └── ...
│
├── scripts/
│   ├── run_code_finetuning.sh       # Code finetuning (multi-GPU)
│   ├── run_code_finetuning-slurm.sh # Code finetuning (SLURM)
│   ├── download_code_tokenizers.sh  # Download tokenizers from HF
│   ├── run_batch_finetuning-las.sh  # NL LAS finetuning (multi-GPU)
│   └── ...
│
├── sentencepiece/                   # SentencePiece submodule
└── tokenizers/                      # HuggingFace tokenizers submodule
```

## HuggingFace Resources

### Tokenizers
- NL: `MultilingualUnigramLM/las-tokenizers-Olmo-3-1025-7B-{lang}`
- Code: `MultilingualUnigramLM/las-code-tokenizers-OLMo-2-1124-7B-ass_bat_c_plus27`

### Datasets
- NL: `MultilingualUnigramLM/FineWeb2-{5M,10M,100M}`
- Code: `MultilingualUnigramLM/LangMap-TheStack-{lang}-{size}-olmo2`

### Finetuned Models
- NL: `MultilingualUnigramLM/ft-las-olmo3-7b-fineweb100M-{lang}`
- Code: `MultilingualUnigramLM/ft-code-las-olmo2-7b-thestack100M-{lang}`
