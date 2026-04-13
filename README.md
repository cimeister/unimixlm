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

# Option A: Run setup inside an existing conda/virtualenv (recommended)
conda activate myenv          # or: source .venv/bin/activate
bash setup.sh

# Option B: Let setup.sh create a venv for you
bash setup.sh --create-venv

# Login to HuggingFace (needed for dataset access and model upload)
huggingface-cli login
```

`setup.sh` handles everything: Python dependencies, Rust install, tokenizers build (via maturin), and SentencePiece C++ build with warm-start support.

**Important**: The tokenizers build step (`maturin develop`) requires an active virtualenv or conda environment. If neither is detected, the script warns you. Use `--create-venv` to have it create one, or activate your own env first. If both `VIRTUAL_ENV` and `CONDA_PREFIX` are set, the script automatically unsets `CONDA_PREFIX` to avoid the maturin conflict.

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

Creates **one HF repo per language**: `{user}/LangMap-TheStack-{lang}-{size}`.

```bash
# Create 100M-token datasets for Tier 1 languages
python thestack_finetune_dataset.py \
    --hf-user MultilingualUnigramLM \
    --target-tokens 100000000 \
    --languages python javascript java c cpp csharp html php css markdown typescript go

# Smaller dataset for quick experiments
python thestack_finetune_dataset.py \
    --hf-user MultilingualUnigramLM \
    --target-tokens 5000000 \
    --languages rust
```

**Existing datasets on HuggingFace** (100M tokens each):
`python`, `javascript`, `java`, `c`, `go`, `typescript`

**Missing datasets** — `cpp` and `csharp` had empty uploads and need to be re-created on HPC:

```bash
# Re-create cpp and csharp datasets (run on HPC cluster)
# Both safe names (cpp, csharp) and original names (c++, c-sharp) are accepted.
python thestack_finetune_dataset.py \
    --hf-user MultilingualUnigramLM \
    --target-tokens 100000000 \
    --languages cpp csharp
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
bash scripts/run_single_code_finetune.sh rust              # langspec (default)
bash scripts/run_single_code_finetune.sh rust original     # BPE baseline
CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_code_finetune.sh python  # specific GPU

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
| Model | OLMo-3-1025-7B | OLMo-3-1025-7B | |
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
| OLMo-3-7B | ~6-7 hrs | ~50-55 hrs |
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

### Code Benchmark Suite (all 11 languages)

Run the full evaluation pipeline comparing all LAS finetuned models against base OLMo-3-7B:

```bash
bash scripts/run_code_eval.sh
```

The script **parallelises across all available GPUs** — each language is pinned to a dedicated GPU via `CUDA_VISIBLE_DEVICES`, with a FIFO-based pool ensuring at most one language per GPU at a time. On a node with 8× A100-80GB this runs all 8 code languages simultaneously, then the 3 perplexity languages.

The script is **resumable** — already-completed languages are skipped automatically (delete the sentinel file to re-run a language).

```bash
# Run all languages in parallel across all GPUs
bash scripts/run_code_eval.sh

# Skip GPU 0 if it is already running a job
GPUS="1 2 3 4 5 6 7" bash scripts/run_code_eval.sh

# Selected languages only (assigned to free GPUs)
bash scripts/run_code_eval.sh go java typescript

# Skip plotting step
SKIP_PLOTTING=1 bash scripts/run_code_eval.sh
```

#### GPU-hour estimates (8× A100-80GB, bfloat16, n_samples=1)

Observed throughput on OLMo-3-7B + PEFT:

| Run | Samples | Rate | Time per language |
|-----|---------|------|------------------|
| Base model (lm-eval) | 664 | ~6.8 s/sample | ~75 min |
| FT model (lm-eval, PEFT) | 664 | ~15 s/sample | ~170 min |
| **Total per code language** | | | **~4 h 5 min** |
| Perplexity language (BpB, base + FT) | 200 | forward-pass only | ~30 min |

| Mode | Wall-clock (8× A100) | GPU-hours |
|------|---------------------|-----------|
| 8 code languages in parallel | ~4 h 5 min | ~33 GPU-h |
| 3 perplexity languages in parallel | ~30 min | ~1.5 GPU-h |
| **Full suite (11 languages)** | **~4.5–5 h** | **~35 GPU-h** |

*FT eval is ~2.2× slower than base due to PEFT adapter overhead. Rates measured on A100-SXM4-80GB with batch_size=1 (auto-detected).*

#### Sentinel files

| Language type | Sentinel path | Delete to re-run |
|--------------|--------------|-----------------|
| Code (lm-eval) | `eval_results/code_{lang}/.eval_complete` | `rm eval_results/code_go/.eval_complete` |
| Perplexity | `eval_results/code_{lang}/.perplexity_complete` | `rm eval_results/code_html/.perplexity_complete` |

#### Benchmarks

| Language | HumanEval | MBPP | MultiPL-E Task | Metric |
|----------|:---------:|:----:|:--------------|:------:|
| Go | Python retention | Python retention | `multiple_go` | pass@1 |
| TypeScript | Python retention | Python retention | `multiple_ts` | pass@1 |
| Java | Python retention | Python retention | `multiple_java` | pass@1 |
| JavaScript | Python retention | Python retention | `multiple_js` | pass@1 |
| C++ | Python retention | Python retention | `multiple_cpp` | pass@1 |
| C# | Python retention | Python retention | `multiple_cs` | pass@1 |
| PHP | Python retention | Python retention | `multiple_php` | pass@1 |
| C | Python retention | Python retention | — | pass@1 |
| HTML | — | — | — | BpB (perplexity) |
| CSS | — | — | — | BpB (perplexity) |
| Markdown | — | — | — | BpB (perplexity) |

**Notes**:
- HumanEval and MBPP are Python-only; they are run for all languages to measure catastrophic forgetting of Python ability.
- MultiPL-E (`multiple_{lang}`) provides per-language functional correctness via code execution.
- HTML, CSS, and Markdown have no standard functional benchmark; bits-per-byte (BpB) on TheStack test splits is used. Lower BpB = better.
- All comparisons are base `allenai/OLMo-3-1025-7B` vs. `MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-{lang}`.

#### Output Structure

```
eval_results/
├── code_{lang}/
│   ├── eval_results_base_{timestamp}.json   # base model scores
│   ├── eval_results_ft_{timestamp}.json     # LAS finetuned scores
│   └── .eval_complete                       # sentinel (skip-if-done)
└── code_summary/
    ├── per_lang/
    │   └── bar_{lang}.png                   # grouped bar charts (base vs. LAS-FT)
    ├── heatmap_ft_scores.png                # tasks × languages score heatmap
    ├── heatmap_delta.png                    # improvement delta heatmap (RdBu)
    ├── summary_delta.png                    # mean improvement per language
    └── results_summary.csv                  # full numeric results table
```

#### Results

<!-- Run `bash scripts/run_code_eval.sh` to populate this table -->

| Language | Task | Base Score | LAS-FT Score | Δ |
|----------|------|:----------:|:------------:|:-:|
| *Run evaluation to populate* | | | | |

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
│   │   ├── model/                   # Model configs (olmo3)
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
- Code: `MultilingualUnigramLM/LangMap-TheStack-{lang}-{size}`

### Finetuned Models
- NL: `MultilingualUnigramLM/ft-las-olmo3-7b-fineweb100M-{lang}`
- Code: `MultilingualUnigramLM/ft-code-las-olmo3-7b-thestack100M-{lang}`
