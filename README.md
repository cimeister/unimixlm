# UniMixLM - Language-Adaptive Segmentation Tokenizers

Training pipeline for multilingual Unigram tokenizers using Language-Adaptive Segmentation (LAS).

## Tokenizer Training

```bash
# Train tokenizers for languages specified in the ./config/tokenizer_training/languages/yor.yaml file.
python run_las_tok_training.py languages=yor
```

**Training Options:**
- **Single target language**: Train a base tokenizer on one language and create one language-specific tokenizer (e.g., `languages.langs=[yor_Latn]` for Yoruba only)
- **Multiple target languages**: Train a shared base tokenizer on combined corpora from all languages, then re-estimate probabilities for each language independently (e.g. `languages.langs=[yor_Latn,fil_Latn,swh_Latn]` for specific languages)

Both approaches produce a base tokenizer with shared vocabulary and language-specific tokenizers with adapted probabilities.

## Configuration

Config file: `config/tokenizer_training/default.yaml`

### Key Options

- **languages.langs**: Language codes to train (e.g., `yor_Latn`, `fil_Latn`)
- **data.output_dir**: Output directory for trained tokenizers
- **data.lines_per_lang**: Training corpus size per language (default: 10,000)
- **training.em_mode**: EM algorithm for base tokenizer (`soft`, `hard`, `hf`)
- **training.reestimation_em_mode**: EM algorithm for language-specific re-estimation
- **execution.max_workers**: Parallel workers for training (default: 8)
- **hf.username**: HuggingFace username for pushing tokenizers
- **model.name**: Base model for initialization (default: `allenai/Olmo-3-1025-7B`)

## How It Works

### 1. Base Tokenizer Training
- Initializes vocabulary from base model tokenizer (e.g., OLMo-3 BPE)
- Trains a single shared Unigram tokenizer on combined multilingual corpora
- Uses EM algorithm to learn token probabilities across all languages

### 2. Language-Specific Re-estimation
- Freezes the shared vocabulary from base tokenizer
- Re-estimates token probabilities for each language independently
- Uses language-specific corpus and EM algorithm

### 3. Validation
- Ensures pre-tokenizer and post-tokenizer match across all tokenizers
- Verifies token order matches the base model tokenizer
- Uses assertions on JSON file contents

### 4. Output Structure
```
{output_dir}/las-tokenizers-{model}-{langs}/
├── base_tokenizer.json              # Shared vocabulary
├── langspec_eng.json                  # English-specific probabilities
├── langspec_fra.json                  # French-specific probabilities
└── ...
```

## Technical Details

- **Byte-level detection**: Automatically detects from base model's pretokenizer/decoder
- **UNK token handling**: Uses base model's `unk_token` or falls back to `eos_token`
- **Training corpus**: FineWeb2-10K dataset (streamed per language)
- **Parallelization**: Language-specific training runs in parallel via ThreadPoolExecutor

## Advanced Usage

```bash
# Change EM algorithm
python run_las_tok_training.py training.em_mode=hard training.reestimation_em_mode=soft

# Disable HuggingFace push
python run_las_tok_training.py hf.push=false
```
