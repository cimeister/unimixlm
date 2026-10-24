# LangMAP: Language-Adaptive Maximum a Posteriori Tokenization

Code for the paper *LangMAP: A Language-Adaptive Approach to Tokenization* (EMNLP 2026).

LangMAP extends the UnigramLM tokenization algorithm to the multilingual setting. A standard UnigramLM tokenizer has one probability distribution over its vocabulary. LangMAP keeps one shared vocabulary and estimates one distribution per language over it. The per-language distributions are estimated with expectation maximization (EM) from language-labelled text. At inference time no language label is required: the tokenizer segments the input under every language's distribution and keeps the segmentation with the highest probability. This selection is called routing below.

The repository supports two uses of LangMAP:

- **Adapting a pretrained model's tokenizer.** The model's vocabulary is kept. A shared Unigram distribution is first estimated over that vocabulary, and one distribution per target language is then re-estimated from it. Token ids are unchanged, so the model's embedding matrix is reused as is.
- **Training a tokenizer from scratch.** A new vocabulary is learned, a shared Unigram distribution is estimated over it with the forked SentencePiece trainer's fixed-vocabulary mode, and per-language distributions are then estimated over the same vocabulary.

The repository also contains the low-rank adapter (LoRA) fine-tuning and evaluation pipelines used for the paper's downstream experiments, and the artifacts behind the paper's tables, described in [REPRODUCING.md](REPRODUCING.md).

## Repository layout

| Path | Contents |
|---|---|
| `code/` | Tokenizer implementation: EM trainer, shared and per-language Unigram tokenizers, routing. |
| `las_experiment/tokenizer_training/` | End-to-end tokenizer fitting: corpus preparation, shared base tokenizer, per-language re-estimation, validation. |
| `las_experiment/finetune.py` | LoRA fine-tuning of a pretrained model under a chosen tokenizer. |
| `las_experiment/downstream_eval.py`, `las_experiment/bpb_core.py` | Downstream accuracy evaluation through `lm-evaluation-harness`, and bits-per-byte evaluation. |
| `las_experiment/code_benchmark_eval.py` | Code generation benchmarks through `bigcode-evaluation-harness`. |
| `run_las_tok_training.py`, `run_finetune.py`, `run_eval.py` | Hydra entry points. Configuration lives under `config/`. |
| `scripts/` | Drivers and launchers for the measurements reported in the paper. |
| `reproduction/` | Artifacts behind the paper's tables, the patched MorphScore harness, and the fixed code evaluation sets. |
| `tests/` | Pytest suite for tokenizer consistency across training, fine-tuning, and evaluation. |
| `tokenizers/`, `sentencepiece/` | Git submodules holding the two forked libraries described below. |
| `example_tok/`, `unimix_fast.ipynb` | A small example tokenizer bundle and a notebook demonstrating routing with the per-language score vectors cached in the Rust layer. |
| `tokenizer_dashboard.py` | Gradio dashboard comparing segmentations across the original, shared, and per-language tokenizers. |

## Installation

Prerequisites: Python 3.10 or newer, a Rust toolchain (installed by the setup script if absent), CMake, and a C++ compiler.

```bash
git clone --recurse-submodules <repository-url>
cd unimixlm
python -m venv .venv && source .venv/bin/activate
bash setup.sh
```

`setup.sh` installs the pinned Python dependencies from `requirements.txt`, builds the `tokenizers` Python bindings from the submodule, and builds the SentencePiece command-line trainer from the submodule.

Two forked libraries are required. The stock releases do not work as substitutes.

- **`tokenizers` fork** ([Ahmetcanyvz/tokenizers](https://github.com/Ahmetcanyvz/tokenizers), branch `unimixlm_fast`). Holding several per-language score vectors for one vocabulary, and selecting the highest-probability segmentation among them, are implemented in the Rust layer of this fork. The stock `tokenizers` package has neither.
- **SentencePiece fork** ([cimeister/sentencepiece](https://github.com/cimeister/sentencepiece), branch `fixed-vocab-em`). Its `spm_train` accepts `--seed_sentencepieces_file`, which the from-scratch pipeline uses to warm-start EM from a fixed vocabulary.

Run the tests from the repository root:

```bash
pytest tests
```

The package directory is named `code`, which shadows the standard library module of the same name. Run the entry points from the repository root so that the local package is the one imported.

## Fitting LangMAP tokenizers for a pretrained model

```bash
python run_las_tok_training.py languages=yor model=olmo3-1025-7b \
    data.output_dir=./las_tokenizers/olmo3-7b_yor hf.push=false
```

`languages` names a file under `config/tokenizer_training/languages/` listing language codes in the FineWeb-2 convention (`yor_Latn`). `model` names a file under `config/tokenizer_training/model/` giving the Hugging Face model whose tokenizer is adapted. By default the per-language corpora are streamed from the `MultilingualUnigramLM/FineWeb2-10K` dataset on the Hugging Face Hub. Set `data.local_dir` to read parquet files from `{local_dir}/{lang}/` instead. `data.lines_per_lang` sets the number of lines used per language.

The run writes, into `data.output_dir`:

| File | Contents |
|---|---|
| `model_tokenizer/` | A copy of the pretrained model's tokenizer, the source of the vocabulary. |
| `base_tokenizer.json` | The shared Unigram tokenizer over the model's vocabulary, estimated on all languages together. |
| `langspec_<lang>.json` | One Unigram tokenizer per language, with the same vocabulary and ids and that language's re-estimated scores. |
| `pathpiece_tokenizer.json` | A tokenizer over the same vocabulary that picks the segmentation with the fewest pieces, used as a baseline. |
| `corpora/` | The text used for estimation. Set `LAS_TOK_CORPORA_ROOT` to write this directory elsewhere. |

Validation runs at the end of the fit and raises on any mismatch between the model tokenizer, the shared tokenizer, and the per-language tokenizers in pre-tokenizer, normalizer, decoder, or token order. `scripts/validate_tokenizer_dir.py` runs the same checks plus a routing check on a saved directory.

### Using a fitted tokenizer

```python
from code.language_specific_unigramlm import LanguageSpecificUnigramLMTokenizer

tok = LanguageSpecificUnigramLMTokenizer()
tok.load(
    "las_tokenizers/olmo3-7b_yor/base_tokenizer.json",
    language_paths={
        "yor_Latn": "las_tokenizers/olmo3-7b_yor/langspec_yor_Latn.json",
        "fil_Latn": "las_tokenizers/olmo3-7b_yor/langspec_fil_Latn.json",
    },
)

# Segment under one language's distribution.
encoding = tok.encode_lang("Báwo ni o ṣe wà?", "yor_Latn")

# Segment without a language label: the language whose distribution gives the
# highest probability is selected.
language, pieces = tok.best_language_encode("Báwo ni o ṣe wà?")
```

Each `langspec_<lang>.json` file is also a complete tokenizer in the Hugging Face `tokenizers` format and can be loaded with `Tokenizer.from_file` on its own. `unimix_fast.ipynb` shows the faster routing path, which loads all per-language score vectors into the Rust layer once and selects among them in a single call.

## Fine-tuning a model under a tokenizer

```bash
python run_finetune.py model=qwen2_5-0_5b langs=[fin] \
    las_mode=langspec las_tokenizer_path=./las_tokenizers/qwen2_5-0_5b_fin \
    dataset.size=100M dataset.local_dir=null \
    output_dir=./checkpoints/qwen2_5-0_5b_fin_langspec resume_policy=never
```

`las_mode` selects the tokenizer the model is trained under:

| `las_mode` | Tokenizer |
|---|---|
| `original` | The model's own tokenizer. |
| `base` | `base_tokenizer.json`, the shared Unigram tokenizer. |
| `langspec` | `langspec_<lang>.json` for each training language. |
| `pathpiece` | `pathpiece_tokenizer.json`. |
| `placebo` | `placebo_<lang>.json`, a copy of the per-language tokenizer whose scores are shuffled among pieces of similar score, produced by `scripts/make_placebo.py`. |
| `vocab_extend` | The model's own tokenizer with monolingual pieces appended and new embeddings initialised, produced by `scripts/build_f15_vocabext.py`. |

`langs` takes a list (`langs=[fin,eng]`) or a mapping of proportions (`langs="{fin:1.0,eng:0.2}"`). `dataset.size` of `5M`, `10M`, `100M`, or `500M` reads a dataset of that size from the Hugging Face Hub. `dataset.size=local` reads all of `dataset.local_dir`, and `dataset.size=target:<N>` reads the first N tokens of it. `resume_policy` is required on every run: `never` aborts if a checkpoint already exists in `output_dir`, `auto` resumes from the latest one, and `require` aborts if none exists. The remaining options are documented in `config/finetune/default.yaml`. The code variant of the pipeline, which reads The Stack, is configured by `config/finetune/code_default.yaml`.

The datasets used in the paper were built with `fineweb2_finetune_dataset.py` (FineWeb-2) and `thestack_finetune_dataset.py` (The Stack) and are published under the [MultilingualUnigramLM](https://huggingface.co/MultilingualUnigramLM) organization on the Hugging Face Hub, alongside the fitted tokenizers (`las-tokenizers-*`, `las-code-tokenizers-*`) and fine-tuned adapters (`ft-*`).

## Evaluation

`run_eval.py` evaluates a fine-tuned checkpoint on Belebele, PIQA, MultiBLiMP, and TurBLiMP through `lm-evaluation-harness`, with the checkpoint's own tokenizer substituted for the model's. The checkpoint paths have no default and are passed on the command line. The options are listed in `config/eval/default.yaml`.

Bits-per-byte scripts under `scripts/`:

- `eval_flores_bpb.py` scores a base model or checkpoint on FLORES+ sentences. It reads `{FLORES_DATA_DIR}/{lang}.txt`, one sentence per line.
- `eval_heldout_bpb.py` scores a checkpoint on the held-out documents recorded in its `heldout_manifest.json`.
- `eval_zeroshot_bpb.py` compares tokenizers on a pretrained model without fine-tuning.

The intrinsic measurements of the paper (MorphScore, alignment with abstract syntax tree (AST) leaves on code, fertility convergence, routing accuracy) are described in [REPRODUCING.md](REPRODUCING.md).

## Environment variables

Every machine-specific location is read from an environment variable or a command-line flag. A script that requires one and does not find it aborts and names the variable.

| Variable | Used by | Meaning |
|---|---|---|
| `HF_TOKEN` | all Hub access | Token for gated models and for pushing artifacts. |
| `WANDB_ENTITY`, `WANDB_PROJECT` | `run_finetune.py` | Weights & Biases destination. Logging is on by default; set `logger.enabled=false` to turn it off. |
| `LAS_TOK_CORPORA_ROOT` | tokenizer fitting | Optional directory for estimation corpora, in place of `{output_dir}/corpora`. |
| `MORPHSCORE_DATA_DIR` | MorphScore scripts | Directory holding the per-language `*_data.csv` files from the [MorphScore dataset](https://huggingface.co/datasets/catherinearnett/morphscore). |
| `MORPHSCORE_PARTS_DIR`, `BOOTSTRAP_PARTS_DIR` | `reproduction/morphscore/` | Scratch directories for per-cell part files. |
| `FLORES_DATA_DIR` | bits-per-byte scripts | Directory of FLORES+ sentence files. |
| `SCRATCH_DIR`, `ENGCODE_CACHE_ROOT` | from-scratch pipeline | Scratch roots for checkpoints and tokenized caches. |
| `CODE_AST_FIT_PARQUET_ROOT` | `scripts/sweep_code_ast_v1.py` | Directory of The Stack samples, one `<lang>/data.parquet` per language. |
| `TOKEVAL_REPO` | `scripts/sweep_code_ast_v1.py` | Checkout of `tokenizer-intrinsic-evals`, which holds the AST alignment metric. |
| `LAS_TOKENIZER_DIR` | `tokenizer_dashboard.py` | Tokenizer directory to display. |
| `BIGCODE_EVAL_MAIN` | `code_benchmark_eval.py` | Path to the `bigcode-evaluation-harness` entry point. |
| `SLURM_ACCOUNT` | `scripts/launch_*.sh` | Account for SLURM submissions. |

## Citation

```bibtex
@inproceedings{meister2026langmap,
  title     = {{LangMAP}: A Language-Adaptive Approach to Tokenization},
  author    = {Meister, Clara and Salhan, Suchir and Szablewski, Andrzej and Lesci, Pietro and Buttery, Paula and Pimentel, Tiago},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

## License

MIT License. The full text is in [LICENSE](LICENSE).
