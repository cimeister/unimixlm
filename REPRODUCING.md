# Reproducing the paper's measurements

This file maps each table of the paper to the artifact in `reproduction/` that holds its numbers, to the script that produced the artifact, and to the inputs that script requires. Every artifact records the SHA-256 of the inputs that determined it (tokenizer files, gold files, harness, corpora), so a recomputed value can be checked against the recorded one.

Conventions used throughout. *Original* is the pretrained model's own tokenizer. *Base* is the shared Unigram tokenizer estimated over the same vocabulary on all languages together. *LangMAP* is the per-language tokenizer re-estimated from Base. The manifests call the fitting of Base for a model's vocabulary a *conversion*, and a LangMAP tokenizer an *overlay*, because it changes only the scores of an existing vocabulary. A *cell* in the file names below is one (model family, language) measurement. AST stands for abstract syntax tree.

## Inputs that are not in the repository

- **MorphScore gold data.** Download the per-language `*_data.csv` files of the [MorphScore dataset](https://huggingface.co/datasets/catherinearnett/morphscore) and set `MORPHSCORE_DATA_DIR` to that directory. The Arabic file is replaced by the repaired version described below.
- **Natural-language estimation corpora.** The per-language FineWeb-2 slices (`MultilingualUnigramLM/FineWeb2-<lang>-100M`) are on the Hugging Face Hub under the [MultilingualUnigramLM](https://huggingface.co/MultilingualUnigramLM) organization. The drivers download them on first use.
- **Code estimation corpora.** The code sweep reads a local directory of The Stack samples, one `<lang>/data.parquet` per programming language, named by `CODE_AST_FIT_PARQUET_ROOT`. The published slices `MultilingualUnigramLM/LangMap-TheStack-<lang>-100M` are the source of these samples.
- **The TokEval library.** The AST-leaf alignment metric is implemented in [cimeister/tokenizer-intrinsic-evals](https://github.com/cimeister/tokenizer-intrinsic-evals). The code sweep imports it from the checkout named by `TOKEVAL_REPO` and reads the per-language AST evaluation sets (parquet files) from the same checkout.
- **Model tokenizers.** Downloaded from the Hub on first use. Gemma, Llama, and Mistral tokenizers are gated and require an accepted license and `HF_TOKEN`.
- **Fitted tokenizers.** The Base conversions and LangMAP overlays measured in the paper are not committed. The drivers below refit them when missing and record the SHA-256 of every file they produce, so a refit can be compared against the hashes in the manifests.
- **FLORES+ sentences.** Required by the bits-per-byte and routing measurements. Place one file per language, one sentence per line, under `FLORES_DATA_DIR`.

## MorphScore on the fine-tuned tokenizers (Tables `tab:morphscore`, `tab:app-morphscore-full`)

Artifacts: `reproduction/morphscore/f15_morphscore_final_v2.json` (all four MorphScore metrics for Original, Base, and LangMAP on the ten cells) and `reproduction/morphscore/f15_morphscore_bootstrap_v2.json` (95% confidence intervals on the LangMAP minus Original differences, from the resampling procedure described below).

Scripts: `reproduction/morphscore/f15_final_v2.py` computes the metrics, one cell per invocation with `--cell <family>:<lang>`, then merges the parts with `--merge`. `reproduction/morphscore/f15_morphscore_bootstrap_v2.py` draws 2,000 resamples of the word types with replacement, scores both tokenizers on the same resample each time, and reports the 2.5th and 97.5th percentiles of the resulting differences as the 95% confidence interval. Both scripts read the tokenizers from `las_tokenizers/f15/<family>/`, which `scripts/prep_f15_tokenizers.py` produces.

The harness is `reproduction/harness_patched/morphscore_patched.py`. It differs from the upstream MorphScore harness in one place: predicted morpheme boundaries and per-token strings are taken from the `tokenizers` library's character offsets rather than from decoding one token id at a time. Decoding one id at a time replaces a partial multi-byte character with a single replacement character, which shifts every later boundary on non-Latin scripts. The patched file states the change and keeps everything else identical to upstream.

The Arabic gold file is repaired before use. In the released `arabic_data.csv`, the lemma column is vocalized while the word form is not, so the stem boundaries could not have been derived by locating the lemma in the word. `reproduction/morphscore/repair_arabic_gold.py` removes the vocalization marks from the lemma for matching only, locates it inside the word form, and re-derives the split. The result is `reproduction/morphscore/arabic_data_repaired.csv`. `arabic_repair_stats.json` records the number of rows in which the lemma was located inside the word form, and how often the re-derived split agrees fully, partially, or not at all with the original one. The paper reports the repaired file as the Arabic row and the summary JSON also carries the values on the original file, labelled `gold=original-broken`.

## MorphScore across ten tokenizer families (Tables `tab:app-sweep-recall`, `tab:olmo-family-grid`)

Artifact: `reproduction/intrinsic_sweep/manifest.json`. Its `cells` entry carries, for each of the 74 (family, language) cells, all MorphScore metrics for the three tokenizers and the SHA-256 of the harness, the gold file, and each tokenizer file. Its `families` entry lists the Hub model id of every family and `excluded_families` explains the one exclusion (Phi-3-mini, whose tokenizer has no pre-tokenizer and cannot be converted). `paper_table_corrections` records two corrections to `tab:tokenizers`: the vocabulary size and the label of the Gemma row.

Artifact: `reproduction/intrinsic_sweep/sweep_degeneracy_verdicts.json`. For each non-Latin cell it records whether the comparison between tokenizers is informative, and the two diagnostics behind that verdict: whether the gold boundaries fall at positions a byte-level tokenizer can produce at all (otherwise every tokenizer scores the same ceiling), and whether the three tokenizers produce identical boundaries (otherwise their difference is zero by construction).

Script: `scripts/sweep_intrinsic_v2.py`, one cell per invocation. For a family it downloads the model tokenizer, fits the Base conversion on the combined Turkish, Finnish, Hungarian, Arabic, and Hindi corpus with soft EM if it is missing, fits the LangMAP overlay for the language with 10 EM iterations on 50,000 script-filtered lines if it is missing, and scores the three tokenizers. `scripts/launch_intrinsic_sweep_v2.sh` runs the full grid and `scripts/launch_olmo_generality_grid.sh` runs the three additional families of `tab:olmo-family-grid`.

## AST-leaf alignment on code (Tables `tab:code-summary`, `tab:code-overall-grid`, `tab:code-overall-values`)

Artifact: `reproduction/code_ast/manifest.json`. Its `rows` entry holds, for each of the 72 (family, programming language) cells, the alignment of Original, LangMAP, and Base, the LangMAP minus Original difference, and the number of leaf spans evaluated.

Evaluation sets: `reproduction/code_ast/evalsets/<lang>/` holds the 100 source files per language on which alignment was measured, with a manifest recording each file's SHA-256 and the selection rule. These are committed so the measurement can be repeated on the identical inputs.

Script: `scripts/sweep_code_ast_v1.py`, one cell per invocation. It reuses the family's Base conversion from the natural-language sweep above, fits the LangMAP overlay for the programming language on the rows of the local The Stack sample whose ids are not in the AST evaluation set, parses each evaluation file with the TokEval library, and measures the fraction of leaf-node spans whose boundaries coincide with token boundaries. `scripts/launch_code_ast_v1.sh` runs the full grid.

## Convergence of the per-language distribution (Table `tab:convergence`)

Artifacts: `reproduction/convergence/convergence_v2.json` holds, for German, Finnish, and Turkish and for N in {125, 500, 1,000, 4,000, 16,000, 32,000} documents plus one point at 25 million tokens, the fertility (output pieces per character on FLORES-200 devtest) of an overlay estimated from the first N documents, and its difference from the overlay estimated on 32,000 documents. `corpora_manifest_<lang>.json` records the Hub dataset, revision, and row ranges of each estimation corpus. `fit_manifest.json` records the SHA-256 and fit time of every overlay.

The scripts that built the corpora, fitted the overlays, and measured fertility are not in the repository. The overlays were fitted with the same protocol as `scripts/prep_f15_tokenizers.py` (Base-seeded per-language EM, 10 iterations) from the Qwen2.5-0.5B Base tokenizer.

## Downstream accuracy after adaptation (Tables `tab:f15-multiblimp`, `tab:f15-joint`, `tab:routing`)

Artifacts in `reproduction/downstream/`:

- `summary_acc_batchfix.json`: per-cell MultiBLiMP accuracy means over seeds, and for each tokenizer its difference from Original. Each difference is computed within a seed and averaged over seeds, its 95% confidence interval comes from 10,000 resamples of the seed-level differences, and the significance threshold is adjusted for the number of differences tested with the Holm-Bonferroni procedure. Produced by `scripts/compile_f15_stats.py` from the per-cell evaluation records written by `scripts/eval_f14_multiblimp.py`.
- `summary_acc_vs_placebo_batchfix.json`: the same contrasts with the placebo tokenizer as reference.
- `summary_joint_acc_both.json`: per-language MultiBLiMP accuracy for the models adapted to all five languages jointly, under label-free routing and under the true label, from `scripts/eval_f15_joint.py`.
- `routing_accuracy_<family>.json`: for each of the five languages, the fraction of FLORES sentences routed to that language's distribution, with the full confusion counts and the set of per-language tokenizers the router selected among.

Training: the adapted models were trained with `run_finetune.py` through the SLURM launchers `scripts/launch_f15_wave1.sh` (one language at a time), `scripts/launch_f15_wave2.sh` (all five languages jointly), and the per-family and per-seed launchers beside them. Each launcher states the model, languages, tokenizers, seeds, and token budget it submits.

Bits per byte (Tables `tab:f15-bpb`, `tab:f15-llama-bpb`) was measured with `scripts/eval_f15_bpb.py` on the same checkpoints. The per-cell bits-per-byte records are not in `reproduction/`.

## Tables without an artifact in this repository

`tab:tokenizers`, `tab:deva-eng-subwords`, `tab:app-surface-perlang`, and `tab:app-fairness` are built from the tokenizer files directly (vocabulary inventories and per-language efficiency metrics). The scripts that produced them are not in the repository.
