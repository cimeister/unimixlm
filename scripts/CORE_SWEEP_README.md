# Core sweep — what's prepared, prerequisites, how to run

Branch `sct-core-sweep`, worktree `/local/scratch/sas245/unimixlm-sct`,
forked from `origin/suchir-code-tok`. `log_space` is untouched.

## What this sweep is
The cheapest configuration where the intrinsic results say LangMAP is
strongest (so a finetuned win is most probable per the paper's own
headroom thesis):

| Cell | Why | Arms |
|---|---|---|
| gemma2-2b × fin | max-headroom cell (Δbpb≈0.296) | original, base, langspec, placebo (+2nd seed on base/langspec) |
| gemma2-2b × hun | 2nd-strongest language | base, langspec |
| phi3-mini × fin | paradigm-different robustness | base, langspec |

Regime: LoRA r16, **frozen embeddings** (cheapest), seq 1024, 100M tok,
1 epoch. ≈ ~3 continuous A100-days (≈ ~7 nights overnight-only) at 100M.

## Code changes made (this branch only)
- `las_experiment/finetune.py`:
  - precision auto-detect (bf16 on A100, fp16 on local Pascal/Maxwell,
    fp32 CPU) — same code on every tier, no per-host edits;
  - `gradient_checkpointing` + `eval_steps` now config-driven;
  - **new `placebo` las_mode** (loads `placebo_{lang}[_Latn].json`,
    mirrors langspec) — the causal control;
  - `gemma2` + `phi3` added to LoRA target inference (would have
    crashed otherwise).
- `config/finetune/model/{gemma2-2b,phi3-mini}.yaml` — new.
- `scripts/make_placebo.py` — scrambles langspec scores, keeps vocab/IDs
  identical (embedding stays aligned).
- `scripts/run_core_sweep.sh` — resumable runner + run-status manifest;
  prints `matrix_status` so a partial matrix can't be silently consumed.

All Python compiles; bash syntax-checked. **Nothing has been executed**
(no GPU/data/network here) — this is scaffolding only.

## Prerequisites — must clear BEFORE spending A100 time

1. **Environment.** A venv with torch, transformers, peft, hydra-core,
   tokenizers, datasets, accelerate (and lm-eval only for the later
   downstream step). Confirm `python3 -c "import torch,transformers,peft,hydra,tokenizers"`.

2. **HF access (untested from this host).**
   - `google/gemma-2-2b` is **gated** — needs an accepted license + a
     HF token (`huggingface-cli login`).
   - `microsoft/Phi-3-mini-4k-instruct` (no separate public base — see
     note in its model config; paired Δ cancels instruct-vs-base).
   - FineWeb2 repos `MultilingualUnigramLM/FineWeb2-100M-*` reachable.

3. **KILL-SWITCH: overlay tokenizers + the unpinned NL corpus.**
   `config/tokenizer_training/default.yaml` is code-shaped while
   `training.py` is NL — the NL overlay-EM corpus is **not pinned**.
   Audit `las_experiment/tokenizer_training/training.py`, then build:
   ```
   python run_las_tok_training.py languages=emnlp_fin_hun \
       model.model_name=google/gemma-2-2b hf.push=false \
       data.output_dir=./las_gemma2_2b
   python run_las_tok_training.py languages=emnlp_fin_hun \
       model.model_name=microsoft/Phi-3-mini-4k-instruct hf.push=false \
       data.output_dir=./las_phi3_mini
   ```
   (Create `config/tokenizer_training/languages/emnlp_fin_hun.yaml`
   with `langs: [fin_Latn, hun_Latn]`.) Must yield, per dir:
   `base_tokenizer.json`, `langspec_fin_Latn.json`,
   `langspec_hun_Latn.json`, and a `model_tokenizer/` dir.
   **Pin the resolved corpus + revision SHA in a committed manifest.**

4. **Placebo files** (after overlays exist):
   ```
   python scripts/make_placebo.py --langspec las_gemma2_2b/langspec_fin_Latn.json \
       --out las_gemma2_2b/placebo_fin_Latn.json --seed 1234
   ```

5. **Tokenizer/embedding sanity gate** (after overlays, before sweep):
   for each (model, overlay) assert every overlay token id maps to the
   same piece as the model tokenizer's row, |V| matches the embedding,
   UNK rate ≈ 0 on a FLORES sample. A failure here = the effect would
   be a tokenization artifact. Do not skip.

## Run
```
cd /local/scratch/sas245/unimixlm-sct
SIZE=100M bash scripts/run_core_sweep.sh
```
Resumable (skips `completed` cells). Tune via env vars: `SIZE`, `SEQ`,
`LORA`, `UNFREEZE`, `GRAD_CKPT`, `LAS_GEMMA`, `LAS_PHI`, `OUT`, `RUNS`.
If A100-80G OOMs (unlikely at 2–4B frozen): `GRAD_CKPT=true`.

## Honest open risks
- Prereq 3 (unpinned NL corpus) is the real blocker and is **not yet
  resolved** — it needs the `training.py` audit before any GPU spend.
- Throughput/time estimates are ±~50% until the first real run.
- `make_placebo.py` is unit-untestable here without a real langspec
  file; validate its vocab-invariant assertion on the first overlay.
