# LangMAP finetuning runbook

Prepared scope:

| Stage | Models | Langs | Arms | Regime |
|---|---|---|---|---|
| Initial (ASAP) | Qwen2.5-1.5B | tur → fin → hun | original now, langspec gated | unfrozen+LoRA, 500M, 1ep, 5 ckpts |
| FT sweep | Qwen2.5-1.5B, Gemma-2-2b, Mistral-7B-v0.3 | tur, fin, hun | original, langspec | unfrozen+LoRA, 500M, 1ep, 5 ckpts |

Checkpoints: `num_checkpoints=5` over a single 500M-token / 1-epoch run →
`save_strategy="steps"` with `save_steps ≈ total_steps/5` ≈ one checkpoint
per ~100M tokens. All milestone checkpoints are kept.

## One-time prep on the pod
```bash
cd unimixlm && source .venv/bin/activate   # or bash setup.sh first
huggingface-cli login                       # dataset + model access
bash scripts/prep_model_tokenizers.sh       # snapshots for the original arm
```

## Run — original arm (no overlay needed, runs immediately)
```bash
# Initial Qwen experiment, Turkish first:
LANGS="tur" ARMS="original" bash scripts/run_langmap_qwen_initial.sh
# then fin, hun:
LANGS="fin hun" ARMS="original" bash scripts/run_langmap_qwen_initial.sh

# Full sweep, original arm only:
ARMS="original" bash scripts/run_langmap_ft_sweep.sh
```

## langspec arm — BLOCKED until overlay tokenizers exist
`las_mode=langspec` needs, per model, a directory containing
`base_tokenizer.json`, `langspec_{tur,fin,hun}_Latn.json`, and a
`model_tokenizer/` subdir:

| model config | expected overlay dir (override via env) |
|---|---|
| qwen2_5-1_5b | `./las_qwen2_5_1_5b`  (`LAS_QWEN=`) |
| gemma2-2b | `./las_gemma2_2b`  (`LAS_GEMMA=`) |
| mistral-7b-v0_3 | `./las_mistral_7b_v0_3`  (`LAS_MISTRAL=`) |

These are produced by the LAS tokenizer-training pipeline (the
`langmap-orchestrate` Sweep-1 tokenizer step / `unimixlm` tokenizer
training) for the specific base vocab — none exist for these three
models yet. Until then both launchers auto-mark langspec cells as
`gated` in the run manifest and skip them — **no silent GPU spend**.
Once an overlay dir is present, re-run the same command with
`ARMS="langspec"` (or default `ARMS`) and only the gated cells execute.

## Multi-GPU pod
The launchers run one `python run_finetune.py` per cell (no DDP). To use
N GPUs, run N copies pinned to disjoint slices, e.g.:
```bash
CUDA_VISIBLE_DEVICES=0 MODELS=mistral-7b-v0_3 bash scripts/run_langmap_ft_sweep.sh &
CUDA_VISIBLE_DEVICES=1 MODELS=gemma2-2b       bash scripts/run_langmap_ft_sweep.sh &
CUDA_VISIBLE_DEVICES=2 MODELS=qwen2_5-1_5b    bash scripts/run_langmap_ft_sweep.sh &
wait
```
Each 7B/2.6B/1.5B run (unfrozen+LoRA, seq 1024) fits on a single
H200 (141 GB) — parallel independent runs maximise throughput at the
same $/GPU-hr.
