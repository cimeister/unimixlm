# EMNLP main-paper SLURM job suite

End-to-end submission harness for the EMNLP main-paper experiments on
Cambridge CSD3 / Wilkes3.

```
scripts/emnlp/
├── run_emnlp_sl2.sh    # SL2 wrapper (36 h cap, 64 GPU cap) — Core suite
├── run_emnlp_sl3.sh    # SL3 wrapper (12 h cap, 32 GPU cap) — Extension suite
├── _payload_tok.sh     # tokenizer-training payload
├── _payload_ft.sh      # finetuning payload
├── _payload_eval.sh    # evaluation payload
├── submit_all.sh       # driver: enumerates Core + Extension and sbatch's each cell
└── README.md
```

## Prerequisites (one-time, before any sbatch)

The plan assumes the following are already in place (see
`the experiment plan` §1):

1. New model configs registered:
   - `config/finetune/model/qwen2_5-0_5b.yaml`
   - `config/finetune/model/gemma3-1b.yaml`
2. **Vocab-size override** (`vocab_size: 100000 | 150000 | 256000`) wired
   into `code/em_unigram_trainer.py`,
   `las_experiment/tokenizer_training/{training,code_tokenizer_training}.py`
   and the entry points `run_las_tok_training.py` / `run_las_code_tok.py`.
3. **English rehearsal slice** added on the code path
   (`thestack_finetune_dataset.py`).
4. **50M / 500M dataset variants** built and pushed to HF.
5. **Multilingual eval tasks** (MultiBLiMP, XNLI, WikiANN) wired into
   `las_experiment/downstream_eval.py`.
6. `HF_TOKEN` exported in your environment.
7. `venvs/demo/bin/activate` (or `.venv/bin/activate`) populated by
   `setup.sh`.

## Per-job design

Each sbatch reserves **1 ampere node = 4 A100 GPUs** and runs a single
finetune / eval / tokenizer-training cell. Concurrency is capped naturally by
the SLURM account limits:

| Queue | GPUs concurrent | Max wallclock |
|-------|----------------:|--------------:|
| SL2   |              64 |          36 h |
| SL3   |              32 |          12 h |

The wrapper uses the user's `stream_missing_28b_sl2.sh` skeleton: module
load, venv activation, supervisor retry loop (`MAX_ATTEMPTS=8`), HF-token
check, and env-driven experiment axes.

## Routing rule

- **Core suite** (paper-critical) → SL2 (`run_emnlp_sl2.sh`).
- **Extension suite** (sweeps) → SL3 (`run_emnlp_sl3.sh`).
- **One overflow cell** — Olmo3-7B / unfrozen / 500M, ~30 h — is force-routed
  to SL2 by `submit_all.sh`, even when it appears in §5.2 of the Extension
  grid.

## Usage

```bash
# Inspect what would be submitted, no sbatch.
bash scripts/emnlp/submit_all.sh --dry-run

# Submit only the Core suite (SL2).
bash scripts/emnlp/submit_all.sh --filter suite=core

# Submit the tokenizer-training jobs first (prerequisite for langspec FT).
bash scripts/emnlp/submit_all.sh --filter kind=tok

# Submit only Qwen / code experiments.
bash scripts/emnlp/submit_all.sh --filter model=qwen2_5-0_5b --filter domain=code

# Submit a single section of the plan (e.g. §4.2 Code FT only).
bash scripts/emnlp/submit_all.sh --filter section=4.2

# Stop after submitting the first 16 jobs (handy for canary runs).
bash scripts/emnlp/submit_all.sh --max-jobs 16
```

Filters AND-combine; pass `--filter` multiple times. Valid keys:
`suite`, `kind`, `model`, `domain`, `section`.

## Submitting one cell by hand

Skip the driver and submit a single cell directly:

```bash
sbatch --export=ALL,\
HF_TOKEN="$HF_TOKEN",\
EXP_KIND=ft,\
EXP_DOMAIN=code,\
EXP_MODEL=qwen2_5-0_5b,\
EXP_LANG=python,\
EXP_REHEARSAL=eng,\
EXP_LAS_MODE=langspec,\
EXP_VOCAB=256k,\
EXP_UNFREEZE=1,\
EXP_LR=2e-4,\
EXP_DATA_SIZE=100M,\
EXP_LORA_PRESET=r64-k128,\
EXP_EPOCHS=5 \
scripts/emnlp/run_emnlp_sl2.sh
```

Switch `run_emnlp_sl2.sh` → `run_emnlp_sl3.sh` and ensure the cell fits in
12 h.

## Logs and outputs

```
logs/emnlp/                         # SLURM stdout/stderr per job
finetuned_model_emnlp/<run-tag>/<lang>/   # FT artefacts (also pushed to HF)
eval_results_emnlp/<run-tag>/<lang>/      # eval JSONs + .eval_complete sentinel
tokenizers_emnlp/las-<domain>-<model>-vocab<vocab>/   # LAS tokenizers
```

`<run-tag>` =
`${EXP_MODEL}_${EXP_DOMAIN}_${EXP_LAS_MODE}_v${EXP_VOCAB}_e${EXP_UNFREEZE}_${EXP_DATA_SIZE}_lr${EXP_LR}_lora${EXP_LORA_PRESET}_ep${EXP_EPOCHS}`

Eval is gated by `.eval_complete` so retries are idempotent — delete the
sentinel to re-run that cell.

## Budget

| Suite     | Jobs |   GPU-h | Queue |
|-----------|-----:|--------:|-------|
| Core      |  136 |    ~493 | SL2   |
| Extension |  298 |  ~1,214 | SL3   |
| **Total** |  434 |  ~1,707 | (of 5,024 available)  |

See `linear-roaming-squirrel.md` §3–§6 for the per-job runtime model and the
per-row breakdown.
