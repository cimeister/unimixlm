#!/usr/bin/env python3
"""
run_las_code_tok_training.py

Hydra entry point for training code-language-specific LAS UnigramLM
tokenizers from The-Stack-10K.

Mirrors run_las_tok_training.py exactly, but delegates to
las_experiment.code_tokenizer_training instead of
las_experiment.tokenizer_training.training.

Config lives in:  config/code_tokenizer_training/default.yaml

Example
-------
# Train on default language set (all 30 from The Stack)
python run_las_code_tok_training.py

# Override languages and char budget on the command line
python run_las_code_tok_training.py \
    languages.langs="[python,rust,javascript]" \
    data.char_budget=2000000

# Dry-run without pushing to Hub
python run_las_code_tok_training.py hf.push=false
"""

import logging

import hydra
from omegaconf import DictConfig

from las_experiment.tokenizer_training.code_tokenizer_training import run_code_tokenizer_training

@hydra.main(
    config_path="config/tokenizer_training",
    config_name="default",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_code_tokenizer_training(
        langs=list(cfg.languages.langs),
        output_dir=cfg.data.output_dir,
        char_budget=cfg.data.char_budget,
        train_new_base_tokenizer=cfg.training.train_new_base_tokenizer,
        hf_username=cfg.hf.username,
        model_name=cfg.model.model_name,
        training_cfg=dict(cfg.training),
        is_push_to_hf=cfg.hf.push,
    )


if __name__ == "__main__":
    main()