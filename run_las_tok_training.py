#!/usr/bin/env python3
"""
Hydra entry point for LAS tokenizer training using Hydra configs in
``config/tokenizer_training``.

Example:
    python run_las_tok_training.py
"""

import hydra
from omegaconf import DictConfig
import logging

from las_experiment.tokenizer_training.training import run_tokenizer_training


# ---------------------- Entry point ----------------------
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

    langs = list(cfg.languages.langs)

    run_tokenizer_training(
        langs,
        output_dir=cfg.data.output_dir,
        lines_per_lang=cfg.data.lines_per_lang,
        train_new_base_tokenizer=cfg.training.train_new_base_tokenizer,
        hf_username=cfg.hf.username,
        model_name=cfg.model.model_name,
        training_cfg=cfg.training,
        is_push_to_hf=cfg.hf.push,
    )


if __name__ == "__main__":
    main()
