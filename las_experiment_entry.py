#!/usr/bin/env python3
"""Hydra entry point for LAS experiments."""

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import logging

from las_experiment.run_las_tokenizer import run_tokenizer
from las_experiment.run_las_finetune import run_finetune
from las_experiment.las_lighteval_example import example_lighteval_usage


@hydra.main(config_path="config", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y‑%m‑%d %H:%M:%S",
    )
    logging.info("Loaded configuration:")
    logging.info(cfg)

    if cfg.experiment.get("type", None) == "tokenizer_training":
        logging.info("Starting mode: tokenizer_training")
        run_tokenizer(cfg.experiment)
    elif cfg.experiment.get("type", None) == "finetuning":
        logging.info("Starting mode: finetuning")
        run_finetune(cfg.experiment)
    elif cfg.experiment.get("type", None) == "eval":
        logging.info("Starting mode: eval")
        example_lighteval_usage(cfg.experiment)


if __name__ == "__main__":
    main()
