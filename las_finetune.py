#!/usr/bin/env python3
"""Hydra entry point for LAS experiments."""

import hydra
from omegaconf import DictConfig
import logging

from las_experiment.las_finetune import run_finetune


@hydra.main(config_path="config/finetune", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Loaded configuration:")
    logging.info(cfg)

    logging.info("Starting mode: finetuning")
    run_finetune(cfg)


if __name__ == "__main__":
    main()
