#!/usr/bin/env python3
"""Hydra entry point for LAS experiments with FineWeb2 {5M, 10M} Datasets"""

import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path

from las_experiment.las_finetune import run_finetune


@hydra.main(config_path="config/finetune", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Ensure only one language per job
    if len(cfg.langs) != 1:
        raise ValueError(f"Expected exactly one language per job, got: {cfg.langs}")

    # Auto-name output_dir and wandb run
    lang = cfg.langs[0]
    dataset_size = str(cfg.dataset.size)
    base_output_dir = Path(cfg.output_dir)

    cfg.output_dir = str(base_output_dir / f"{lang}_{dataset_size}")
    
    if cfg.logger.enabled:
        cfg.logger.run_name = f"{lang}_{dataset_size}_{cfg.logger.run_name}"

    logging.info("Loaded configuration:")
    logging.info(cfg)
    logging.info(f"Output directory set to: {cfg.output_dir}")
    logging.info("Starting mode: finetuning")

    run_finetune(cfg)


if __name__ == "__main__":
    main()
