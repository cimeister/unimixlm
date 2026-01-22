#!/usr/bin/env python3
"""
Hydra entry point for LAS experiments with FineWeb2 {5M, 10M} Datasets
python las_finetune.py \
  langs=[yor] \  # or fil,hau,ibo,ind,lit,nya,ron,sna,som,swh,yor,zsm
  dataset.size=10M  # or 5M
"""

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
    langs = "-".join(cfg.langs)
    dataset_size = str(cfg.dataset.size)
    base_output_dir = Path(cfg.output_dir)

    cfg.output_dir = str(
        base_output_dir
        / f"{'ft' if not cfg.las_tokenizer_path else 'ft-las'}_{langs}_{dataset_size}_modified"
    )
    logging.info("Loaded configuration:")
    logging.info(cfg)
    logging.info(f"Output directory set to: {cfg.output_dir}")
    logging.info("Starting mode: finetuning")

    run_finetune(cfg)


if __name__ == "__main__":
    main()
