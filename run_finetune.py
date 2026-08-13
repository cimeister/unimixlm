#!/usr/bin/env python3
"""
Hydra entry point for LAS experiments with FineWeb2 {5M, 10M} Datasets
python run_finetune.py \
  langs=[yor] \  # or fil,hau,ibo,ind,lit,nya,ron,sna,som,swh,yor,zsm
  dataset.size=10M  # or 5M
"""

import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path

from las_experiment.finetune import run_finetune, parse_langs


@hydra.main(config_path="config/finetune", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Auto-name output_dir and wandb run
    lang_codes, _ = parse_langs(cfg.langs)
    langs = "-".join(lang_codes)
    dataset_size = str(cfg.dataset.size)
    base_output_dir = Path(cfg.output_dir)

    mode_prefix = {
        "original": "ft",
        "base": "ft-las-base",
        "pathpiece": "ft-pathpiece",
        "langspec": "ft-las-langspec",
    }[cfg.las_mode]
    epoch_suffix = f"_ep{cfg.epochs}" if cfg.epochs > 1 else ""
    lr_str = f"{cfg.learning_rate:.0e}".replace("+", "")  # e.g. "1e-04"
    use_lora = getattr(cfg, "use_lora", True)
    if use_lora:
        lora_suffix = f"_r{cfg.lora.r}"
        uf_suffix = "_uf" if cfg.unfreeze_embeddings else ""
    else:
        lora_suffix = "_embedonly"
        uf_suffix = ""
    cfg.output_dir = str(
        base_output_dir
        / f"{mode_prefix}_{langs}_{dataset_size}{epoch_suffix}_lr{lr_str}{lora_suffix}{uf_suffix}"
    )
    logging.info("Loaded configuration:")
    logging.info(cfg)
    logging.info(f"Output directory set to: {cfg.output_dir}")
    logging.info("Starting mode: finetuning")

    run_finetune(cfg)


if __name__ == "__main__":
    main()
