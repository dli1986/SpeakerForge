from __future__ import annotations

import click

DEFAULT_CONFIG = "config.yaml"


@click.group()
def cli():
    """SpeakerForge Phase 2: CSM-1B LoRA voice injection."""


@cli.command()
@click.option("--config", default=DEFAULT_CONFIG, show_default=True)
def stage1(config):
    """Stage 1: Prepare HuggingFace dataset from Phase 1 output."""
    from sesame.config import load_sesame_config
    from sesame.stages.stage1_prep import run
    run(load_sesame_config(config))


@cli.command()
@click.option("--config", default=DEFAULT_CONFIG, show_default=True)
def stage2(config):
    """Stage 2: LoRA fine-tuning with Unsloth."""
    from sesame.config import load_sesame_config
    from sesame.stages.stage2_train import run
    run(load_sesame_config(config))


@cli.command()
@click.option("--config", default=DEFAULT_CONFIG, show_default=True)
def stage3(config):
    """Stage 3: Inference and resemblyzer similarity evaluation."""
    from sesame.config import load_sesame_config
    from sesame.stages.stage3_eval import run
    run(load_sesame_config(config))
