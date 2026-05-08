import click

DEFAULT_PIPELINE = "pipeline.yaml"
DEFAULT_SOURCES = "sources.yaml"


@click.group()
def cli():
    """SpeakerForge: Bilibili -> speech dataset pipeline."""
    pass


@cli.command()
@click.option("--cookie-file", default="~/.speakerforge/bili_cookies.json",
              show_default=True, help="Path to bili_cookies.json")
def login(cookie_file):
    """Interactive QR-code login to Bilibili (one-time setup)."""
    from speakerforge.bili_auth import _qr_login
    _qr_login(cookie_file)


@cli.command()
@click.option("--speaker", required=True, help="Speaker name from sources.yaml")
@click.option("--sources", default=DEFAULT_SOURCES, show_default=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
@click.option("--dry-run", is_flag=True)
def stage0(speaker, sources, config, dry_run):
    """Stage 0: Resolve Bilibili sources and download videos."""
    from speakerforge.stages.stage0_acquire import run
    from speakerforge.config import load_sources, load_pipeline
    run(load_sources(sources), load_pipeline(config), speaker=speaker, dry_run=dry_run)


@cli.command()
@click.option("--speaker", required=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
def stage1(speaker, config):
    """Stage 1: Extract audio from downloaded videos."""
    from speakerforge.stages.stage1_extract import run
    from speakerforge.config import load_pipeline
    run(load_pipeline(config), speaker=speaker)


@cli.command()
@click.option("--speaker", required=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
def stage2(speaker, config):
    """Stage 2: Vocal isolation with Demucs."""
    from speakerforge.stages.stage2_demucs import run
    from speakerforge.config import load_pipeline
    run(load_pipeline(config), speaker=speaker)


@cli.command()
@click.option("--speaker", required=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
def stage3(speaker, config):
    """Stage 3: VAD segmentation with silero-vad."""
    from speakerforge.stages.stage3_vad import run
    from speakerforge.config import load_pipeline
    run(load_pipeline(config), speaker=speaker)


@cli.command()
@click.option("--speaker", required=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
def stage4(speaker, config):
    """Stage 4: WhisperX transcription + forced alignment."""
    from speakerforge.stages.stage4_align import run
    from speakerforge.config import load_pipeline
    run(load_pipeline(config), speaker=speaker)


@cli.command()
@click.option("--speaker", required=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
def stage5(speaker, config):
    """Stage 5: Quality filtering."""
    from speakerforge.stages.stage5_filter import run
    from speakerforge.config import load_pipeline
    run(load_pipeline(config), speaker=speaker)


@cli.command()
@click.option("--speaker", required=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
def stage6(speaker, config):
    """Stage 6: RMS normalization + silence trimming."""
    from speakerforge.stages.stage6_normalize import run
    from speakerforge.config import load_pipeline
    run(load_pipeline(config), speaker=speaker)


@cli.command()
@click.option("--speaker", required=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
@click.option("--version", default="B", type=click.Choice(["A", "B"]),
              help="A=baseline (no Demucs/VAD), B=full pipeline")
def stage7(speaker, config, version):
    """Stage 7: Package into LJSpeech-format dataset."""
    from speakerforge.stages.stage7_package import run
    from speakerforge.config import load_pipeline
    run(load_pipeline(config), speaker=speaker, version=version)


@cli.command()
@click.option("--speaker", required=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
@click.option("--version", default="B", type=click.Choice(["A", "B"]))
def stage8(speaker, config, version):
    """Stage 8: CosyVoice 2 LoRA fine-tuning."""
    from speakerforge.stages.stage8_finetune import run
    from speakerforge.config import load_pipeline
    run(load_pipeline(config), speaker=speaker, version=version)


@cli.command()
@click.option("--speaker", required=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
@click.option("--version", default="B", type=click.Choice(["A", "B"]))
def stage9(speaker, config, version):
    """Stage 9: Inference + speaker similarity evaluation."""
    from speakerforge.stages.stage9_eval import run
    from speakerforge.config import load_pipeline
    run(load_pipeline(config), speaker=speaker, version=version)


@cli.command(name="run")
@click.option("--speaker", required=True)
@click.option("--sources", default=DEFAULT_SOURCES, show_default=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
@click.option("--stages", "stages_range", default="0-7", show_default=True,
              help="Stage range e.g. '0-7' or '2-5'")
@click.option("--version", default="B", type=click.Choice(["A", "B"]))
def run_pipeline(speaker, sources, config, stages_range, version):
    """Run a range of stages in sequence."""
    from speakerforge.config import load_sources, load_pipeline
    src_cfg = load_sources(sources)
    pipe_cfg = load_pipeline(config)

    try:
        start, end = (int(x) for x in stages_range.split("-"))
    except ValueError:
        raise click.BadParameter("must be in the form 'START-END', e.g. '0-7'", param_hint="--stages")
    if start > end:
        raise click.BadParameter("start must be <= end", param_hint="--stages")
    if not (0 <= start <= 9 and 0 <= end <= 9):
        raise click.BadParameter("stages must be in range 0-9", param_hint="--stages")

    for i in range(start, end + 1):
        click.echo(f"\n{'='*50}\nRunning stage {i}...\n{'='*50}")
        _dispatch_stage(i, src_cfg, pipe_cfg, speaker, version)


def _dispatch_stage(i, src_cfg, pipe_cfg, speaker, version):
    if i == 0:
        from speakerforge.stages.stage0_acquire import run
        run(src_cfg, pipe_cfg, speaker=speaker)
    elif i == 1:
        from speakerforge.stages.stage1_extract import run
        run(pipe_cfg, speaker=speaker)
    elif i == 2:
        from speakerforge.stages.stage2_demucs import run
        run(pipe_cfg, speaker=speaker)
    elif i == 3:
        from speakerforge.stages.stage3_vad import run
        run(pipe_cfg, speaker=speaker)
    elif i == 4:
        from speakerforge.stages.stage4_align import run
        run(pipe_cfg, speaker=speaker)
    elif i == 5:
        from speakerforge.stages.stage5_filter import run
        run(pipe_cfg, speaker=speaker)
    elif i == 6:
        from speakerforge.stages.stage6_normalize import run
        run(pipe_cfg, speaker=speaker)
    elif i == 7:
        from speakerforge.stages.stage7_package import run
        run(pipe_cfg, speaker=speaker, version=version)
    elif i == 8:
        from speakerforge.stages.stage8_finetune import run
        run(pipe_cfg, speaker=speaker, version=version)
    elif i == 9:
        from speakerforge.stages.stage9_eval import run
        run(pipe_cfg, speaker=speaker, version=version)


@cli.command()
@click.option("--speaker", required=True)
def stats(speaker):
    """Show dataset statistics for a speaker."""
    import pandas as pd
    from pathlib import Path
    found = False
    for version in ["A", "B"]:
        csv = Path(f"dataset/{speaker}_version{version}/metadata.csv")
        if csv.exists():
            found = True
            df = pd.read_csv(csv)
            total = df["duration"].sum()
            click.echo(f"\nVersion {version}: {len(df)} segments, {total/60:.1f} min total")
            click.echo(df["duration"].describe().to_string())
    if not found:
        click.echo(f"No dataset found for speaker '{speaker}'. Run stage7 first.")
