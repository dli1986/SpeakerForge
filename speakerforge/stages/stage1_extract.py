from __future__ import annotations
from pathlib import Path

import ffmpeg
from tqdm import tqdm

from speakerforge.config import PipelineConfig

PROC_DIR = Path("processed")


def run(pipeline_cfg: PipelineConfig, speaker: str) -> None:
    """Extract audio from raw MP4 files to mono WAV."""
    sr = pipeline_cfg.stage1.get("sample_rate", 24000)
    raw_dir = Path(pipeline_cfg.stage0.get("output_root", "raw_sources")) / "bilibili" / speaker
    out_dir = PROC_DIR / speaker / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    mp4_files = sorted(raw_dir.glob("BV*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"No MP4 files in {raw_dir}. Run stage0 first.")

    for mp4 in tqdm(mp4_files, desc="Stage 1: extracting audio"):
        wav_out = out_dir / (mp4.stem + ".wav")
        if wav_out.exists():
            continue
        extract_audio(mp4, wav_out, sample_rate=sr)


def extract_audio(input_path: Path, output_path: Path, sample_rate: int = 24000) -> None:
    """Extract audio from video file as mono WAV at given sample rate."""
    (
        ffmpeg
        .input(str(input_path))
        .output(str(output_path), ac=1, ar=sample_rate, vn=None, acodec="pcm_s16le")
        .overwrite_output()
        .run(quiet=True)
    )
