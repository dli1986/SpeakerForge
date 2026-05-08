"""Tests for stage7_package.py — LJSpeech-format dataset packaging."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import speakerforge.stages.stage7_package as stage7
from speakerforge.config import PipelineConfig


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_wav(path: Path, duration: float = 2.0, sr: int = 24000) -> None:
    """Write a synthetic 440 Hz sine-wave WAV at *path*."""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.1).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_version_b_packages_correctly(tmp_path, monkeypatch):
    """Version B: reads manifest.json and copies WAVs with correct metadata.csv."""
    proc_dir = tmp_path / "proc"
    dataset_dir = tmp_path / "dataset"
    monkeypatch.setattr(stage7, "PROC_DIR", proc_dir)
    monkeypatch.setattr(stage7, "DATASET_DIR", dataset_dir)

    norm_dir = proc_dir / "alice" / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for i in range(2):
        wav_name = f"BV1abc_{i:04d}.wav"
        _make_wav(norm_dir / wav_name, duration=1.5)
        entries.append({"wav": wav_name, "text": f"hello world {i}", "duration": 1.5})

    (norm_dir / "manifest.json").write_text(
        json.dumps(entries, ensure_ascii=False), encoding="utf-8"
    )

    stage7.run(PipelineConfig(), "alice", version="B")

    out_wavs = dataset_dir / "alice_versionB" / "wavs"
    assert out_wavs.is_dir()
    wav_files = list(out_wavs.glob("*.wav"))
    assert len(wav_files) == 2

    csv_path = dataset_dir / "alice_versionB" / "metadata.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    assert len(df) == 2
    assert list(df.columns) == ["filename", "text", "duration"]


def test_version_a_packages_correctly(tmp_path, monkeypatch):
    """Version A: pairs audio/ WAVs with transcripts/ JSONs."""
    proc_dir = tmp_path / "proc"
    dataset_dir = tmp_path / "dataset"
    monkeypatch.setattr(stage7, "PROC_DIR", proc_dir)
    monkeypatch.setattr(stage7, "DATASET_DIR", dataset_dir)

    audio_dir = proc_dir / "alice" / "audio"
    transcript_dir = proc_dir / "alice" / "transcripts"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    stems = ["clip_0000", "clip_0001"]
    for stem in stems:
        _make_wav(audio_dir / f"{stem}.wav", duration=2.0)
        (transcript_dir / f"{stem}.json").write_text(
            json.dumps({"text": f"transcript for {stem}"}), encoding="utf-8"
        )

    stage7.run(PipelineConfig(), "alice", version="A")

    out_wavs = dataset_dir / "alice_versionA" / "wavs"
    assert out_wavs.is_dir()
    assert len(list(out_wavs.glob("*.wav"))) == 2

    csv_path = dataset_dir / "alice_versionA" / "metadata.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    assert len(df) == 2
    assert set(df["filename"]) == set(stems)


def test_version_b_raises_if_no_manifest(tmp_path, monkeypatch):
    """FileNotFoundError when normalized/manifest.json is absent."""
    proc_dir = tmp_path / "proc"
    dataset_dir = tmp_path / "dataset"
    monkeypatch.setattr(stage7, "PROC_DIR", proc_dir)
    monkeypatch.setattr(stage7, "DATASET_DIR", dataset_dir)

    # Create speaker dir but no manifest
    (proc_dir / "alice" / "normalized").mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="No normalized manifest for alice"):
        stage7.run(PipelineConfig(), "alice", version="B")


def test_version_b_skips_existing_wav(tmp_path, monkeypatch):
    """Pre-existing WAV in output wavs/ should not be overwritten."""
    proc_dir = tmp_path / "proc"
    dataset_dir = tmp_path / "dataset"
    monkeypatch.setattr(stage7, "PROC_DIR", proc_dir)
    monkeypatch.setattr(stage7, "DATASET_DIR", dataset_dir)

    norm_dir = proc_dir / "alice" / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)

    wav_name = "BV1abc_0000.wav"
    _make_wav(norm_dir / wav_name, duration=1.0)

    entries = [{"wav": wav_name, "text": "hello", "duration": 1.0}]
    (norm_dir / "manifest.json").write_text(
        json.dumps(entries, ensure_ascii=False), encoding="utf-8"
    )

    # Pre-create the output WAV
    out_wavs_dir = dataset_dir / "alice_versionB" / "wavs"
    out_wavs_dir.mkdir(parents=True, exist_ok=True)
    dst = out_wavs_dir / wav_name
    _make_wav(dst, duration=1.0)
    original_mtime = dst.stat().st_mtime

    # Small sleep so any write would bump mtime
    time.sleep(0.05)

    stage7.run(PipelineConfig(), "alice", version="B")

    assert dst.stat().st_mtime == original_mtime, "Existing WAV should not be overwritten"


def test_metadata_csv_has_correct_columns(tmp_path, monkeypatch):
    """metadata.csv must have exactly columns: filename, text, duration."""
    proc_dir = tmp_path / "proc"
    dataset_dir = tmp_path / "dataset"
    monkeypatch.setattr(stage7, "PROC_DIR", proc_dir)
    monkeypatch.setattr(stage7, "DATASET_DIR", dataset_dir)

    norm_dir = proc_dir / "alice" / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)

    wav_name = "BV1abc_0000.wav"
    _make_wav(norm_dir / wav_name, duration=2.0)
    entries = [{"wav": wav_name, "text": "test utterance", "duration": 2.0}]
    (norm_dir / "manifest.json").write_text(
        json.dumps(entries, ensure_ascii=False), encoding="utf-8"
    )

    stage7.run(PipelineConfig(), "alice", version="B")

    csv_path = dataset_dir / "alice_versionB" / "metadata.csv"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    assert list(df.columns) == ["filename", "text", "duration"]
    assert df.iloc[0]["filename"] == "BV1abc_0000"
    assert df.iloc[0]["text"] == "test utterance"
    assert isinstance(df.iloc[0]["duration"], float)
