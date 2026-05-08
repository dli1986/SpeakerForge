"""Tests for stage6_normalize.py — uses real temp WAV files via numpy + soundfile."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import speakerforge.stages.stage6_normalize as stage6
from speakerforge.config import PipelineConfig


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_wav(path: Path, duration: float = 2.0, sr: int = 24000) -> None:
    """Write a synthetic 440 Hz sine-wave WAV at *path*."""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.1).astype(np.float32)  # low amplitude
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)


def _make_filtered_json(speaker_dir: Path, entries: list[dict]) -> None:
    (speaker_dir / "filtered.json").write_text(
        json.dumps(entries, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_run_raises_if_no_filtered_json(tmp_path, monkeypatch):
    """FileNotFoundError when filtered.json is absent."""
    monkeypatch.setattr(stage6, "PROC_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="No filtered.json for"):
        stage6.run(PipelineConfig(), speaker="spk1")


def test_run_raises_if_empty_filtered_json(tmp_path, monkeypatch):
    """ValueError when filtered.json contains an empty list."""
    monkeypatch.setattr(stage6, "PROC_DIR", tmp_path)
    speaker_dir = tmp_path / "spk1"
    speaker_dir.mkdir(parents=True, exist_ok=True)
    _make_filtered_json(speaker_dir, [])
    with pytest.raises(ValueError, match="filtered.json is empty for"):
        stage6.run(PipelineConfig(), speaker="spk1")


def test_normalize_wav_changes_rms(tmp_path):
    """_normalize_wav output should have RMS close to target_rms_db."""
    sr = 24000
    src = tmp_path / "in.wav"
    dst = tmp_path / "out.wav"
    _make_wav(src, duration=1.0, sr=sr)

    target_db = -20.0
    stage6._normalize_wav(src, dst, target_rms_db=target_db, trim_db=-80.0)

    audio_out, _ = sf.read(str(dst), dtype="float32")
    rms_out = float(np.sqrt(np.mean(audio_out ** 2)))
    target_rms = 10 ** (target_db / 20.0)
    assert abs(rms_out - target_rms) < 0.005, (
        f"RMS {rms_out:.4f} not close to target {target_rms:.4f}"
    )


def test_trim_silence_removes_edges(tmp_path):
    """_trim_silence should remove silent frames at start and end."""
    sr = 24000
    silence_samples = 4800  # 0.2 s of silence
    signal_samples = 24000  # 1 s of 440 Hz tone

    t = np.linspace(0, 1.0, signal_samples, endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    silence = np.zeros(silence_samples, dtype=np.float32)
    audio = np.concatenate([silence, tone, silence])
    original_len = len(audio)

    trimmed = stage6._trim_silence(audio, sr, threshold_db=-40.0)

    assert len(trimmed) < original_len, "trimmed audio should be shorter than original"
    # The silence frames should be gone
    assert len(trimmed) <= signal_samples + 2  # allow 1-sample rounding each side


def test_run_skips_existing_output(tmp_path, monkeypatch):
    """If a normalized WAV already exists it must not be overwritten."""
    monkeypatch.setattr(stage6, "PROC_DIR", tmp_path)
    speaker = "spk1"
    speaker_dir = tmp_path / speaker
    seg_dir = speaker_dir / "segments"
    norm_dir = speaker_dir / "normalized"

    seg_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)

    wav_name = "seg_0001.wav"
    _make_wav(seg_dir / wav_name)

    # Pre-populate normalized dir with a sentinel file
    dst = norm_dir / wav_name
    _make_wav(dst)
    original_mtime = dst.stat().st_mtime

    entries = [{"wav": wav_name, "text": "hello", "duration": 2.0, "confidence": 0.9, "snr_db": 15.0}]
    _make_filtered_json(speaker_dir, entries)

    # Small sleep to ensure any write would bump mtime
    time.sleep(0.05)

    stage6.run(PipelineConfig(), speaker=speaker)

    assert dst.stat().st_mtime == original_mtime, "Existing file should not have been overwritten"


def test_run_writes_manifest(tmp_path, monkeypatch):
    """Full run with 2 WAVs should produce manifest.json with 2 entries."""
    monkeypatch.setattr(stage6, "PROC_DIR", tmp_path)
    speaker = "spk1"
    speaker_dir = tmp_path / speaker
    seg_dir = speaker_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for i in range(2):
        wav_name = f"seg_{i:04d}.wav"
        _make_wav(seg_dir / wav_name, duration=1.0)
        entries.append(
            {
                "wav": wav_name,
                "text": f"utterance {i}",
                "duration": 1.0,
                "confidence": 0.95,
                "snr_db": 20.0 + i,
            }
        )

    _make_filtered_json(speaker_dir, entries)

    stage6.run(PipelineConfig(), speaker=speaker)

    manifest_path = tmp_path / speaker / "normalized" / "manifest.json"
    assert manifest_path.exists(), "manifest.json should be created"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest) == 2, f"Expected 2 entries, got {len(manifest)}"

    required_keys = {"wav", "text", "duration", "confidence", "snr_db"}
    for entry in manifest:
        assert required_keys.issubset(entry.keys()), (
            f"Entry missing keys: {required_keys - entry.keys()}"
        )
        assert isinstance(entry["duration"], float)
        assert entry["duration"] > 0
