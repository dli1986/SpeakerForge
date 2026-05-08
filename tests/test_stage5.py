"""Tests for stage5_filter.py — no whisperx dependency."""
from __future__ import annotations

import json
import numpy as np
import soundfile as sf
import pytest
from pathlib import Path

import speakerforge.stages.stage5_filter as stage5
from speakerforge.config import PipelineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(path: Path, duration: float = 3.0, sr: int = 24000, signal: bool = True) -> Path:
    """Write a real WAV to *path*. signal=True writes a sine tone, False writes silence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if signal:
        t = np.linspace(0, duration, int(duration * sr), endpoint=False, dtype=np.float32)
        samples = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    else:
        samples = np.zeros(int(duration * sr), dtype=np.float32)
    sf.write(str(path), samples, sr)
    return path


def _make_transcript(path: Path, text: str = "hello", scores: list[float] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    words = []
    if scores is not None:
        for i, s in enumerate(scores):
            words.append({"word": f"w{i}", "start": i * 0.2, "end": (i + 1) * 0.2, "score": s})
    transcript = {"text": text, "language": "en", "words": words}
    path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1 – FileNotFoundError when no segments exist
# ---------------------------------------------------------------------------

def test_run_raises_if_no_segments(tmp_path, monkeypatch):
    monkeypatch.setattr(stage5, "PROC_DIR", tmp_path)
    speaker = "spk1"
    seg_dir = tmp_path / speaker / "segments"
    seg_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No WAV files in"):
        stage5.run(PipelineConfig(), speaker)


# ---------------------------------------------------------------------------
# Test 2 – Good segment passes all filters
# ---------------------------------------------------------------------------

def test_run_keeps_good_segment(tmp_path, monkeypatch):
    monkeypatch.setattr(stage5, "PROC_DIR", tmp_path)
    speaker = "spk1"
    seg_dir = tmp_path / speaker / "segments"
    tr_dir = tmp_path / speaker / "transcripts"

    wav_path = _make_wav(seg_dir / "BV1abc_0000.wav", duration=3.0)
    _make_transcript(tr_dir / "BV1abc_0000.json", text="hello world", scores=[0.95, 0.92])

    # Patch SNR so we don't depend on actual audio content
    monkeypatch.setattr(stage5, "_compute_snr", lambda p: 30.0)

    stage5.run(PipelineConfig(), speaker)

    out = json.loads((tmp_path / speaker / "filtered.json").read_text(encoding="utf-8"))
    assert len(out) == 1
    entry = out[0]
    assert entry["wav"] == "BV1abc_0000.wav"
    assert entry["text"] == "hello world"
    assert "duration" in entry
    assert "confidence" in entry
    assert "snr_db" in entry


# ---------------------------------------------------------------------------
# Test 3 – Low confidence is filtered out
# ---------------------------------------------------------------------------

def test_run_filters_low_confidence(tmp_path, monkeypatch):
    monkeypatch.setattr(stage5, "PROC_DIR", tmp_path)
    speaker = "spk1"
    seg_dir = tmp_path / speaker / "segments"
    tr_dir = tmp_path / speaker / "transcripts"

    _make_wav(seg_dir / "BV1abc_0000.wav", duration=3.0)
    # Mean confidence = 0.3 — below default min_confidence 0.6
    _make_transcript(tr_dir / "BV1abc_0000.json", text="bad", scores=[0.3, 0.3])

    monkeypatch.setattr(stage5, "_compute_snr", lambda p: 30.0)

    stage5.run(PipelineConfig(), speaker)

    out = json.loads((tmp_path / speaker / "filtered.json").read_text(encoding="utf-8"))
    assert out == []


# ---------------------------------------------------------------------------
# Test 4 – Short duration is filtered out
# ---------------------------------------------------------------------------

def test_run_filters_by_duration(tmp_path, monkeypatch):
    monkeypatch.setattr(stage5, "PROC_DIR", tmp_path)
    speaker = "spk1"
    seg_dir = tmp_path / speaker / "segments"
    tr_dir = tmp_path / speaker / "transcripts"

    # 0.5 s — below default min_duration 1.5 s
    _make_wav(seg_dir / "BV1abc_0000.wav", duration=0.5)
    _make_transcript(tr_dir / "BV1abc_0000.json", text="hi", scores=[0.9])

    # No SNR patch needed — duration filter fires first

    stage5.run(PipelineConfig(), speaker)

    out = json.loads((tmp_path / speaker / "filtered.json").read_text(encoding="utf-8"))
    assert out == []


# ---------------------------------------------------------------------------
# Test 5 – Missing transcript causes segment to be skipped with warning
# ---------------------------------------------------------------------------

def test_run_skips_missing_transcript(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(stage5, "PROC_DIR", tmp_path)
    speaker = "spk1"
    seg_dir = tmp_path / speaker / "segments"

    _make_wav(seg_dir / "BV1abc_0000.wav", duration=3.0)
    # Deliberately omit transcript file

    monkeypatch.setattr(stage5, "_compute_snr", lambda p: 30.0)

    stage5.run(PipelineConfig(), speaker)

    out = json.loads((tmp_path / speaker / "filtered.json").read_text(encoding="utf-8"))
    assert out == []

    captured = capsys.readouterr()
    assert "[WARN]" in captured.out
    assert "BV1abc_0000" in captured.out
