"""Tests for stage4_align.py — all whisperx calls are mocked."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to build a minimal fake whisperx module so the lazy import inside
# stage4_align.py resolves even when the real package is absent.
# ---------------------------------------------------------------------------

def _make_fake_whisperx():
    """Return a MagicMock that looks enough like whisperx for our tests."""
    wx = MagicMock(name="whisperx")
    wx.load_audio = MagicMock(return_value=[0.0] * 16000)  # 1 s silence
    wx.load_model = MagicMock()
    wx.load_align_model = MagicMock(return_value=(MagicMock(), MagicMock()))
    wx.align = MagicMock(
        return_value={
            "segments": [
                {
                    "text": "你好世界",
                    "words": [
                        {"word": "你好", "start": 0.12, "end": 0.45, "score": 0.98},
                        {"word": "世界", "start": 0.50, "end": 0.80, "score": 0.95},
                    ],
                }
            ]
        }
    )
    return wx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_speaker(tmp_path):
    """Create a temporary processed/<speaker> directory tree."""
    seg_dir = tmp_path / "processed" / "spk1" / "segments"
    seg_dir.mkdir(parents=True)
    (tmp_path / "processed" / "spk1" / "transcripts").mkdir(parents=True)
    return tmp_path, "spk1"


def _make_wavs(seg_dir: Path, n: int = 1):
    for i in range(n):
        (seg_dir / f"BV1abc_{i:04d}.wav").write_bytes(b"\x00" * 100)


# ---------------------------------------------------------------------------
# Test: skip if JSON already exists
# ---------------------------------------------------------------------------

def test_run_skips_existing_transcript(tmp_speaker, monkeypatch):
    tmp_path, speaker = tmp_speaker
    seg_dir = tmp_path / "processed" / speaker / "segments"
    tr_dir = tmp_path / "processed" / speaker / "transcripts"

    # Create one segment WAV
    _make_wavs(seg_dir, 1)
    stem = "BV1abc_0000"
    # Pre-create the transcript JSON
    (tr_dir / f"{stem}.json").write_text("{}", encoding="utf-8")

    fake_wx = _make_fake_whisperx()
    fake_model = MagicMock()
    fake_wx.load_model.return_value = fake_model

    monkeypatch.chdir(tmp_path)

    with patch.dict(sys.modules, {"whisperx": fake_wx}), \
         patch("speakerforge.stages.stage4_align._load_whisperx_model",
               return_value=(fake_model, MagicMock(), MagicMock())), \
         patch("speakerforge.stages.stage4_align._default_device", return_value="cpu"):

        from speakerforge.stages import stage4_align
        from speakerforge.config import PipelineConfig

        stage4_align.run(PipelineConfig(), speaker)

    # transcribe should NOT have been called because the JSON existed
    fake_model.transcribe.assert_not_called()


# ---------------------------------------------------------------------------
# Test: FileNotFoundError when no segments exist
# ---------------------------------------------------------------------------

def test_run_raises_if_no_segments(tmp_speaker, monkeypatch):
    tmp_path, speaker = tmp_speaker
    monkeypatch.chdir(tmp_path)

    with patch("speakerforge.stages.stage4_align._load_whisperx_model",
               return_value=(MagicMock(), MagicMock(), MagicMock())), \
         patch("speakerforge.stages.stage4_align._default_device", return_value="cpu"):

        from speakerforge.stages import stage4_align
        from speakerforge.config import PipelineConfig

        with pytest.raises(FileNotFoundError, match="No segments in"):
            stage4_align.run(PipelineConfig(), speaker)


# ---------------------------------------------------------------------------
# Test: output JSON has correct structure
# ---------------------------------------------------------------------------

def test_transcript_json_structure(tmp_speaker, monkeypatch):
    tmp_path, speaker = tmp_speaker
    seg_dir = tmp_path / "processed" / speaker / "segments"
    tr_dir = tmp_path / "processed" / speaker / "transcripts"
    _make_wavs(seg_dir, 1)

    fake_wx = _make_fake_whisperx()
    fake_model = MagicMock()
    fake_model.transcribe.return_value = {"segments": [{"text": "你好世界"}]}
    fake_wx.load_model.return_value = fake_model

    monkeypatch.chdir(tmp_path)

    with patch.dict(sys.modules, {"whisperx": fake_wx}), \
         patch("speakerforge.stages.stage4_align._load_whisperx_model",
               return_value=(fake_model, MagicMock(), MagicMock())), \
         patch("speakerforge.stages.stage4_align._default_device", return_value="cpu"):

        from speakerforge.stages import stage4_align
        from speakerforge.config import PipelineConfig

        stage4_align.run(PipelineConfig(), speaker)

    json_files = list(tr_dir.glob("*.json"))
    assert len(json_files) == 1

    data = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert "text" in data
    assert "language" in data
    assert "words" in data
    assert isinstance(data["words"], list)
    assert len(data["words"]) > 0
    w = data["words"][0]
    assert "word" in w
    assert "start" in w
    assert "end" in w
    assert "score" in w


# ---------------------------------------------------------------------------
# Test: continues on failure (warn but keep going)
# ---------------------------------------------------------------------------

def test_run_continues_on_failure(tmp_speaker, monkeypatch, capsys):
    tmp_path, speaker = tmp_speaker
    seg_dir = tmp_path / "processed" / speaker / "segments"
    _make_wavs(seg_dir, 3)

    call_count = {"n": 0}

    def boom_on_second(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated transcription failure")
        # Return a valid aligned result
        return {
            "segments": [
                {
                    "text": "ok",
                    "words": [{"word": "ok", "start": 0.1, "end": 0.2, "score": 0.9}],
                }
            ]
        }

    fake_wx = _make_fake_whisperx()
    fake_wx.align.side_effect = boom_on_second

    fake_model = MagicMock()
    fake_model.transcribe.return_value = {"segments": [{"text": "ok"}]}

    monkeypatch.chdir(tmp_path)

    with patch.dict(sys.modules, {"whisperx": fake_wx}), \
         patch("speakerforge.stages.stage4_align._load_whisperx_model",
               return_value=(fake_model, MagicMock(), MagicMock())), \
         patch("speakerforge.stages.stage4_align._default_device", return_value="cpu"):

        from speakerforge.stages import stage4_align
        from speakerforge.config import PipelineConfig

        # Should NOT raise
        stage4_align.run(PipelineConfig(), speaker)

    # align was called 3 times (once per file)
    assert fake_wx.align.call_count == 3

    captured = capsys.readouterr()
    assert "[WARN]" in captured.out
