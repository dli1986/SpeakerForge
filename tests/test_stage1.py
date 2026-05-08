import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from speakerforge.stages.stage1_extract import extract_audio, run
from speakerforge.config import PipelineConfig


def test_extract_audio_calls_ffmpeg(tmp_path):
    input_mp4 = tmp_path / "BV1abc.mp4"
    input_mp4.write_bytes(b"fake")
    output_wav = tmp_path / "BV1abc.wav"
    with patch("speakerforge.stages.stage1_extract.ffmpeg") as mock_ff:
        mock_stream = MagicMock()
        mock_ff.input.return_value = mock_stream
        mock_stream.output.return_value = mock_stream
        mock_stream.overwrite_output.return_value = mock_stream
        extract_audio(input_mp4, output_wav, sample_rate=24000)
        mock_ff.input.assert_called_once_with(str(input_mp4))


def test_run_skips_existing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "raw_sources/bilibili/alice"
    raw.mkdir(parents=True)
    (raw / "BV1.mp4").write_bytes(b"fake")
    out = tmp_path / "processed/alice/audio"
    out.mkdir(parents=True)
    (out / "BV1.wav").write_bytes(b"exists")

    with patch("speakerforge.stages.stage1_extract.extract_audio") as mock_ex:
        run(PipelineConfig(stage1={"sample_rate": 24000}), speaker="alice")
        mock_ex.assert_not_called()


def test_run_raises_if_no_mp4(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "raw_sources/bilibili/bob").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        run(PipelineConfig(stage1={"sample_rate": 24000}), speaker="bob")
