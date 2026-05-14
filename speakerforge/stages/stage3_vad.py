from __future__ import annotations
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from speakerforge.config import PipelineConfig

PROC_DIR = Path("processed")


def run(pipeline_cfg: PipelineConfig, speaker: str) -> None:
    """Segment audio into speech chunks using silero-vad."""
    cfg = pipeline_cfg.stage3
    min_dur = cfg.get("min_duration", 0.8)
    max_dur = cfg.get("max_duration", 8.0)
    min_silence = cfg.get("min_silence_duration", 0.3)

    in_dir = PROC_DIR / speaker / "vocals"
    out_dir = PROC_DIR / speaker / "segments"
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(in_dir.glob("BV*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files in {in_dir}. Run stage2 first.")

    vad_model, get_speech_timestamps, read_audio = _load_silero_vad()

    for wav in tqdm(wav_files, desc="Stage 3: VAD segmentation"):
        _segment_file(wav, out_dir, vad_model, get_speech_timestamps, read_audio,
                      min_dur, max_dur, min_silence)


def _load_silero_vad():
    """Load silero-vad using the pip-installed package API (v5+)."""
    from silero_vad import load_silero_vad, get_speech_timestamps, read_audio
    model = load_silero_vad()
    return model, get_speech_timestamps, read_audio


def _segment_file(wav_path, out_dir, model, get_speech_timestamps, read_audio,
                  min_dur, max_dur, min_silence):
    audio = read_audio(str(wav_path), sampling_rate=16000)
    timestamps = get_speech_timestamps(audio, model, sampling_rate=16000, return_seconds=True)
    raw_segments = [(t["start"], t["end"]) for t in timestamps]
    segments = filter_segments(merge_segments(raw_segments, min_silence), min_dur, max_dur)

    full_audio, sr = sf.read(str(wav_path), dtype="float32")
    for i, (start, end) in enumerate(segments):
        s = int(start * sr)
        e = int(end * sr)
        chunk = full_audio[s:e]
        out_name = f"{wav_path.stem}_{i:04d}.wav"
        sf.write(str(out_dir / out_name), chunk, sr, subtype="PCM_16")
        tr_path = out_dir.parent / "transcripts" / f"{wav_path.stem}_{i:04d}.json"
        if tr_path.exists():
            tr_path.unlink()


def merge_segments(segments: list[tuple[float, float]],
                   min_silence: float) -> list[tuple[float, float]]:
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_end = merged[-1][1]
        if start - prev_end < min_silence:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def filter_segments(segments: list[tuple[float, float]],
                    min_dur: float, max_dur: float) -> list[tuple[float, float]]:
    result = []
    for start, end in segments:
        dur = end - start
        if dur < min_dur:
            continue
        if dur <= max_dur:
            result.append((start, end))
        else:
            # Split long segments into max_dur chunks, drop tail if < min_dur
            cursor = start
            while cursor < end:
                chunk_end = min(cursor + max_dur, end)
                if chunk_end - cursor >= min_dur:
                    result.append((cursor, chunk_end))
                cursor = chunk_end
    return result
