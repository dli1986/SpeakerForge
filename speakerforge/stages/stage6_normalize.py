from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from speakerforge.config import PipelineConfig

PROC_DIR = Path("processed")


def run(pipeline_cfg: PipelineConfig, speaker: str) -> None:
    """RMS-normalize and silence-trim filtered WAV segments."""
    cfg = pipeline_cfg.stage6
    target_rms_db: float = cfg.get("target_rms_db", -20.0)
    trim_silence_db: float = cfg.get("trim_silence_db", -40.0)

    speaker_dir = PROC_DIR / speaker
    filtered_json = speaker_dir / "filtered.json"

    if not filtered_json.exists():
        raise FileNotFoundError(f"No filtered.json for {speaker}. Run stage5 first.")

    entries: list[dict] = json.loads(filtered_json.read_text(encoding="utf-8"))
    if not entries:
        raise ValueError(f"filtered.json is empty for {speaker}")

    out_dir = speaker_dir / "normalized"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale normalized files no longer in filtered.json
    expected = {e["wav"] for e in entries}
    for stale in out_dir.glob("*.wav"):
        if stale.name not in expected:
            stale.unlink()

    seg_dir = speaker_dir / "segments"

    for entry in tqdm(entries, desc="Stage 6: Normalization"):
        wav_name: str = entry["wav"]
        src = seg_dir / wav_name
        dst = out_dir / wav_name

        try:
            _normalize_wav(src, dst, target_rms_db, trim_silence_db)
        except Exception as exc:
            print(f"[WARN] Failed to normalize {wav_name}: {exc}")

    # Build output manifest from successfully normalized WAVs
    manifest: list[dict] = []
    for entry in entries:
        out_path = out_dir / entry["wav"]
        if out_path.exists():
            manifest.append(
                {
                    "wav": entry["wav"],
                    "text": entry.get("text", ""),
                    "duration": sf.info(str(out_path)).duration,
                    "confidence": entry.get("confidence"),
                    "snr_db": entry.get("snr_db"),
                }
            )

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _normalize_wav(src: Path, dst: Path, target_rms_db: float, trim_db: float) -> None:
    audio, sr = sf.read(str(src), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Trim leading/trailing silence
    audio = _trim_silence(audio, sr, trim_db)
    # RMS normalize
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 1e-10:
        target_rms = 10 ** (target_rms_db / 20.0)
        audio = audio * (target_rms / rms)
        audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(dst), audio, sr, subtype="PCM_16")


def _trim_silence(audio: np.ndarray, sr: int, threshold_db: float) -> np.ndarray:
    """Trim leading and trailing silence below threshold_db."""
    threshold_linear = 10 ** (threshold_db / 20.0)
    abs_audio = np.abs(audio)
    above = abs_audio > threshold_linear
    if not above.any():
        return audio  # all silence, return as-is
    first = int(np.argmax(above))
    last = int(len(above) - np.argmax(above[::-1]) - 1)
    return audio[first : last + 1]
