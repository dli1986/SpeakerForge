from __future__ import annotations

import json
import logging
from pathlib import Path

from speakerforge.config import PipelineConfig

PROC_DIR = Path("processed")

logger = logging.getLogger(__name__)


def run(pipeline_cfg: PipelineConfig, speaker: str) -> None:
    """Filter segments by duration, transcript confidence, and SNR."""
    cfg = pipeline_cfg.stage5
    min_duration: float = cfg.get("min_duration", 1.5)
    max_duration: float = cfg.get("max_duration", 15.0)
    min_confidence: float = cfg.get("min_confidence", 0.6)
    min_snr_db: float = cfg.get("min_snr_db", 15.0)

    seg_dir = PROC_DIR / speaker / "segments"
    tr_dir = PROC_DIR / speaker / "transcripts"
    out_path = PROC_DIR / speaker / "filtered.json"

    wav_files = sorted(seg_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files in {seg_dir}. Run stage3 first.")

    manifest: list[dict] = []
    total = len(wav_files)

    for wav_path in wav_files:
        stem = wav_path.stem

        # --- duration check ---
        import soundfile as sf
        duration = sf.info(str(wav_path)).duration
        if not (min_duration <= duration <= max_duration):
            continue

        # --- transcript / confidence check ---
        tr_path = tr_dir / f"{stem}.json"
        if not tr_path.exists():
            print(f"[WARN] No transcript for {stem}, skipping")
            continue

        transcript = json.loads(tr_path.read_text(encoding="utf-8"))
        text = transcript.get("text", "")
        words = transcript.get("words", [])
        if words:
            scores = [(w.get("score") or 0.0) for w in words]
            confidence = sum(scores) / len(scores)
        else:
            confidence = 0.0

        if confidence < min_confidence:
            continue

        # --- SNR check ---
        snr_db = _compute_snr(wav_path)
        if snr_db < min_snr_db:
            continue

        manifest.append(
            {
                "wav": wav_path.name,
                "text": text,
                "duration": round(duration, 4),
                "confidence": round(confidence, 4),
                "snr_db": round(snr_db, 4),
            }
        )

    out_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    kept = len(manifest)
    print(f"Filtered: {kept} / {total} segments kept")


def _compute_snr(wav_path: Path) -> float:
    """Estimate SNR in dB using RMS of signal vs estimated noise floor."""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    rms_signal = np.sqrt(np.mean(audio ** 2))
    # Estimate noise from quietest 10% of frames (20ms frames)
    frame_len = int(0.02 * sr)
    if frame_len == 0 or len(audio) < frame_len:
        return 0.0
    frames = [audio[i : i + frame_len] for i in range(0, len(audio) - frame_len, frame_len)]
    frame_rms = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])
    noise_floor = np.percentile(frame_rms, 10)
    if noise_floor < 1e-10:
        return 60.0  # near-silent noise floor = very high SNR
    return float(20 * np.log10(rms_signal / noise_floor))
