from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from speakerforge.config import PipelineConfig

PROC_DIR = Path("processed")


def run(pipeline_cfg: PipelineConfig, speaker: str) -> None:
    """Transcribe and force-align speech segments using WhisperX."""
    cfg = pipeline_cfg.stage4
    model_name = cfg.get("model", "large-v3")
    language = cfg.get("language", "zh")

    device = cfg.get("device", _default_device())
    compute_type = cfg.get("compute_type", "float16")

    seg_dir = PROC_DIR / speaker / "segments"
    out_dir = PROC_DIR / speaker / "transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(seg_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No segments in {seg_dir}. Run stage3 first.")

    model, align_model, align_metadata = _load_whisperx_model(
        model_name, device, compute_type, language
    )

    for wav_path in tqdm(wav_files, desc="Stage 4: WhisperX alignment"):
        _transcribe_segment(
            wav_path, out_dir, model, align_model, align_metadata, device, language
        )


def _default_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_whisperx_model(model_name: str, device: str, compute_type: str, language: str):
    """Load WhisperX transcription and alignment models."""
    import whisperx
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)
    align_model, align_metadata = whisperx.load_align_model(
        language_code=language, device=device
    )
    return model, align_model, align_metadata


def _transcribe_segment(
    wav_path: Path,
    out_dir: Path,
    model,
    align_model,
    align_metadata,
    audio_device: str,
    language: str,
) -> None:
    """Transcribe and align a single WAV segment, writing JSON output."""
    out_path = out_dir / f"{wav_path.stem}.json"

    if out_path.exists():
        print(f"  Skip (exists): {wav_path.stem}")
        return

    try:
        import whisperx

        audio = whisperx.load_audio(str(wav_path))
        result = model.transcribe(audio, batch_size=16, language=language)
        result_aligned = whisperx.align(
            result["segments"], align_model, align_metadata, audio, audio_device
        )

        full_text = " ".join(
            seg.get("text", "").strip() for seg in result_aligned["segments"]
        ).strip()

        words = []
        for seg in result_aligned["segments"]:
            for w in seg.get("words", []):
                words.append(
                    {
                        "word": w.get("word", ""),
                        "start": w.get("start"),
                        "end": w.get("end"),
                        "score": w.get("score"),
                    }
                )

        transcript = {
            "text": full_text,
            "language": language,
            "words": words,
        }

        out_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")

    except Exception as exc:
        print(f"  [WARN] Failed to transcribe {wav_path.stem}: {exc}")
