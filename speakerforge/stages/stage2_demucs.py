from __future__ import annotations
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from speakerforge.config import PipelineConfig

PROC_DIR = Path("processed")


def run(pipeline_cfg: PipelineConfig, speaker: str) -> None:
    """Isolate vocals from audio using Demucs htdemucs_ft."""
    model_name = pipeline_cfg.stage2.get("model", "htdemucs_ft")
    in_dir = PROC_DIR / speaker / "audio"
    out_dir = PROC_DIR / speaker / "vocals"
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(in_dir.glob("BV*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files in {in_dir}. Run stage1 first.")

    model = _load_model(model_name)

    for wav in tqdm(wav_files, desc="Stage 2: vocal isolation"):
        out_wav = out_dir / wav.name
        if out_wav.exists():
            continue
        _separate(model, wav, out_wav)


def _load_model(model_name: str):
    from demucs.pretrained import get_model
    return get_model(model_name)


def _separate(model, input_wav: Path, output_wav: Path) -> None:
    import torch
    import torchaudio
    from demucs.apply import apply_model

    audio, sr = sf.read(str(input_wav), dtype="float32")
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    else:
        audio = audio.T

    wav_tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, C, T)

    model_sr = model.samplerate
    if sr != model_sr:
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, model_sr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        sources = apply_model(model, wav_tensor.to(device), device=device)

    stem_names = list(model.sources)
    vocals_idx = stem_names.index("vocals")
    vocals = sources[0, vocals_idx].cpu()  # (C, T)

    vocals_mono = vocals.mean(dim=0).numpy()
    if model_sr != 24000:
        vocals_tensor = torch.from_numpy(vocals_mono).unsqueeze(0)
        vocals_tensor = torchaudio.functional.resample(vocals_tensor, model_sr, 24000)
        vocals_mono = vocals_tensor.squeeze(0).numpy()

    sf.write(str(output_wav), vocals_mono, 24000, subtype="PCM_16")
