from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from speakerforge.config import PipelineConfig

PROC_DIR = Path("processed")
DATASET_DIR = Path("dataset")


def run(pipeline_cfg: PipelineConfig, speaker: str, version: str = "B") -> None:
    """Package processed audio into an LJSpeech-compatible dataset."""
    _cfg = pipeline_cfg.stage7  # reserved for future use

    proc_dir = PROC_DIR
    dataset_dir = DATASET_DIR

    if version == "B":
        entries = _load_version_b_entries(proc_dir, speaker)
    elif version == "A":
        entries = _load_version_a_entries(proc_dir, speaker)
    else:
        raise ValueError(f"Unknown version '{version}'. Expected 'A' or 'B'.")

    out_dir = dataset_dir / f"{speaker}_version{version}"
    wavs_dir = out_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    expected_stems = set()
    for entry in tqdm(entries, desc="Stage 7: Packaging"):
        wav_name: str = entry["wav"]
        src_wav: Path

        if version == "B":
            src_wav = proc_dir / speaker / "normalized" / wav_name
        else:
            src_wav = proc_dir / speaker / "audio" / wav_name

        dst_wav = wavs_dir / wav_name
        shutil.copy2(str(src_wav), str(dst_wav))
        expected_stems.add(wav_name)

        stem = Path(wav_name).stem
        rows.append(
            {
                "filename": stem,
                "text": entry.get("text", ""),
                "duration": round(float(entry["duration"]), 3),
            }
        )

    # Remove stale wavs no longer in filtered.json
    for stale in wavs_dir.glob("*.wav"):
        if stale.name not in expected_stems:
            stale.unlink()

    df = pd.DataFrame(rows, columns=["filename", "text", "duration"])
    df.to_csv(out_dir / "metadata.csv", index=False, encoding="utf-8-sig")

    print(f"Packaged {len(rows)} segments → dataset/{speaker}_version{version}/")


def _load_version_b_entries(proc_dir: Path, speaker: str) -> list[dict]:
    """Load entries from normalized/manifest.json."""
    manifest = proc_dir / speaker / "normalized" / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"No normalized manifest for {speaker}. Run stage6 first.")
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if not entries:
        raise ValueError(f"Normalized manifest is empty for {speaker}.")
    return entries  # already has wav, text, duration keys


def _load_version_a_entries(proc_dir: Path, speaker: str) -> list[dict]:
    """Load entries by pairing audio/ WAVs with transcripts/ JSONs."""
    audio_dir = proc_dir / speaker / "audio"
    transcript_dir = proc_dir / speaker / "transcripts"
    wavs = sorted(audio_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No audio WAVs for {speaker}. Run stage1 first.")
    entries = []
    for wav in wavs:
        tj = transcript_dir / f"{wav.stem}.json"
        if not tj.exists():
            print(f"  [WARN] No transcript for {wav.stem}, skipping")
            continue
        data = json.loads(tj.read_text(encoding="utf-8"))
        dur = sf.info(str(wav)).duration
        entries.append({"wav": wav.name, "text": data.get("text", ""), "duration": round(dur, 3)})
    return entries
