from __future__ import annotations

import csv
import shutil
from pathlib import Path

from sesame.config import SesameConfig


def run(cfg: SesameConfig) -> None:
    from datasets import Dataset, Features, Value

    speaker = cfg.dataset.get("speaker", "Akinokoe")
    version = cfg.dataset.get("version", "B")
    dataset_root = Path(cfg.dataset.get("root", "../speakerforge/dataset"))

    dataset_dir = dataset_root / f"{speaker}_version{version}"
    metadata_csv = dataset_dir / "metadata.csv"
    wavs_dir = dataset_dir / "wavs"

    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"metadata.csv not found: {metadata_csv}. Run Phase 1 stage7 first."
        )

    # Read WAV files as raw bytes — bit-perfect copy of the source PCM_16 files.
    # Stored as {"bytes": ...} so datasets skips torchcodec encoding entirely.
    rows = []
    skipped = 0
    with open(metadata_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            wav_path = wavs_dir / f"{row['filename']}.wav"
            if not wav_path.exists():
                print(f"  [WARN] WAV not found, skipping: {wav_path.name}")
                skipped += 1
                continue
            rows.append({
                "audio": {"bytes": wav_path.read_bytes(), "path": wav_path.name},
                "text": row["text"],
            })

    if not rows:
        raise RuntimeError("No valid samples found. Check that stage7 wavs/ is populated.")
    if skipped:
        print(f"  [WARN] Skipped {skipped} missing WAVs.")

    # No features= here: avoids Audio.encode_example() which triggers torchcodec
    # on Windows. Colab's cast_column("audio", Audio(...)) handles the conversion.
    print(f"[stage1] Building embedded HF dataset from {len(rows)} samples...")
    ds = Dataset.from_list(rows)

    out_dir = Path("data") / f"{speaker}_v{version}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ds.save_to_disk(str(out_dir))

    size_mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1024**2
    print(f"[stage1] Saved {len(ds)} samples → {out_dir}  ({size_mb:.0f} MB)")
    print("[stage1] Audio embedded as WAV bytes — portable to Colab.")
