"""
stage3_4_cleanup.py — Sync transcripts/ to match segments/ after manual cleanup.

Run this after manually deleting unwanted segments (between stage3 and stage4)
to remove orphaned transcripts that no longer have a corresponding segment.

Usage:
    python stages/stage3_4_cleanup.py --speaker Akinokoe
    python stages/stage3_4_cleanup.py --speaker Akinokoe --dry-run
"""
from __future__ import annotations

import argparse
from pathlib import Path

PROC_DIR = Path("processed")


def sync(speaker: str, dry_run: bool = False) -> None:
    seg_dir = PROC_DIR / speaker / "segments"
    tr_dir  = PROC_DIR / speaker / "transcripts"

    if not seg_dir.exists():
        raise FileNotFoundError(f"Segments dir not found: {seg_dir}")
    if not tr_dir.exists():
        raise FileNotFoundError(f"Transcripts dir not found: {tr_dir}")

    segs = {p.stem for p in seg_dir.glob("*.wav")}
    trs  = {p.stem for p in tr_dir.glob("*.json")}

    orphans = sorted(trs - segs)

    print(f"Segments:               {len(segs)}")
    print(f"Transcripts (before):   {len(trs)}")
    print(f"Orphan transcripts:     {len(orphans)}")

    if not orphans:
        print("Already in sync.")
        return

    for stem in orphans:
        path = tr_dir / f"{stem}.json"
        if dry_run:
            print(f"  [dry-run] would delete: {path.name}")
        else:
            path.unlink()
            print(f"  Deleted: {path.name}")

    if not dry_run:
        remaining = sum(1 for _ in tr_dir.glob("*.json"))
        print(f"Transcripts (after):    {remaining}")
        print("Sync complete. Ready to run stage4.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove orphaned transcripts after manual segment cleanup (between stage3 and stage4)."
    )
    parser.add_argument("--speaker", required=True, help="Speaker name (e.g. Akinokoe)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    args = parser.parse_args()
    sync(args.speaker, dry_run=args.dry_run)
