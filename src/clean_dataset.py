"""
clean_dataset.py

Removes corrupted video files from a dataset directory in-place.

Author: Sanele Hlabisa

python -m src.clean_dataset \
    --dataset_dir "datasets/abnormal_activities" \
    --dry_run
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Remove corrupted videos in-place.")
parser.add_argument("--dataset_dir", type=str, required=True)
parser.add_argument("--dry_run", action="store_true")

SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is not None:
        return
    print("❌ ffmpeg not found. Install with:")
    print("sudo apt install -y ffmpeg")
    sys.exit(1)


def is_valid_video(path: Path) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return result.returncode == 0


def main() -> None:
    args = parser.parse_args()
    check_ffmpeg()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"❌ Not found: {dataset_dir}")
        sys.exit(1)

    videos = [f for f in dataset_dir.glob("*/*") if f.suffix.lower() in SUPPORTED_EXTS]
    if not videos:
        print(f"⚠️  No videos found in {dataset_dir}")
        sys.exit(0)

    print(f"📂 Dataset : {dataset_dir}")
    print(f"🎬 Videos  : {len(videos)} found")
    if args.dry_run:
        print("🔍 Dry run — nothing will be deleted\n")

    kept = removed = 0
    for video in sorted(videos):
        if is_valid_video(video):
            print(f"  ✅ {video.relative_to(dataset_dir)}")
            kept += 1
        else:
            print(f"  ❌ CORRUPT → {video.relative_to(dataset_dir)}")
            if not args.dry_run:
                video.unlink()
            removed += 1

    print(
        f"\n  ✅ Kept: {kept}  🗑  Removed: {removed}{'  (dry run)' if args.dry_run else ''}"
    )


if __name__ == "__main__":
    main()
