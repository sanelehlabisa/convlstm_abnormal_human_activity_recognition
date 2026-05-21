"""
preprocess_dataset.py

Two jobs in one script:
  1. Fix corrupted videos in-place (re-encode with ffmpeg)
  2. Pre-process dataset into frame images for fast loading

Author: Sanele Hlabisa

# Fix corrupted videos only
python -m src.preprocess_dataset --dataset_dir "datasets/abnormal_activities" --fix_only

# Fix + preprocess to frames
python -m src.preprocess_dataset \
    --dataset_dir "datasets/abnormal_activities" \
    --output_dir  "datasets/abnormal_activities_frames" \
    --sequence_length 32 \
    --fps 8
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .utils import save_frames_dataset, TARGET_FPS

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=True)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Where to save frame images. If omitted, skips frame extraction.",
)
parser.add_argument("--sequence_length", type=int, default=32)
parser.add_argument("--fps", type=int, default=TARGET_FPS)
parser.add_argument(
    "--fix_only",
    action="store_true",
    help="Only fix corrupted videos, skip frame extraction.",
)
parser.add_argument("--dry_run", action="store_true")

SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def fix_videos(dataset_dir: Path, dry_run: bool) -> None:
    """Re-encode all videos with ffmpeg to fix corruption in-place."""
    if shutil.which("ffmpeg") is None:
        print("❌ ffmpeg not found. Install: sudo apt install -y ffmpeg")
        sys.exit(1)

    videos = [
        v for v in sorted(dataset_dir.glob("*/*")) if v.suffix.lower() in SUPPORTED_EXTS
    ]
    if not videos:
        print(f"⚠️  No videos found in {dataset_dir}")
        return

    print(f"🔧 Fixing {len(videos)} videos in {dataset_dir}")
    kept = fixed = failed = 0

    for video in videos:
        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False, dir=video.parent
        ) as tmp:
            tmp_path = Path(tmp.name)

        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(video),
                "-map",
                "0:v:0",
                "-vsync",
                "0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-y",
                str(tmp_path),
            ],
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0 or tmp_path.stat().st_size < 1024:
            print(f"  ❌ Unrecoverable → {video.name}")
            tmp_path.unlink(missing_ok=True)
            if not dry_run:
                video.unlink()
            failed += 1
        else:
            clean_path = video.with_suffix(".mp4")
            if not dry_run:
                shutil.move(str(tmp_path), str(clean_path))
                if clean_path != video:
                    video.unlink(missing_ok=True)
            else:
                tmp_path.unlink(missing_ok=True)
            print(f"  ✅ Fixed → {clean_path.name}")
            fixed += 1

    print(
        f"\n  ✅ Fixed: {fixed}  ❌ Removed: {failed}  {'(dry run)' if dry_run else ''}"
    )


def main() -> None:
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)

    if not dataset_dir.exists():
        print(f"❌ Not found: {dataset_dir}")
        sys.exit(1)

    fix_videos(dataset_dir, dry_run=args.dry_run)

    if not args.fix_only and args.output_dir:
        save_frames_dataset(
            dataset_dir=dataset_dir,
            output_dir=Path(args.output_dir),
            sequence_length=args.sequence_length,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
