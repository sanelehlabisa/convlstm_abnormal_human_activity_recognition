"""
clean_dataset.py

Fix corrupted videos in the dataset by re-encoding with ffmpeg. This ensures all videos are decodable and consistent.

Author: Sanele Hlabisa

python -m src.clean_dataset \
    --dataset_dir "datasets/abnormal_activities"
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import shutil
import tempfile
import argparse

parser = argparse.ArgumentParser(description="Fix corrupted dataset videos by re-encoding with ffmpeg")

parser.add_argument("--dataset_dir", type=str, default="datasets/abnormal_activities")

args: argparse.Namespace = parser.parse_args()

dataset_dir = Path(args.dataset_dir)

SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

def main():
    kept = fixed = failed = 0

    for video in sorted(dataset_dir.glob("*/*")):
        if video.suffix.lower() not in SUPPORTED_EXTS:
            continue

        # Re-encode to a temp file first — never touch original until success
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=video.parent) as tmp:
            tmp_path = Path(tmp.name)

        cmd = [
            "ffmpeg",
            "-v", "error",
            "-i", str(video),
            "-map", "0:v:0",
            "-vsync", "0",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y",
            str(tmp_path),
        ]

        result = subprocess.run(cmd, stderr=subprocess.PIPE)

        if result.returncode != 0 or tmp_path.stat().st_size < 1024:
            # Truly unrecoverable — remove original and temp
            print(f"  ❌ Unrecoverable → {video.name}")
            tmp_path.unlink(missing_ok=True)
            video.unlink()
            failed += 1
        else:
            # Always save as .mp4 since we encoded with libx264
            clean_path = video.with_suffix(".mp4")
            shutil.move(str(tmp_path), str(clean_path))
            if clean_path != video:
                video.unlink(missing_ok=True)   # remove original .avi if we renamed to .mp4
            print(f"  ✅ Fixed in-place → {clean_path.name}")
            fixed += 1

    print(f"\n  ✅ Fixed : {fixed}")
    print(f"  ❌ Removed (unrecoverable): {failed}")
    print(f"  📁 Dataset dir unchanged: {dataset_dir}")

if __name__ == "__main__":
    main()
