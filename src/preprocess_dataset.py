"""
preprocess_dataset.py

Preprocesses a raw video dataset into two faster-loading formats.
Handles flat structures and deeply nested structures (like Kinetics sub-directories).

Author: Sanele Hlabisa

python -m src.preprocess_dataset \
    --dataset_dir "datasets/raw/.kinetics" \
    --output_name "kinetics-dataset" \
    --frame_size 256
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .utils import TARGET_FPS

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_dir",
    type=str,
    required=True,
    help="Path to raw dataset e.g. datasets/raw/abnormal_activities or datasets/raw/.kinetics",
)
parser.add_argument(
    "--output_name",
    type=str,
    default=None,
    help="Override the output folder name (defaults to the dataset directory name)",
)
parser.add_argument(
    "--frame_size",
    type=int,
    default=256,
    help="Resize videos/frames to this square size (default 256)",
)
parser.add_argument(
    "--fix_only",
    action="store_true",
    help="Only fix corrupted videos in-place, skip processed output",
)
parser.add_argument("--dry_run", action="store_true")

SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def get_all_videos(dataset_dir: Path) -> list[Path]:
    """
    Recursively discovers all supported video files within the dataset directory.
    """
    return [
        v for v in sorted(dataset_dir.rglob("*"))
        if v.is_file() and v.suffix.lower() in SUPPORTED_EXTS
    ]


# Step 1 - Fix corrupted videos in-place
def fix_videos(videos: list[Path], dry_run: bool) -> None:
    """
    Fixes corrupted videos in-place using ffmpeg.
    """
    if shutil.which("ffmpeg") is None:
        print("❌ ffmpeg not found. Install: sudo apt install -y ffmpeg")
        sys.exit(1)

    if not videos:
        print("⚠️  No videos found to fix.")
        return

    print(f"🔧 Checking and fixing {len(videos)} videos...")
    fixed = failed = 0

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
            print(f"  ❌ Unrecoverable - {video.name}")
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
            fixed += 1

    print(
        f"\n  ✅ Fixed/Verified: {fixed}  ❌ Removed: {failed}{'  (dry run)' if dry_run else ''}"
    )


# Step 2 - Build processed video dataset (resized, original fps)
def make_video_dataset(videos: list[Path], video_out_dir: Path, frame_size: int) -> None:
    """
    Re-encodes and resizes all videos to a uniform square resolution while keeping original fps.
    """
    if shutil.which("ffmpeg") is None:
        print("❌ ffmpeg not found.")
        sys.exit(1)

    print(f"🎬 Building video dataset: {len(videos)} videos -> {video_out_dir}")

    for i, video in enumerate(videos):
        # The immediate parent directory string is ALWAYS the action class name
        cls = video.parent.name
        out_path = video_out_dir / cls / video.with_suffix(".mp4").name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(video),
                "-vf",
                f"scale={frame_size}:{frame_size}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-y",
                str(out_path),
            ],
            stderr=subprocess.PIPE,
        )

        if (i + 1) % 50 == 0 or (i + 1) == len(videos):
            print(f"   {i+1}/{len(videos)}")

    print(f"✅ Video dataset ready at {video_out_dir}")


# Step 3 - Build frames dataset from processed videos
def make_frames_dataset(video_dir: Path, frames_out_dir: Path) -> None:
    """
    Extracts and saves all individual frames from processed videos as PNG image files.
    """
    import torch
    import torchvision
    import json
    from torchvision.utils import save_image

    # Processed directories are flat (video_dir/class_name/video.mp4)
    videos = [
        v for v in sorted(video_dir.glob("*/*")) if v.suffix.lower() in SUPPORTED_EXTS
    ]
    print(f"🖼  Extracting frames: {len(videos)} videos -> {frames_out_dir}")

    for i, video_path in enumerate(videos):
        cls = video_path.parent.name
        clip_dir = frames_out_dir / cls / video_path.stem
        clip_dir.mkdir(parents=True, exist_ok=True)

        try:
            frames, _, info = torchvision.io.read_video(
                str(video_path), pts_unit="sec", output_format="TCHW"
            )
            source_fps = info.get("video_fps", 30.0)
        except Exception as e:
            print(f"  ⚠️  Skipped {video_path.name}: {e}")
            continue

        for j, frame in enumerate(frames):
            save_image(frame.float().div(255.0), str(clip_dir / f"{j:04d}.png"))

        torchvision.io.write_video(
            str(clip_dir / "clip.mp4"),
            frames.permute(0, 2, 3, 1).cpu(),
            fps=source_fps,
            video_codec="libx264",
        )

        with open(clip_dir / "meta.json", "w") as f:
            json.dump({"label": cls, "total_frames": len(frames), "fps": source_fps}, f)

        if (i + 1) % 50 == 0 or (i + 1) == len(videos):
            print(f"   {i+1}/{len(videos)}")

    print(f"✅ Frames dataset ready at {frames_out_dir}")


def main() -> None:
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    
    # Use explicit override name if given, else fall back to folder name
    dataset_name = args.output_name if args.output_name else dataset_dir.name

    if not dataset_dir.exists():
        print(f"❌ Not found: {dataset_dir}")
        sys.exit(1)

    # Recursively discover all video files across nested subfolders
    videos = get_all_videos(dataset_dir)

    # Fix raw videos in-place first
    fix_videos(videos, dry_run=args.dry_run)

    if args.fix_only or args.dry_run:
        return

    # Output dirs under datasets/processed/
    processed_root = dataset_dir.absolute().parent
    while processed_root.name and processed_root.name != "datasets":
        processed_root = processed_root.parent
    
    processed_root = processed_root / "processed"
    video_out_dir = processed_root / f"videos_{dataset_name}"
    frames_out_dir = processed_root / f"frames_{dataset_name}"

    print(f"\n📁 Target Output structure:")
    print(f"   {video_out_dir}")
    print(f"   {frames_out_dir}\n")

    # Refresh the list in case any files were unlinked during the fix phase
    videos = get_all_videos(dataset_dir)

    make_video_dataset(videos, video_out_dir, frame_size=args.frame_size)
    make_frames_dataset(video_out_dir, frames_out_dir)


if __name__ == "__main__":
    main()
