"""
preprocess_dataset.py

Preprocesses a raw video dataset into two faster-loading formats:

  datasets/
    raw/
      abnormal_activities/         - original untouched
    processed/
      videos_abnormal_activities/  - re-encoded, resized to 256x256, original fps
      frames_abnormal_activities/  - PNG frames extracted from processed videos

Author: Sanele Hlabisa

# Fix corrupted videos in raw/ only
python -m src.preprocess_dataset \
    --dataset_dir "datasets/raw/abnormal_activities" \
    --fix_only

# Full pipeline: fix + make processed video + frames datasets
python -m src.preprocess_dataset \
    --dataset_dir "datasets/raw/abnormal_activities" \
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
    help="Path to raw dataset e.g. datasets/raw/abnormal_activities",
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


# Step 1 - Fix corrupted videos in-place
def fix_videos(dataset_dir: Path, dry_run: bool) -> None:
    """
    Fixes corrupted videos in-place using ffmpeg.

    Parameters:
        dataset_dir (Path): Path to the raw dataset directory containing the videos.
        dry_run (bool): Flag determining whether to execute or just simulate the fixes.

    Returns:
        None: This function does not return any value.
    """
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
            print(f"  ✅ Fixed - {clean_path.name}")
            fixed += 1

    print(
        f"\n  ✅ Fixed: {fixed}  ❌ Removed: {failed}{'  (dry run)' if dry_run else ''}"
    )


# Step 2 - Build processed video dataset (resized, original fps)
def make_video_dataset(raw_dir: Path, video_out_dir: Path, frame_size: int) -> None:
    """
    Re-encodes and resizes all videos to a uniform square resolution while keeping original fps.

    Parameters:
        raw_dir (Path): Path to the directory containing the original raw videos.
        video_out_dir (Path): Path to the destination directory for the processed videos.
        frame_size (int): The target width and height to resize the videos.

    Returns:
        None: This function does not return any value.
    """
    if shutil.which("ffmpeg") is None:
        print("❌ ffmpeg not found.")
        sys.exit(1)

    videos = [
        v for v in sorted(raw_dir.glob("*/*")) if v.suffix.lower() in SUPPORTED_EXTS
    ]
    print(f"🎬 Building video dataset: {len(videos)} videos - {video_out_dir}")

    for i, video in enumerate(videos):
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

    Parameters:
        video_dir (Path): Path to the directory containing the resized processed videos.
        frames_out_dir (Path): Path to the destination directory for the extracted frame images.

    Returns:
        None: This function does not return any value.
    """
    import torch
    import torchvision
    import json
    from torchvision.utils import save_image

    videos = [
        v for v in sorted(video_dir.glob("*/*")) if v.suffix.lower() in SUPPORTED_EXTS
    ]
    print(f"🖼  Extracting frames: {len(videos)} videos - {frames_out_dir}")

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

        # Save every frame - no downsampling here, dataset class handles sequence_length
        for j, frame in enumerate(frames):
            save_image(frame.float().div(255.0), str(clip_dir / f"{j:04d}.png"))

        # Save clip video for inspection
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


# ============================================================
# Main
# ============================================================


def main() -> None:
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    dataset_name = dataset_dir.name

    if not dataset_dir.exists():
        print(f"❌ Not found: {dataset_dir}")
        sys.exit(1)

    # Fix raw videos in-place first
    fix_videos(dataset_dir, dry_run=args.dry_run)

    if args.fix_only or args.dry_run:
        return

    # Output dirs under datasets/processed/
    processed_root = dataset_dir.parent.parent / "processed"
    video_out_dir = processed_root / f"videos_{dataset_name}"
    frames_out_dir = processed_root / f"frames_{dataset_name}"

    print(f"\n📁 Output structure:")
    print(f"   {video_out_dir}")
    print(f"   {frames_out_dir}\n")

    make_video_dataset(dataset_dir, video_out_dir, frame_size=args.frame_size)
    make_frames_dataset(video_out_dir, frames_out_dir)


if __name__ == "__main__":
    main()
