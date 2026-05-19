"""
dataset.py

Lazy-loading dataset for AHAR. Class names come from subdirectory names.

Author: Sanele Hlabisa

python -m src.dataset \
    --dataset_dir "datasets/abnormal_activities" \
    --num_samples 4
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import read_video_torchvision, write_video_torchvision

TARGET_FPS: int = 8


class AHARDataset(Dataset):
    """
    Lazy-loading video dataset. Class names are taken from subdirectory names.

    Returns: video (T, C, H, W), label (int)
    """

    SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(
        self,
        dataset_dir: str | Path,
        sequence_length: int = 10,
        frame_size: tuple[int, int] = (224, 224),
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.transform = transform
        self.TARGET_FPS: int = 8

        # Class names from subdirectory names, sorted for stable indices
        self.class_names: list[str] = sorted(
            d.name for d in self.dataset_dir.iterdir() if d.is_dir()
        )
        self.class_to_idx: dict[str, int] = {
            c: i for i, c in enumerate(self.class_names)
        }
        self.num_classes: int = len(self.class_names)

        # Index all videos
        self.samples: list[tuple[Path, int]] = []
        for class_name in self.class_names:
            class_dir = self.dataset_dir / class_name
            for file in sorted(class_dir.iterdir()):
                if file.suffix.lower() in self.SUPPORTED_EXTS:
                    self.samples.append((file, self.class_to_idx[class_name]))

        print(
            f"✅ {len(self.samples)} videos | {self.num_classes} classes: {self.class_names}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_frames(self, video: torch.Tensor, source_fps: float) -> torch.Tensor:
        """
        Sample sequence_length frames at TARGET_FPS using fixed stride.
        Pads by repeating last frame (preserves temporal context, not black pixels).
        video: (T, H, W, C) uint8
        """
        T = video.shape[0]
        stride = max(1, round(source_fps / TARGET_FPS))
        indices = list(range(0, T, stride))

        if len(indices) >= self.sequence_length:
            indices = indices[: self.sequence_length]
        else:
            indices += [indices[-1]] * (self.sequence_length - len(indices))

        return video[torch.tensor(indices)]  # (sequence_length, H, W, C)

    def __getitem__(self, index: int):
        video_path, label = self.samples[index]
        try:
            video, fps = read_video_torchvision(video_path)  # (T, H, W, C) uint8
        except Exception:
            return self.__getitem__((index + 1) % len(self))

        video = self._sample_frames(video, fps)  # (T, H, W, C)

        # Permute and resize in one step - keep uint8 until after resize to save memory
        video = video.permute(0, 3, 1, 2)  # (T, C, H, W)
        video = F.interpolate(
            video.float(), size=self.frame_size, mode="bilinear", align_corners=False
        )  # (T, C, H, W) float

        # Normalise before transform so transform receives [0,1] float
        video = video.div(255.0)

        if self.transform:
            seed = torch.randint(0, 1_000_000, (1,)).item()
            frames = []
            for frame in video:
                torch.manual_seed(seed)
                frames.append(self.transform(frame))
            video = torch.stack(frames)

        return video, label


class CachedAHARDataset(AHARDataset):
    """
    Decodes all videos into RAM on init. After that __getitem__ is instant.
    Use for small datasets (≤2000 clips) where RAM allows.
    350 clips × 32 frames × 64×64 ≈ 1.7 GB - fine for Colab A100 (83 GB).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        print(f"📥 Caching {len(self.samples)} videos into RAM...")
        self._cache: list[tuple[torch.Tensor, int]] = []
        for i in range(len(self.samples)):
            # Call parent __getitem__ which does decode + resize + normalise
            # Store the result WITHOUT transform - transform applied per-access
            video_path, label = self.samples[i]
            try:
                video, fps = read_video_torchvision(video_path)
            except Exception:
                video, fps = read_video_torchvision(
                    self.samples[(i + 1) % len(self.samples)][0]
                )
                label = self.samples[(i + 1) % len(self.samples)][1]

            video = self._sample_frames(video, fps)
            video = video.permute(0, 3, 1, 2)
            video = F.interpolate(
                video.float(),
                size=self.frame_size,
                mode="bilinear",
                align_corners=False,
            ).div(255.0)
            self._cache.append((video, label))

            if (i + 1) % 50 == 0 or (i + 1) == len(self.samples):
                print(f"   {i+1}/{len(self.samples)}")

        mem_gb = sum(v.nbytes for v, _ in self._cache) / 1e9
        print(f"✅ Cached {len(self._cache)} clips - ~{mem_gb:.2f} GB RAM used")

    def __getitem__(self, index: int):
        video, label = self._cache[index]
        if self.transform:
            seed = torch.randint(0, 1_000_000, (1,)).item()
            frames = []
            for frame in video:
                torch.manual_seed(seed)
                frames.append(self.transform(frame))
            video = torch.stack(frames)
        return video, label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", type=str, default="datasets/abnormal_activities"
    )
    parser.add_argument("--sequence_length", type=int, default=8)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path("outputs") / "dataset_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = AHARDataset(
        args.dataset_dir, args.sequence_length, (args.height, args.width)
    )
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    print(f"\n🎬 Saving {len(indices)} clips → {out_dir}")
    for idx in indices:
        video, label = dataset[idx]
        stem = Path(dataset.samples[idx][0]).stem
        fname = f"{stem}_class-{dataset.class_names[label]}.mp4"
        write_video_torchvision(video, out_dir / fname, fps=args.fps)
        print(f"  ✅ {fname}")


if __name__ == "__main__":
    main()
