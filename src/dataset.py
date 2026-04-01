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
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


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

    def _sample_frames(self, video: torch.Tensor) -> torch.Tensor:
        """Uniformly sample sequence_length frames from (T, H, W, C)."""
        T = video.shape[0]
        if T >= self.sequence_length:
            idx = torch.linspace(0, T - 1, self.sequence_length).long()
            return video[idx]
        pad = self.sequence_length - T
        return torch.cat([video, video[-1:].repeat(pad, 1, 1, 1)], dim=0)

    def __getitem__(self, index: int):
        video_path, label = self.samples[index]
        try:
            video, _, _ = torchvision.io.read_video(str(video_path), pts_unit="sec")
        except Exception:
            return self[(index + 1) % len(self)]

        video = self._sample_frames(video)  # (T, H, W, C)
        video = video.permute(0, 3, 1, 2).float().div(255.0)  # (T, C, H, W)
        video = F.interpolate(
            video, size=self.frame_size, mode="bilinear", align_corners=False
        )
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
        clip = (video * 255).byte().permute(0, 2, 3, 1).cpu()
        torchvision.io.write_video(str(out_dir / fname), clip, fps=args.fps)
        print(f"  ✅ {fname}")


if __name__ == "__main__":
    main()
