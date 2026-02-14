"""
Lazy-loading Dataset + Visualization utilities for ConvLSTM AHAR

Author: Sanele Hlabisa
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from .utils import display_video_grid

# ============================================================
# Dataset
# ============================================================

class AHARDataset(Dataset):
    """
    Lazy-loading dataset for Abnormal Human Activity Recognition (AHAR).

    Returns:
        video: Tensor (T, C, H, W)
        label: int
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

        # Default transform
        self.transform = transform or transforms.Compose(
            [
                transforms.ToTensor(),          # (H, W, C) → (C, H, W)
                transforms.Resize(frame_size),
            ]
        )

        # Discover classes
        self.class_names = sorted(
            d.name for d in self.dataset_dir.iterdir() if d.is_dir()
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        # Index all videos (lazy loading)
        self.samples: list[tuple[Path, int]] = []
        for class_name in self.class_names:
            class_dir = self.dataset_dir / class_name
            for file in class_dir.iterdir():
                if file.suffix.lower() in self.SUPPORTED_EXTS:
                    self.samples.append((file, self.class_to_idx[class_name]))

        print(f"✅ Found {len(self.samples)} videos across {self.num_classes} classes")
        print("⚡ Lazy-loading enabled")

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_frames(self, video: torch.Tensor) -> torch.Tensor:
        """
        Uniformly sample sequence_length frames.
        video shape: (T, H, W, C)
        """
        total_frames = video.shape[0]

        if total_frames >= self.sequence_length:
            idx = torch.linspace(0, total_frames - 1, self.sequence_length).long()
            video = video[idx]
        else:
            pad = self.sequence_length - total_frames
            video = torch.cat([video, video[-1:].repeat(pad, 1, 1, 1)], dim=0)

        return video

    def __getitem__(self, index: int):
        video_path, label = self.samples[index]
    
        try:
            video, _, _ = torchvision.io.read_video(
                str(video_path),
                pts_unit="sec",
            )  # (T, H, W, C) uint8
        except Exception:
            new_index = (index + 1) % len(self)
            return self[new_index]
    
        # ---- Temporal sampling first (cheap) ----
        video = self._sample_frames(video)  # (T, H, W, C)
    
        # ---- Convert to (T, C, H, W) float32 ----
        video = video.permute(0, 3, 1, 2)      # (T, C, H, W)
        video = video.float().div(255.0)       # normalize
    
        # ---- Resize ALL frames at once ----
        video = F.interpolate(
            video,
            size=self.frame_size,
            mode="bilinear",
            align_corners=False,
        )
    
        return video, label

# ============================================================
# Quick sanity test
# ============================================================

def main() -> None:
    dataset = AHARDataset(
        dataset_dir="dataset_clean",
        sequence_length=8,
        frame_size=(224, 224),
    )

    idx = torch.randint(0, len(dataset), (1,)).item()
    video, label = dataset[idx]

    print("Video shape:", video.shape)      # (T, C, H, W)
    print("Label:", label)
    print("Pixel range:", video.min().item(), video.max().item())

    display_video_grid(
        video,
        class_names=dataset.class_names,
        true_label=label,
        pred_label=None,  # can pass model output later
        max_cols=4,
        frame_size=3,
        show=False,
        save_path="output.png",
    )


if __name__ == "__main__":
    main()
