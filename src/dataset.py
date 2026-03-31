"""
Lazy-loading Dataset + Visualization utilities for ConvLSTM AHAR

Author: Sanele Hlabisa

python -m src.dataset \
    --dataset_dir "datasets/abnormal_activities" \
    --num_samples 4
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import random
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from .labels import LabelRegistry

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
        labels_path: Optional[Path] = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.sequence_length = sequence_length
        self.frame_size = frame_size

        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(frame_size),
        ])

        # ---- Registry: class names and indices always come from labels.json ----
        self.registry = LabelRegistry(labels_path) if labels_path else LabelRegistry()

        # These are ALWAYS the full multiclass set from labels.json
        # so num_classes is stable regardless of which dataset dir is used
        self.class_names: list[str] = self.registry.class_names
        self.class_to_idx: dict[str, int] = self.registry.class_to_idx
        self.num_classes: int = self.registry.num_classes

        # ---- Discover and validate dataset folders ----
        folder_names = sorted(
            d.name for d in self.dataset_dir.iterdir() if d.is_dir()
        )
        self.registry.validate_folders(folder_names)

        # ---- Detect dataset mode ----
        # "multiclass": dataset has multiple activity classes
        # "detection":  dataset has only 2 folders (normal / abnormal style)
        self.mode: str = self.registry.detect_mode(folder_names)
        print(f"📋 Dataset mode: {self.mode}")

        # ---- Index all videos ----
        # Folders resolve to canonical names → stable integer indices
        self.samples: list[tuple[Path, int]] = []
        skipped_folders: list[str] = []

        for fname in folder_names:
            canonical = self.registry.resolve(fname)
            if canonical is None:
                skipped_folders.append(fname)
                continue
            label_idx = self.class_to_idx[canonical]
            class_dir = self.dataset_dir / fname
            for file in sorted(class_dir.iterdir()):
                if file.suffix.lower() in self.SUPPORTED_EXTS:
                    self.samples.append((file, label_idx))

        if skipped_folders:
            print(f"⏭️  Skipped unresolved folders: {skipped_folders}")

        print(f"✅ Found {len(self.samples)} videos")
        print(f"🏷️  Model classes ({self.num_classes}): {self.class_names}")
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
    """
    Sanity check: load dataset, sample a few clips, save as MP4s.

    python -m src.dataset --dataset_dir "datasets/abnormal_activities" --num_samples 4
    """
    import argparse
    import torchvision
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Dataset sanity check — saves sample clips")
    parser.add_argument("--dataset_dir",     type=str, default="datasets/abnormal_activities")
    parser.add_argument("--sequence_length", type=int, default=8)
    parser.add_argument("--height",          type=int, default=224)
    parser.add_argument("--width",           type=int, default=224)
    parser.add_argument("--num_samples",     type=int, default=4)
    parser.add_argument("--fps",             type=int, default=8)
    args = parser.parse_args()

    out_dir = Path("outputs") / "dataset_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = AHARDataset(
        dataset_dir=args.dataset_dir,
        sequence_length=args.sequence_length,
        frame_size=(args.height, args.width),
    )

    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    print(f"\n🎬 Saving {len(indices)} sample clips → {out_dir}")

    for idx in indices:
        video, label = dataset[idx]                        # (T, C, H, W) float [0,1]

        class_name = dataset.class_names[label]

        # Recover original filename for traceability
        try:
            stem = Path(dataset.samples[idx][0]).stem
        except Exception:
            stem = f"sample_{idx:04d}"

        fname    = f"{stem}_class-{class_name}.mp4"
        out_path = out_dir / fname

        # torchvision.io.write_video expects (T, H, W, C) uint8
        clip_uint8 = (video * 255).byte().permute(0, 2, 3, 1).cpu()

        torchvision.io.write_video(str(out_path), clip_uint8, fps=args.fps)
        print(f"  ✅ {fname}")

    print(f"\n📁 Done — clips saved to {out_dir}")

if __name__ == "__main__":
    main()
