"""
dataset.py

Video and frames dataset classes for AHAR.

Author: Sanele Hlabisa

python -m src.dataset \
    --dataset_dir "datasets/processed/videos_abnormal_activities" \
    --frames \
    --num_samples 4
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import read_video_torchvision, write_video_torchvision, TARGET_FPS


class AHARDataset(Dataset):
    """
    Unified dataset class for loading video or pre-extracted frame datasets.
    """

    SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(
        self,
        dataset_dir: str | Path,
        sequence_length: int = 32,
        frame_size: tuple[int, int] = (64, 64),
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """
        Initializes the dataset and automatically detects the data format.

        Parameters:
            dataset_dir (str | Path): Path to the dataset directory.
            sequence_length (int): Number of frames to sample per clip.
            frame_size (tuple[int, int]): Target spatial resolution for the frames.
            transform (Optional[transforms.Compose]): Data augmentations to apply.

        Returns:
            None
        """
        self.dataset_dir = Path(dataset_dir)
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.transform = transform

        self.class_names: list[str] = sorted(
            d.name for d in self.dataset_dir.iterdir() if d.is_dir()
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        # Detect mode: frames (dirs of PNGs) or video
        self._mode = self._detect_mode()

        self.samples: list[tuple[Path, int]] = []
        for cls in self.class_names:
            cls_dir = self.dataset_dir / cls
            if self._mode == "frames":
                for clip_dir in sorted(cls_dir.iterdir()):
                    if clip_dir.is_dir() and any(clip_dir.glob("*.png")):
                        self.samples.append((clip_dir, self.class_to_idx[cls]))
            else:
                for f in sorted(cls_dir.iterdir()):
                    if f.suffix.lower() in self.SUPPORTED_EXTS:
                        self.samples.append((f, self.class_to_idx[cls]))

        print(
            f"✅ {len(self.samples)} {'clips' if self._mode == 'frames' else 'videos'} "
            f"| mode={self._mode} | {self.num_classes} classes: {self.class_names}"
        )

    def _detect_mode(self) -> str:
        """
        Checks the first class directory to determine if the dataset uses frames or videos.

        Parameters:
            None

        Returns:
            mode (str): The detected dataset format ('frames' or 'video').
        """
        for cls in self.class_names:
            cls_dir = self.dataset_dir / cls
            for item in cls_dir.iterdir():
                if item.is_dir() and any(item.glob("*.png")):
                    return "frames"
                if item.suffix.lower() in self.SUPPORTED_EXTS:
                    return "video"
        return "video"

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Parameters:
            None

        Returns:
            length (int): Total sample count.
        """
        return len(self.samples)

    def _sample_frames_tensor(
        self, frames: torch.Tensor, source_fps: float
    ) -> torch.Tensor:
        """
        Uniformly samples a fixed sequence of frames based on the target frames per second.

        Parameters:
            frames (torch.Tensor): The input frame sequence tensor.
            source_fps (float): The original frames per second of the video.

        Returns:
            sampled_frames (torch.Tensor): The reduced and padded frame sequence tensor.
        """
        T = frames.shape[0]
        stride = max(1, round(source_fps / TARGET_FPS))
        indices = list(range(0, T, stride))
        if len(indices) >= self.sequence_length:
            indices = indices[: self.sequence_length]
        else:
            indices += [indices[-1]] * (self.sequence_length - len(indices))
        return frames[torch.tensor(indices)]

    def _load_video(self, path: Path) -> torch.Tensor:
        """
        Loads a video file from disk and processes it into a normalized frame tensor.

        Parameters:
            path (Path): Path to the video file.

        Returns:
            video_tensor (torch.Tensor): Processed video tensor of shape (T, C, H, W).
        """
        video, fps = read_video_torchvision(path)  # (T, H, W, C) uint8
        video = self._sample_frames_tensor(video, fps)  # (T, H, W, C)
        video = video.permute(0, 3, 1, 2).float()  # (T, C, H, W)
        video = F.interpolate(
            video, size=self.frame_size, mode="bilinear", align_corners=False
        ).div(255.0)
        return video

    def _load_frames(self, clip_dir: Path) -> torch.Tensor:
        """
        Loads pre-extracted PNG frames from a directory into a normalized tensor.

        Parameters:
            clip_dir (Path): The directory containing the PNG frames and metadata.

        Returns:
            frames_tensor (torch.Tensor): Processed image tensor of shape (T, C, H, W).
        """
        # Read meta to get original fps
        meta_path = clip_dir / "meta.json"
        source_fps = TARGET_FPS
        if meta_path.exists():
            with open(meta_path) as f:
                source_fps = json.load(f).get("fps", TARGET_FPS)

        all_pngs = sorted(clip_dir.glob("*.png"))
        # Apply fps-based stride over available frames
        total = len(all_pngs)
        stride = max(1, round(source_fps / TARGET_FPS))
        indices = list(range(0, total, stride))
        if len(indices) >= self.sequence_length:
            indices = indices[: self.sequence_length]
        else:
            indices += [indices[-1]] * (self.sequence_length - len(indices))

        frames = []
        for i in indices:
            img = torchvision.io.read_image(str(all_pngs[i])).float().div(255.0)
            if tuple(img.shape[1:]) != tuple(self.frame_size):
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=self.frame_size,
                    mode="bilinear",
                    align_corners=False,
                )[0]
            frames.append(img)

        return torch.stack(frames)  # (T, C, H, W)

    def __getitem__(self, index: int):
        """
        Fetches and optionally transforms a single dataset sample by index.

        Parameters:
            index (int): The index of the sample to retrieve.

        Returns:
            sample (tuple): A tuple containing the processed video tensor and its label index.
        """
        path, label = self.samples[index]

        try:
            if self._mode == "frames":
                video = self._load_frames(path)
            else:
                video = self._load_video(path)
        except Exception:
            return self.__getitem__((index + 1) % len(self))

        if self.transform:
            seed = torch.randint(0, 1_000_000, (1,)).item()
            frames_out = []
            for frame in video:
                torch.manual_seed(seed)
                frames_out.append(self.transform(frame))
            video = torch.stack(frames_out)

        return video, label


class CachedAHARDataset(AHARDataset):
    """
    Dataset subclass that caches all processed clips in RAM to eliminate disk bottlenecks.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the dataset and pre-loads all samples into memory.

        Parameters:
            *args (Any): Positional arguments passed to the parent AHARDataset class.
            **kwargs (Any): Keyword arguments passed to the parent AHARDataset class.

        Returns:
            None
        """
        super().__init__(*args, **kwargs)
        print(f"📥 Caching {len(self.samples)} clips into RAM...")
        self._cache: list[tuple[torch.Tensor, int]] = []

        saved_transform = self.transform
        self.transform = None  # disable during caching

        for i in range(len(self.samples)):
            video, label = super().__getitem__(i)
            self._cache.append((video, label))
            if (i + 1) % 50 == 0 or (i + 1) == len(self.samples):
                print(f"   {i+1}/{len(self.samples)}")

        self.transform = saved_transform  # restore

        mem_gb = sum(v.nbytes for v, _ in self._cache) / 1e9
        print(f"✅ Cached {len(self._cache)} clips - ~{mem_gb:.2f} GB RAM")

    def __getitem__(self, index: int):
        """
        Fetches a single dataset sample from the RAM cache and applies runtime transforms.

        Parameters:
            index (int): The index of the sample to retrieve.

        Returns:
            sample (tuple): A tuple containing the augmented video tensor and its label index.
        """
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
        "--dataset_dir", type=str, default="datasets/raw/abnormal_activities"
    )
    parser.add_argument("--sequence_length", type=int, default=32)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--cache", action="store_true", help="Use CachedAHARDataset")
    args = parser.parse_args()

    out_dir = Path("outputs") / "dataset_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    DatasetClass = CachedAHARDataset if args.cache else AHARDataset
    dataset = DatasetClass(
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
