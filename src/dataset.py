"""
dataset.py

Dataset loader for the ConvLSTM-based Abnormal Human Activity Recognition (AHAR) project.
Automatically detects class folders, loads videos, applies augmentations, and
stores data in memory for model training or visualization.

Author: Sanele Hlabisa
"""

from __future__ import annotations
import os
import random
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from typing import Any

from utils import (
    WIDTH,
    HEIGHT,
    DATASET_SPLIT,
    display_frames,
)


class AHARDataset:
    """
    Dataset class for Abnormal Human Activity Recognition.

    Reads dataset folders structured as:
        dataset/
            walking/
                video1.avi
                video2.avi
            running/
                video1.avi
                ...

    Attributes:
        input_dir (str): Root directory of dataset.
        class_names (list[str]): Names of the classes (from subfolder names).
        num_classes (int): Total number of classes.
        X (list[list[np.ndarray]]): Loaded video frame sequences.
        Y (list[np.ndarray]): Corresponding one-hot encoded labels.
    """

    def __init__(self, input_dir: str, augmentations: Any | None = None) -> None:
        self.input_dir = input_dir
        print(os.listdir(input_dir))
        self.class_names = sorted(
            [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        )
        self.num_classes = len(self.class_names)
        self.augmentations = augmentations or self._default_augmentations()

        self.X: list[list[np.ndarray]] = []
        self.Y: list[np.ndarray] = []

        self._load_dataset()

    # ------------------------
    # Internal helpers
    # ------------------------
    def _default_augmentations(self) -> transforms.Compose:
        """Return default augmentations for frames."""
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.Resize((HEIGHT, WIDTH)),
            ]
        )

    def _extract_frames(self, video_path: str) -> list[np.ndarray]:
        """Extract frames from a video file."""
        frames: list[np.ndarray] = []
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return frames

    def _apply_augmentations(self, frame: np.ndarray) -> np.ndarray:
        """Apply augmentations to a single frame."""
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        augmented = self.augmentations(pil_img)
        return np.array(augmented)

    def _load_dataset(self) -> None:
        """Load all videos and their corresponding labels into memory."""

        for i, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.input_dir, class_name)
            video_files = [f for f in os.listdir(class_dir) if f.endswith((".avi", ".mp4"))]

            for video_file in video_files:
                video_path = os.path.join(class_dir, video_file)
                frames = self._extract_frames(video_path)
                if not frames:
                    continue

                augmented_frames = [self._apply_augmentations(f) for f in frames]
                label = np.zeros(self.num_classes)
                label[i] = 1

                self.X.append(augmented_frames)
                self.Y.append(label)

        print(f"✅ Loaded {len(self.X)} videos across {self.num_classes} classes.")


# ------------------------
# Main entry point
# ------------------------
def main() -> None:
    """
    Test dataset loading and visualization.
    Randomly selects 3–5 samples and displays them using display_frames().
    """
    dataset_path = "../dataset"
    dataset = AHARDataset(dataset_path)

    num_samples = random.randint(3, 5)
    print(f"[INFO] Displaying {num_samples} random samples...")

    for _ in range(num_samples):
        idx = random.randint(0, len(dataset.X) - 1)
        frames = dataset.X[idx]
        label = dataset.Y[idx]
        label_name = dataset.class_names[np.argmax(label)]

        display_frames(frames[:15], label=label_name)  # Show first few frames


if __name__ == "__main__":
    main()
