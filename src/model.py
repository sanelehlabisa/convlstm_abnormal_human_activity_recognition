"""
model.py

ConvLSTM model for video classification in the Abnormal Human Activity Recognition (AHAR) project.
CNN extracts spatial features per frame, LSTM models temporal dynamics.

Author: Sanele Hlabisa
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .dataset import AHARDataset
from .utils import display_video_grid


# ============================================================
# Model
# ============================================================

class ConvLSTMModel(nn.Module):
    """
    CNN + LSTM model for video classification.
    Input shape: (B, T, C, H, W)
    Output shape: (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        dropout: float = 0.5,
        input_shape: tuple[int, int, int] = (3, 224, 224),
    ) -> None:
        super().__init__()

        # -------- CNN backbone --------
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(4)

        # -------- Infer CNN feature size safely --------
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy = self._cnn_forward(dummy)
            self.feature_dim = dummy.view(1, -1).size(1)

        # -------- LSTM + classifier --------
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def _cnn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """CNN forward for a single frame batch."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (B, T, C, H, W)
        Returns:
            logits: Tensor (B, num_classes)
        """
        B, T, C, H, W = x.shape

        # Merge batch & time for CNN
        x = x.view(B * T, C, H, W)
        x = self._cnn_forward(x)

        # Restore temporal dimension
        x = x.view(B, T, -1)

        # Temporal modeling
        x, _ = self.lstm(x)

        # Last timestep
        x = x[:, -1, :]
        x = self.dropout(x)

        return self.fc(x)


# ============================================================
# Sanity test / visualization
# ============================================================

def main() -> None:
    """
    Sanity check:
    - Load dataset
    - Run single forward pass
    - Visualize frames with prediction
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥 Using device: {device}")

    print("📂 Loading dataset...")
    dataset = AHARDataset(
        dataset_dir="dataset",
        sequence_length=10,
        frame_size=(224, 224),
    )

    # Random sample
    idx = random.randint(0, len(dataset) - 1)
    frames, true_label = dataset[idx]          # (T, C, H, W)
    frames = frames.to(device)

    # Model
    model = ConvLSTMModel(
        num_classes=dataset.num_classes,
        input_shape=frames.shape[1:],  # (C, H, W)
    ).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(frames.unsqueeze(0))    # (1, T, C, H, W)
        probs = torch.softmax(logits, dim=1)
        pred_label = probs.argmax(dim=1).item()

    print(f"✅ Input shape: {frames.shape}")
    print(f"✅ Logits shape: {logits.shape}")
    print(f"🔹 True class: {dataset.class_names[true_label]}")
    print(f"🔹 Predicted class: {dataset.class_names[pred_label]}")
    print(f"🔹 Probabilities: {probs[0].cpu().numpy()}")

    # Visualization
    display_video_grid(
        video=frames.cpu(),
        class_names=dataset.class_names,
        true_label=true_label,
        pred_label=pred_label,
        save_path="sample_prediction.png",
    )


if __name__ == "__main__":
    main()
