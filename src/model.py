"""
model.py

ConvLSTM model for video classification in the Abnormal Human Activity Recognition (AHAR) project.
Combines convolutional layers for spatial features and LSTM for temporal modeling.

Author: Sanele Hlabisa
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from src.dataset import AHARDataset
from src.utils import display_frames


class ConvLSTMModel(nn.Module):
    """
    CNN + LSTM model for video classification.
    Extracts spatial features with CNN, then models temporal relationships with LSTM.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        # --- Convolutional feature extractor ---
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(4)  # Downsample H, W by /4

        # Placeholder — will be set after seeing first forward
        self.lstm = None
        self.fc1 = None
        self.num_classes = num_classes
        self.dropout = nn.Dropout(0.5)

    def _init_lstm_fc(self, feature_dim: int) -> None:
        """Initialize LSTM and FC layers dynamically based on feature size."""
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=256, batch_first=True)
        self.fc1 = nn.Linear(256, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ConvLSTM network.

        Args:
            x: Input tensor of shape (batch, seq_len, channels, height, width)

        Returns:
            torch.Tensor: Model output logits (batch, num_classes)
        """
        batch_size, seq_len, C, H, W = x.size()
        #print(f"Input shape: {x.shape}  (B={batch_size}, T={seq_len}, C={C}, H={H}, W={W})")

        # Flatten sequence into batch dimension for CNN
        x = x.view(batch_size * seq_len, C, H, W)

        # --- CNN feature extraction ---
        x = F.relu(self.conv1(x))
        #print(f"After conv1: {x.shape}")  # -> (B*T, 16, H, W)

        x = F.relu(self.conv2(x))
        #print(f"After conv2: {x.shape}")  # -> (B*T, 64, H, W)

        x = self.pool(x)
        #print(f"After maxpool: {x.shape}")  # -> (B*T, 64, H/4, W/4)

        # --- Flatten spatial dimensions ---
        x = x.view(batch_size, seq_len, -1)
        #print(f"After flatten: {x.shape}")  # -> (B, T, F)

        # Initialize LSTM dynamically (only once)
        if self.lstm is None:
            feature_dim = x.size(-1)
            #print(f"Initializing LSTM with input_size={feature_dim}")
            self._init_lstm_fc(feature_dim)

        # --- Temporal modeling with LSTM ---
        x, _ = self.lstm(x)
        #print(f"After LSTM: {x.shape}")  # -> (B, T, hidden_size)

        x = x[:, -1, :]  # take last frame’s output
        #print(f"After last-frame select: {x.shape}")

        # --- Classification ---
        x = self.dropout(x)
        x = self.fc1(x)
        #print(f"Final output: {x.shape}")  # -> (B, num_classes)

        return x


def main() -> None:
    """
    Test model forward pass and visualize random video prediction.
    """
    print("📂 Loading dataset...")
    dataset: AHARDataset = AHARDataset()

    # Initialize model
    model = ConvLSTMModel(num_classes=dataset.num_classes)
    model.eval()

    # Pick a random video sample
    idx = random.randint(0, len(dataset.X) - 1)
    frames = dataset.X[idx]
    label_vec = dataset.Y[idx]
    label_name = dataset.class_names[label_vec.argmax()]

    # Convert frames (NumPy → Tensor) and stack
    seq_tensor = torch.stack([
        torch.tensor(f, dtype=torch.float32).permute(2, 0, 1) / 255.0  # normalize 0–1
        for f in frames
    ]).unsqueeze(0)  # Add batch dim

    print(f"\n🎞 Input tensor shape: {seq_tensor.shape}  "
          f"(batch={seq_tensor.size(0)}, seq={seq_tensor.size(1)}, "
          f"C={seq_tensor.size(2)}, H={seq_tensor.size(3)}, W={seq_tensor.size(4)})")

    # Forward pass
    with torch.no_grad():
        outputs = model(seq_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        pred_class = dataset.class_names[pred_idx]

    print(f"✅ Model output shape: {outputs.shape}")
    print(f"🔹 True class: {label_name}")
    print(f"🔹 Predicted class: {pred_class}")
    print(f"🔹 Probabilities: {probs[0].cpu().numpy()}")

    display_frames(frames, label=label_name, prediction=pred_class, probs=probs[0].cpu().numpy())


if __name__ == "__main__":
    main()
