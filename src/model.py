"""
model.py

ConvLSTM model for AHAR. CNN extracts spatial features, LSTM models temporal dynamics.

Author: Sanele Hlabisa

python -m src.model \
    --dataset_dir "datasets/abnormal_activities" \
    --model_dir "models"
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .dataset import AHARDataset


# ============================================================
# ConvLSTM2D Cell - replicates Keras ConvLSTM2D behaviour
# Input:  sequence (B, T, C, H, W)
# Output: last hidden state (B, filters, H, W)
# ============================================================


class ConvLSTM2DCell(nn.Module):
    """Single ConvLSTM2D cell. Processes one timestep."""

    def __init__(self, in_channels: int, filters: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2

        # Gates: input, forget, cell, output - all in one conv for efficiency
        # Input comes from x_t and h_{t-1} concatenated on channel dim
        self.conv = nn.Conv2d(
            in_channels + filters,
            filters * 4,  # i, f, g, o gates
            kernel_size=kernel_size,
            padding=pad,
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, C, H, W)
        h: torch.Tensor,  # (B, filters, H, W)
        c: torch.Tensor,  # (B, filters, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:

        combined = torch.cat([x, h], dim=1)  # (B, C+filters, H, W)
        gates: torch.Tensor = self.conv(combined)  # (B, filters*4, H, W)

        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM2D(nn.Module):
    """Runs ConvLSTM2DCell over a sequence, returns last hidden state."""

    def __init__(self, in_channels: int, filters: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.filters = filters
        self.cell = ConvLSTM2DCell(in_channels, filters, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        returns: (B, filters, H, W)  - last hidden state only
        """
        B, T, C, H, W = x.shape

        h = torch.zeros(B, self.filters, H, W, device=x.device)
        c = torch.zeros(B, self.filters, H, W, device=x.device)

        for t in range(T):
            h, c = self.cell(x[:, t], h, c)

        return h  # (B, filters, H, W)


# ============================================================
# Full model - matches paper pseudocode exactly
# ============================================================


class ConvLSTMModel(nn.Module):
    """
    Replicates the paper architecture:
      TimeDistributed(Conv2D(16))
      → ConvLSTM2D(64)
      → BatchNorm2D
      → Conv2D(16)
      → Dropout(0.5)
      → Flatten
      → Dense(256)
      → Dropout(0.5)
      → Dense(num_classes)

    Input:  (B, T, C, H, W)
    Output: (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        input_shape: tuple[int, int, int] = (3, 64, 64),
    ) -> None:
        super().__init__()

        C, H, W = input_shape

        # Step 1 - TimeDistributed Conv2D(16)
        # Applied per-frame - we handle time distribution via reshape in forward()
        self.td_conv = nn.Conv2d(C, 16, kernel_size=3, padding=1)

        # Step 2 - ConvLSTM2D(64)
        # Takes sequence of feature maps, outputs last hidden state
        self.convlstm = ConvLSTM2D(in_channels=16, filters=64, kernel_size=3)

        # Step 3 - BatchNorm on spatial output of ConvLSTM
        self.bn = nn.BatchNorm2d(64)

        # Step 4 - Conv2D(16) on final spatial map
        self.conv_post = nn.Conv2d(64, 16, kernel_size=3, padding=1)

        # Step 5 - Dropout
        self.dropout1 = nn.Dropout(0.5)

        # Step 6 - Flatten
        self.flatten = nn.Flatten()

        # Compute flattened size after conv_post

        flat_size = 16 * H * W

        # Step 7 - Dense(256)
        self.fc1 = nn.Linear(flat_size, 256)

        # Step 8 - Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        # Step 9 - Dense(num_classes)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape

        # Step 1 - TimeDistributed Conv2D: apply same conv to every frame
        x = x.view(B * T, C, H, W)
        x = F.relu(self.td_conv(x))  # (B*T, 16, H, W)
        x = x.view(B, T, 16, H, W)  # restore sequence

        # Step 2 - ConvLSTM2D: returns last hidden state
        x = self.convlstm(x)  # (B, 64, H, W)

        # Step 3 - BatchNorm
        x = self.bn(x)

        # Step 4 - Conv2D(16)
        x = F.relu(self.conv_post(x))  # (B, 16, H, W)

        # Step 5 - Dropout
        x = self.dropout1(x)

        # Step 6 - Flatten
        x = self.flatten(x)  # (B, 16*H*W)

        # Step 7 - Dense(256)
        x = F.relu(self.fc1(x))

        # Step 8 - Dropout(0.5)
        x = self.dropout2(x)

        # Step 9 - Dense(num_classes)
        return self.fc2(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", type=str, default="datasets/abnormal_activities"
    )
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--sequence_length", type=int, default=32)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path("outputs") / "model_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AHARDataset(
        args.dataset_dir, args.sequence_length, (args.height, args.width)
    )
    model = ConvLSTMModel(
        dataset.num_classes, input_shape=(3, args.height, args.width)
    ).to(device)

    if args.model_dir:
        from .utils import load_model

        model, _, epoch, loss = load_model(
            model, base_path=args.model_dir, map_location=device
        )
        print(f"📂 Loaded → epoch={epoch}, loss={loss:.4f}")
    else:
        print("⚠️  No model_dir - random weights (shape check only)")

    idx = random.randint(0, len(dataset) - 1)
    frames, true_label = dataset[idx]

    model.eval()
    with torch.no_grad():
        logits = model(frames.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)[0]
        pred_label = probs.argmax().item()
        pred_conf = probs[pred_label].item()

    true_name = dataset.class_names[true_label]
    pred_name = dataset.class_names[pred_label]

    print(f"\n✅ Input : {frames.shape}  Logits: {logits.shape}")
    print(f"🔹 True  : {true_name}")
    print(f"🔹 Pred  : {pred_name}  ({pred_conf:.1%})")
    print(f"🔹 Probs : {np.round(probs.cpu().numpy(), 2)}")

    stem = Path(dataset.samples[idx][0]).stem
    correct = "correct" if pred_label == true_label else "wrong"
    fname = f"{stem}_true-{true_name}_pred-{pred_name}_{correct}.mp4"
    clip = (frames * 255).byte().permute(0, 2, 3, 1).cpu()
    torchvision.io.write_video(str(out_dir / fname), clip, fps=args.fps)
    print(f"\n🎬 Saved → {out_dir / fname}")


if __name__ == "__main__":
    main()
