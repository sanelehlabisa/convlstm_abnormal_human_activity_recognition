"""
model.py

ConvLSTM model for AHAR. CNN extracts spatial features, LSTM models temporal dynamics.

Author: Sanele Hlabisa

python -m src.model \
    --dataset_dir "datasets/processed/videos_abnormal_activities" \
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


class ConvLSTM2DCell(nn.Module):
    """
    Implements a single ConvLSTM2D cell for processing one timestep.
    """

    def __init__(self, in_channels: int, filters: int, kernel_size: int = 3) -> None:
        """
        Initializes the ConvLSTM2D cell with combined gate convolutions.

        Parameters:
            in_channels (int): Number of channels in the input tensor.
            filters (int): Number of output filters for the hidden state.
            kernel_size (int): Size of the convolutional kernel.

        Returns:
            None
        """
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
        """
        Computes the next hidden and cell states for the current timestep.

        Parameters:
            x (torch.Tensor): Input tensor for the current timestep of shape (B, C, H, W).
            h (torch.Tensor): Previous hidden state tensor of shape (B, filters, H, W).
            c (torch.Tensor): Previous cell state tensor of shape (B, filters, H, W).

        Returns:
            states (tuple[torch.Tensor, torch.Tensor]): A tuple containing the new hidden state and new cell state.
        """

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
    """
    Applies a ConvLSTM2D cell sequentially over an entire temporal dimension.
    """

    def __init__(self, in_channels: int, filters: int, kernel_size: int = 3) -> None:
        """
        Initializes the sequential ConvLSTM module.

        Parameters:
            in_channels (int): Number of channels in the input frames.
            filters (int): Number of output filters for the hidden state.
            kernel_size (int): Size of the convolutional kernel.

        Returns:
            None
        """
        super().__init__()
        self.filters = filters
        self.cell = ConvLSTM2DCell(in_channels, filters, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the sequence and returns only the final hidden state.

        Parameters:
            x (torch.Tensor): Input sequence tensor of shape (B, T, C, H, W).

        Returns:
            h (torch.Tensor): The final hidden state tensor of shape (B, filters, H, W).
        """
        B, T, C, H, W = x.shape

        h = torch.zeros(B, self.filters, H, W, device=x.device)
        c = torch.zeros(B, self.filters, H, W, device=x.device)

        for t in range(T):
            h, c = self.cell(x[:, t], h, c)

        return h  # (B, filters, H, W)


# ============================================================
# Models
# ============================================================


class ConvLSTMOriginal(nn.Module):
    """
    Original heavy ConvLSTM architecture for abnormal activity recognition.
    """

    def __init__(self, num_classes: int, input_shape: tuple = (3, 64, 64)) -> None:
        """
        Initializes the large baseline model and prints its parameter count.

        Parameters:
            num_classes (int): Number of output classes for prediction.
            input_shape (tuple[int, int, int]): Shape of the single input frame (C, H, W).

        Returns:
            None
        """
        super().__init__()
        C, H, W = input_shape
        self.td_conv = nn.Conv2d(C, 16, 3, padding=1)
        self.convlstm = ConvLSTM2D(16, 64)
        self.bn = nn.BatchNorm2d(64)
        self.conv_post = nn.Conv2d(64, 16, 3, padding=1)
        self.dropout1 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * H * W, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass returning raw class logits.

        Parameters:
            x (torch.Tensor): Input video tensor of shape (B, T, C, H, W).

        Returns:
            logits (torch.Tensor): Unnormalized class prediction logits of shape (B, num_classes).
        """
        B, T, C, H, W = x.shape
        x = F.relu(self.td_conv(x.view(B * T, C, H, W))).view(B, T, 16, H, W)
        x = self.bn(self.convlstm(x))
        x = self.dropout1(F.relu(self.conv_post(x)))
        return self.fc2(self.dropout2(F.relu(self.fc1(self.flatten(x)))))


class ConvLSTMPooledModel(nn.Module):
    """
    Lightweight ConvLSTM variant using adaptive pooling and dropout to reduce overfitting.
    """

    def __init__(self, num_classes: int, input_shape: tuple = (3, 64, 64)) -> None:
        """
        Initializes the pooled model architecture with dropout regularization.

        Parameters:
            num_classes (int): Number of output classes for prediction.
            input_shape (tuple[int, int, int]): Shape of the single input frame (C, H, W).

        Returns:
            None
        """
        super().__init__()

        C, H, W = input_shape

        self.td_conv = nn.Sequential(
            nn.Conv2d(C, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.convlstm = ConvLSTM2D(32, 32)
        self.bn = nn.BatchNorm2d(32)

        self.conv_post = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

        total = sum(p.numel() for p in self.parameters())

        print(f"ConvLSTMPooledModel initialized with {total:,} parameters.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass returning raw class logits.

        Parameters:
            x (torch.Tensor): Input video tensor of shape (B, T, C, H, W).

        Returns:
            logits (torch.Tensor): Unnormalized class prediction logits of shape (B, num_classes).
        """
        B, T, C, H, W = x.shape

        x = self.td_conv(x.view(B * T, C, H, W)).view(B, T, 32, H, W)
        x = self.convlstm(x)
        x = self.bn(x)
        x = self.conv_post(x)
        x = self.pool(x)
        
        x = x.view(B, -1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x


class ConvLSTMModel(nn.Module):
    """
    Ultra-lightweight ConvLSTM variant with reduced filter sizes.
    """

    def __init__(self, num_classes: int, input_shape: tuple = (3, 64, 64)) -> None:
        """
        Initializes the lightweight model architecture and prints its parameter count.

        Parameters:
            num_classes (int): Number of output classes for prediction.
            input_shape (tuple[int, int, int]): Shape of the single input frame (C, H, W).

        Returns:
            None
        """
        super().__init__()
        C, H, W = input_shape
        self.td_conv = nn.Conv2d(C, 2, 3, padding=1)
        self.convlstm = ConvLSTM2D(2, 4)
        self.bn = nn.BatchNorm2d(4)
        self.conv_post = nn.Conv2d(4, 2, 3, padding=1)
        self.dropout1 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2 * H * W, 32)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, num_classes)

        total = sum(p.numel() for p in self.parameters())
        print(f"🧠 ConvLSTMModel (light) | params={total:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass returning raw class logits.

        Parameters:
            x (torch.Tensor): Input video tensor of shape (B, T, C, H, W).

        Returns:
            logits (torch.Tensor): Unnormalized class prediction logits of shape (B, num_classes).
        """
        B, T, C, H, W = x.shape
        x = F.relu(self.td_conv(x.view(B * T, C, H, W))).view(B, T, 2, H, W)
        x = self.bn(self.convlstm(x))
        x = self.dropout1(F.relu(self.conv_post(x)))
        return self.fc2(self.dropout2(F.relu(self.fc1(self.flatten(x)))))


class ConvLSTMCustom(nn.Module):
    """
    Customizable ConvLSTM architecture parameterized by a list of filter sizes.
    """

    def __init__(
        self,
        num_classes: int,
        input_shape: tuple = (3, 64, 64),
        filters: list[int] = [16, 64, 16, 256],
    ) -> None:
        """
        Initializes the custom model using the provided network dimensionalities.

        Parameters:
            num_classes (int): Number of output classes for prediction.
            input_shape (tuple[int, int, int]): Shape of the single input frame (C, H, W).
            filters (list[int]): Four integers representing sizes for [td_conv, convlstm, conv_post, fc1].

        Returns:
            None
        """
        super().__init__()
        C, H, W = input_shape
        f_td, f_lstm, f_post, f_fc = filters

        self.td_conv = nn.Conv2d(C, f_td, 3, padding=1)
        self.convlstm = ConvLSTM2D(f_td, f_lstm)
        self.bn = nn.BatchNorm2d(f_lstm)
        self.conv_post = nn.Conv2d(f_lstm, f_post, 3, padding=1)
        self.dropout1 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(f_post * H * W, f_fc)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(f_fc, num_classes)

        self._f_td = f_td  # Save for view reshaping in forward

        total = sum(p.numel() for p in self.parameters())
        print(f"ConvLSTMCustom initialized with {total:,} parameters.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass returning raw class logits.

        Parameters:
            x (torch.Tensor): Input video tensor of shape (B, T, C, H, W).

        Returns:
            logits (torch.Tensor): Unnormalized class prediction logits of shape (B, num_classes).
        """
        B, T, C, H, W = x.shape
        x = F.relu(self.td_conv(x.view(B * T, C, H, W))).view(B, T, self._f_td, H, W)
        x = self.bn(self.convlstm(x))
        x = self.dropout1(F.relu(self.conv_post(x)))
        return self.fc2(self.dropout2(F.relu(self.fc1(self.flatten(x)))))


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
    parser.add_argument(
        "--original", action="store_true", help="Use original large model"
    )
    args = parser.parse_args()

    out_dir = Path("outputs") / "model_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AHARDataset(
        args.dataset_dir, args.sequence_length, (args.height, args.width)
    )

    ModelClass = ConvLSTMOriginal if args.original else ConvLSTMModel
    model = ModelClass(
        dataset.num_classes, input_shape=(3, args.height, args.width)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model: {ModelClass.__name__} | Params: {total_params:,}")

    if args.model_dir:
        from .utils import load_model

        model, _, epoch, loss = load_model(
            model,
            checkpoint_path=f"{args.model_dir}/best_model.pth",
            map_location=device,
        )
        print(f"📂 Loaded - epoch={epoch}, loss={loss:.4f}")
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
    print(f"\n🎬 Saved - {out_dir / fname}")


if __name__ == "__main__":
    main()
