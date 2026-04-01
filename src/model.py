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
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(4)

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy = self._cnn_forward(dummy)
            feature_dim = dummy.view(1, -1).size(1)

        self.lstm = nn.LSTM(feature_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def _cnn_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = self._cnn_forward(x.view(B * T, C, H, W)).reshape(B, T, -1)
        x, _ = self.lstm(x)
        return self.fc(self.dropout(x[:, -1, :]))


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
        print("⚠️  No model_dir — random weights (shape check only)")

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
