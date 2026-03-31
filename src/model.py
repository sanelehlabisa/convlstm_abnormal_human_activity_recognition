"""
model.py

ConvLSTM model for video classification in the Abnormal Human Activity Recognition (AHAR) project.
CNN extracts spatial features per frame, LSTM models temporal dynamics.

Author: Sanele Hlabisa

python -m src.model \
    --dataset_dir "datasets/abnormal_activities" \
    --model_dir "models"
"""

from __future__ import annotations

import random
import argparse
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .dataset import AHARDataset


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
        x = x.reshape(B, T, -1)

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
    - Build model (optionally load checkpoint)
    - Run single forward pass
    - Save predicted clip to outputs/model_samples/
    """

    parser = argparse.ArgumentParser(description="Model sanity check — forward pass + clip save")
    parser.add_argument("--dataset_dir",     type=str, default="datasets/abnormal_activities")
    parser.add_argument("--model_dir",       type=str, default=None,
                        help="Optional: path to models/ dir to load best_model.pth")
    parser.add_argument("--sequence_length", type=int, default=32)
    parser.add_argument("--height",          type=int, default=64)
    parser.add_argument("--width",           type=int, default=64)
    parser.add_argument("--fps",             type=int, default=8)
    args = parser.parse_args()

    out_dir = Path("outputs") / "model_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Using device: {device}")

    # ---- Dataset ----
    dataset = AHARDataset(
        dataset_dir=args.dataset_dir,
        sequence_length=args.sequence_length,
        frame_size=(args.height, args.width),
    )

    idx = random.randint(0, len(dataset) - 1)
    frames, true_label = dataset[idx]          # (T, C, H, W) float [0,1]

    # ---- Model ----
    model = ConvLSTMModel(
        num_classes=dataset.num_classes,
        input_shape=(3, args.height, args.width),
    ).to(device)

    # Optionally load checkpoint if model_dir provided
    if args.model_dir is not None:
        from .utils import load_model
        model, _, epoch, loss = load_model(model, base_path=args.model_dir, map_location=device)
        print(f"📂 Checkpoint loaded → epoch={epoch}, loss={loss:.4f}")
    else:
        print("⚠️  No model_dir provided — running with random weights (shape check only)")

    # ---- Forward pass ----
    model.eval()
    with torch.no_grad():
        logits = model(frames.unsqueeze(0).to(device))   # (1, num_classes)
        probs = torch.softmax(logits, dim=1)[0]
        pred_label = probs.argmax().item()
        pred_conf  = probs[pred_label].item()

    true_name = dataset.class_names[true_label]
    pred_name = dataset.class_names[pred_label]

    print(f"\n✅ Input shape  : {frames.shape}")
    print(f"✅ Logits shape : {logits.shape}")
    print(f"🔹 True class   : {true_name}")
    print(f"🔹 Pred class   : {pred_name}  ({pred_conf:.1%})")
    print(f"🔹 Probs        : {np.round(probs.cpu().numpy(), 2)}")

    # ---- Save clip ----
    try:
        stem = Path(dataset.samples[idx][0]).stem
    except Exception:
        stem = f"sample_{idx:04d}"

    correct = "correct" if pred_label == true_label else "wrong"
    fname   = f"{stem}_true-{true_name}_pred-{pred_name}_{correct}.mp4"
    out_path = out_dir / fname

    clip_uint8 = (frames * 255).byte().permute(0, 2, 3, 1).cpu()
    torchvision.io.write_video(str(out_path), clip_uint8, fps=args.fps)

    print(f"\n🎬 Saved clip → {out_path}")


if __name__ == "__main__":
    main()
