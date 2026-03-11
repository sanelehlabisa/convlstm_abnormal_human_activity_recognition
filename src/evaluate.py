"""
evaluate.py

Evaluation script for ConvLSTM-based Abnormal Human Activity Recognition (AHAR).
Loads the best saved model and evaluates on the held-out test split.

Author: Sanele Hlabisa
"""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchmetrics

from .dataset import AHARDataset
from .model import ConvLSTMModel
from .utils import load_model

# ============================================================
# Argument parser
# ============================================================

parser = argparse.ArgumentParser(description="Evaluate ConvLSTM model for AHAR")

parser.add_argument("--dataset_dir",     type=str,   default="dataset_clean")
parser.add_argument("--model_dir",       type=str,   default="models")
parser.add_argument("--batch_size",      type=int,   default=8)

parser.add_argument("--sequence_length", type=int,   default=32)
parser.add_argument("--width",           type=int,   default=128)
parser.add_argument("--height",          type=int,   default=128)

parser.add_argument("--train_ratio",     type=float, default=0.7)
parser.add_argument("--val_ratio",       type=float, default=0.15)

parser.add_argument("--num_workers",     type=int,   default=0)
parser.add_argument("--pin_memory",      action="store_true")

# ============================================================
# Evaluate loop
# ============================================================

@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    metrics: dict[str, torchmetrics.Metric],
    device: torch.device,
) -> dict[str, float]:
    """
    Run one full evaluation pass and return all metric results.

    Args:
        model:     Trained model.
        loader:    DataLoader for the split to evaluate.
        criterion: Loss function.
        metrics:   Dict of {display_name: torchmetrics.Metric}.
        device:    CPU or CUDA device.

    Returns:
        Dict of {metric_name: value}.
    """
    model.eval()

    for m in metrics.values():
        m.reset()

    total_loss = 0.0

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(X)
        total_loss += criterion(logits, y).item()

        preds = logits.argmax(dim=1)
        for m in metrics.values():
            m(preds, y)

    results = {"loss": total_loss / len(loader)}
    for name, m in metrics.items():
        results[name] = m.compute().item()

    return results


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Using device: {device}")

    # ---------------- Dataset ----------------
    dataset = AHARDataset(
        dataset_dir=args.dataset_dir,
        sequence_length=args.sequence_length,
        frame_size=(args.width, args.height),
    )

    num_classes = dataset.num_classes
    print(f"📦 Dataset: {len(dataset)} samples | {num_classes} classes")
    print(f"🏷  Classes: {dataset.class_names}")

    # ---------------- Reproduce same split as train.py ----------------
    n_total = len(dataset)
    n_train = int(args.train_ratio * n_total)
    n_val   = int(args.val_ratio   * n_total)
    n_test  = n_total - n_train - n_val

    _, _, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),   # same seed → same split
    )

    print(f"📊 Test split: {n_test} samples")

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # ---------------- Model ----------------
    model = ConvLSTMModel(
        num_classes=num_classes,
        input_shape=(3, args.height, args.width),
    ).to(device)

    model, _, epoch, ckpt_loss = load_model(
        model,
        base_path=args.model_dir,
        map_location=device,
    )

    print(f"📂 Checkpoint → epoch={epoch}, saved_loss={ckpt_loss:.4f}")

    # ---------------- Metrics ----------------
    criterion = nn.CrossEntropyLoss()

    metrics = {
        "accuracy":  torchmetrics.Accuracy( task="multiclass", num_classes=num_classes).to(device),
        "precision": torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro").to(device),
        "recall":    torchmetrics.Recall(   task="multiclass", num_classes=num_classes, average="macro").to(device),
        "f1":        torchmetrics.F1Score(  task="multiclass", num_classes=num_classes, average="macro").to(device),
    }

    # ---------------- Evaluate ----------------
    results = evaluate(model, test_loader, criterion, metrics, device)

    print("\n🏁 Evaluation Results")
    print("─" * 32)
    for name, value in results.items():
        print(f"  {name:<12}: {value:.4f}")


if __name__ == "__main__":
    main()

"""
python -m src.evaluate \
  --dataset_dir "/kaggle/input/abnormal-activities/abnormal_activities" \
  --batch_size 32 \
  --sequence_length 32 \
  --height 64 \
  --width 64 \
  --num_workers 2 \
  --pin_memory
"""