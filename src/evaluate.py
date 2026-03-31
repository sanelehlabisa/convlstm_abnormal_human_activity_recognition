"""
evaluate.py

Evaluation script for ConvLSTM-based Abnormal Human Activity Recognition (AHAR).
Loads the best saved model and evaluates on the held-out test split.
Saves a timestamped experiment folder containing:
  - metrics.json  (loss, accuracy, precision, recall, f1)
  - confusion_matrix.png
  - sample_predictions.png  (8 random test samples)

Author: Sanele Hlabisa

python -m src.evaluate \
    --dataset_dir "datasets/abnormal_activities" \
    --model_dir "models" \
    --experiments_dir "experiments" \
    --batch_size 32 \
    --sequence_length 32 \
    --height 64 \
    --width 64 \
    --num_workers 2 \
    --pin_memory \
    --num_samples 8
"""

from __future__ import annotations

import argparse
import json
import random
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np

from .dataset import AHARDataset
from .model import ConvLSTMModel
from .utils import load_model, plot_confusion_matrix, save_prediction_clips

# ============================================================
# Argument parser
# ============================================================

parser = argparse.ArgumentParser(description="Evaluate ConvLSTM model for AHAR")

parser.add_argument("--dataset_dir", type=str, default="dataset_clean")
parser.add_argument("--model_dir", type=str, default="models")
parser.add_argument("--experiments_dir", type=str, default="experiments")
parser.add_argument("--batch_size", type=int, default=8)

parser.add_argument("--sequence_length", type=int, default=32)
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--height", type=int, default=128)

parser.add_argument("--train_ratio", type=float, default=0.7)
parser.add_argument("--val_ratio", type=float, default=0.15)

parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--pin_memory", action="store_true")

parser.add_argument("--num_samples", type=int, default=8, help="Number of random test samples to visualize")

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
    pred_transform=None,
) -> dict[str, float]:
    """
    Run one full evaluation pass and return all metric results.

    Args:
        model:     Trained model.
        loader:    DataLoader for the split to evaluate.
        criterion: Loss function.
        metrics:   Dict of {display_name: torchmetrics.Metric}.
        device:    CPU or CUDA device.
        pred_transform: Optional function to transform predictions (e.g., for binary classification).

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

        # Post-process to binary if dataset is detection mode
        if pred_transform is not None:
            preds = torch.tensor(pred_transform(preds.cpu().tolist()), device=device)
            y = torch.tensor(pred_transform(y.cpu().tolist()),     device=device)

        for m in metrics.values():
            m(preds, y)

    results = {"loss": total_loss / len(loader)}
    for name, m in metrics.items():
        results[name] = m.compute().item()

    return results


# ============================================================
# Helpers
# ============================================================

@torch.inference_mode()
def _collect_preds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Return (all_true_labels, all_pred_labels) over the full loader."""
    model.eval()
    all_true, all_pred = [], []

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        logits = model(X)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_pred.extend(preds)
        all_true.extend(y.tolist())

    return all_true, all_pred


@torch.inference_mode()
def _plot_sample_predictions(
    model: nn.Module,
    dataset,
    class_names: list[str],
    device: torch.device,
    num_samples: int,
    save_path: str,
) -> None:
    """Pick random samples, run inference, save a grid of frame-strip predictions."""

    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    cols = 4
    rows = math.ceil(num_samples / cols)

    # Each cell shows first frame of the clip + true/pred label
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2), squeeze=False)

    for cell_idx, sample_idx in enumerate(indices):
        ax = axes[cell_idx // cols][cell_idx % cols]

        frames, true_label = dataset[sample_idx]          # (T, C, H, W)
        logits = model(frames.unsqueeze(0).to(device))
        pred_label = logits.argmax(dim=1).item()

        # Show middle frame for a representative thumbnail
        mid = frames.shape[0] // 2
        frame_np = frames[mid].permute(1, 2, 0).cpu().numpy()
        frame_np = np.clip(frame_np, 0, 1)

        ax.imshow(frame_np)
        ax.axis("off")

        true_name = class_names[true_label]
        pred_name = class_names[pred_label]
        color = "green" if pred_label == true_label else "red"
        ax.set_title(f"T: {true_name}\nP: {pred_name}", fontsize=8, color=color, pad=3)

    # Hide unused cells
    for cell_idx in range(len(indices), rows * cols):
        axes[cell_idx // cols][cell_idx % cols].axis("off")

    fig.suptitle("Sample Predictions  (green = correct, red = wrong)", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"🖼  Saved sample predictions -> {save_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Main function.

    Args:
        None

    Returns:
        None
    """
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Using device: {device}")

    # ---------------- Experiment folder ----------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(args.experiments_dir) / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Experiment dir → {exp_dir}")

        # ---------------- Dataset ----------------
    dataset = AHARDataset(
        dataset_dir=args.dataset_dir,
        sequence_length=args.sequence_length,
        frame_size=(args.width, args.height),
    )

    num_classes = dataset.num_classes                    # ← moved up, before mode branch
    print(f"📦 Dataset: {len(dataset)} samples | {num_classes} classes")
    print(f"🏷  Classes: {dataset.class_names}")

    # ---------------- Dataset mode → controls metrics and confusion matrix ----------------
    if dataset.mode == "binary":
        print("🔀 Binary evaluation mode (normal / abnormal)")

        def to_binary(indices: list[int]) -> list[int]:
            result = []
            for idx in indices:
                canonical = dataset.class_names[idx]
                verdict   = dataset.registry.binary_verdict(canonical)
                result.append(0 if verdict == "normal" else 1)
            return result

        eval_class_names = ["normal", "abnormal"]
        eval_num_classes = 2
    else:
        print("🔢 Multiclass evaluation mode")
        to_binary        = None
        eval_class_names = dataset.class_names
        eval_num_classes = num_classes

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



    # ---------------- Reproduce same split as train.py ----------------
    n_total = len(dataset)
    n_train = int(args.train_ratio * n_total)
    n_val   = int(args.val_ratio   * n_total)
    n_test  = n_total - n_train - n_val

    _, _, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"📊 Test split: {n_test} samples")

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # ---------------- Dataset mode → controls post-processing ----------------
    if dataset.mode == "detection":
        print("🔍 Detection mode — multiclass predictions collapsed to normal/abnormal")

        def to_detection(indices: list[int]) -> list[int]:
            """Collapse multiclass index → 0 (normal) or 1 (abnormal)."""
            result = []
            for idx in indices:
                canonical = dataset.class_names[idx]
                verdict   = dataset.registry.detection_verdict(canonical)
                # default to abnormal if unmapped — safe fallback
                result.append(0 if verdict == "normal" else 1)
            return result

        post_process     = to_detection
        eval_class_names = ["normal", "abnormal"]
        eval_num_classes = 2

    else:
        print("🔢 Multiclass mode")
        post_process     = None
        eval_class_names = dataset.class_names
        eval_num_classes = dataset.num_classes

    # ---------------- Metrics ----------------
    criterion = nn.CrossEntropyLoss()

    metrics = {
        "accuracy":  torchmetrics.Accuracy( task="multiclass", num_classes=eval_num_classes).to(device),
        "precision": torchmetrics.Precision(task="multiclass", num_classes=eval_num_classes, average="macro").to(device),
        "recall":    torchmetrics.Recall(   task="multiclass", num_classes=eval_num_classes, average="macro").to(device),
        "f1":        torchmetrics.F1Score(  task="multiclass", num_classes=eval_num_classes, average="macro").to(device),
    }

    # ---------------- Evaluate ----------------
    results = evaluate(
        model, test_loader, criterion, metrics, device,
        pred_transform=post_process,
    )

    print("\n🏁 Evaluation Results")
    print("─" * 32)
    for name, value in results.items():
        print(f"  {name:<12}: {value:.4f}")

    # ---------------- Confusion matrix ----------------
    all_true, all_pred = _collect_preds(model, test_loader, device)

    cm_path = str(exp_dir / "confusion_matrix.png")
    if post_process is not None:
        cm_true = post_process(all_true)
        cm_pred = post_process(all_pred)
    else:
        cm_true, cm_pred = all_true, all_pred

    plot_confusion_matrix(
        y_true=cm_true,
        y_pred=cm_pred,
        class_names=eval_class_names,
        save_path=cm_path,
    )

    # ---------------- Sample predictions ----------------
    _plot_sample_predictions(
        model=model,
        dataset=test_set,
        class_names=eval_class_names,
        device=device,
        num_samples=args.num_samples,
        save_path=str(exp_dir / "sample_predictions.png"),
    )

    # ---------------- Save prediction clips ----------------
    print("\n🎬 Saving prediction clips...")
    clip_records = save_prediction_clips(
        model=model,
        dataset=test_set,
        class_names=dataset.class_names,
        device=device,
        exp_dir=exp_dir,
        num_samples=args.num_samples,
        fps=8,
    )

    # ---------------- Save JSON report ----------------
    report = {
        "timestamp": timestamp,
        "checkpoint": {
            "epoch": epoch,
            "saved_loss": round(ckpt_loss, 6),
        },
        "args": vars(args),
        "metrics": {k: round(v, 6) for k, v in results.items()},
        "artifacts": {
            "confusion_matrix":   cm_path,
            "prediction_clips":   clip_records
        },
    }

    json_path = exp_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"📄 Saved metrics report  → {json_path}")


if __name__ == "__main__":
    main()
