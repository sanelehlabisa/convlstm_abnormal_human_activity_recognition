"""
evaluate.py

Evaluation script for AHAR.

Author: Sanele Hlabisa

python -m src.evaluate \
    --dataset_dir "datasets/abnormal_activities" \
    --checkpoint_path "models/best_model.pth" \
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
import math
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader, random_split

from .dataset import AHARDataset
from .model import ConvLSTMModel
from .utils import load_model, plot_confusion_matrix, save_prediction_clips

parser = argparse.ArgumentParser(description="Evaluate ConvLSTM for AHAR")
parser.add_argument("--dataset_dir", type=str, default="datasets/abnormal_activities")
parser.add_argument("--checkpoint_path", type=str, default="models/best_model.pth")
parser.add_argument("--experiments_dir", type=str, default="experiments")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--sequence_length", type=int, default=32)
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--height", type=int, default=128)
parser.add_argument("--train_ratio", type=float, default=0.7)
parser.add_argument("--val_ratio", type=float, default=0.15)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--pin_memory", action="store_true")
parser.add_argument("--num_samples", type=int, default=8)


@torch.inference_mode()
def evaluate(model, loader, criterion, metrics, device):
    model.eval()
    for m in metrics.values():
        m.reset()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        total_loss += criterion(logits, y).item()
        preds = logits.argmax(dim=1)
        for m in metrics.values():
            m(preds, y)
    results = {"loss": total_loss / len(loader)}
    for name, m in metrics.items():
        results[name] = m.compute().item()
    return results


@torch.inference_mode()
def _collect_preds(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        preds = model(X).argmax(dim=1).cpu().tolist()
        all_pred.extend(preds)
        all_true.extend(y.tolist())
    return all_true, all_pred


@torch.inference_mode()
def _save_sample_predictions(
    model, dataset, class_names, device, num_samples, save_path
):
    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    cols = 4
    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2), squeeze=False)

    for cell_idx, sample_idx in enumerate(indices):
        ax = axes[cell_idx // cols][cell_idx % cols]
        frames, true_label = dataset[sample_idx]
        pred_label = model(frames.unsqueeze(0).to(device)).argmax(dim=1).item()
        mid = frames.shape[0] // 2
        frame_np = np.clip(frames[mid].permute(1, 2, 0).cpu().numpy(), 0, 1)
        ax.imshow(frame_np)
        ax.axis("off")
        true_name = class_names[true_label]
        pred_name = class_names[pred_label]
        color = "green" if pred_label == true_label else "red"
        ax.set_title(f"T: {true_name}\nP: {pred_name}", fontsize=8, color=color, pad=3)

    for i in range(len(indices), rows * cols):
        axes[i // cols][i % cols].axis("off")

    fig.suptitle("Sample Predictions (green=correct, red=wrong)", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"🖼  Saved sample predictions → {save_path}")


def main() -> None:
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = Path(args.dataset_dir).name
    exp_dir = Path(args.experiments_dir) / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Experiment dir → {exp_dir}")

    dataset = AHARDataset(
        args.dataset_dir, args.sequence_length, (args.width, args.height)
    )
    num_classes = dataset.num_classes
    print(f"📦 {len(dataset)} samples | {num_classes} classes: {dataset.class_names}")

    n_total = len(dataset)
    n_train = int(args.train_ratio * n_total)
    n_val = int(args.val_ratio * n_total)
    n_test = n_total - n_train - n_val
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

    model = ConvLSTMModel(num_classes, input_shape=(3, args.height, args.width)).to(
        device
    )
    model, _, epoch, ckpt_loss = load_model(
        model, checkpoint_path=args.checkpoint_path, map_location=device
    )
    print(f"📂 Checkpoint → epoch={epoch}, loss={ckpt_loss:.4f}")

    criterion = nn.CrossEntropyLoss()
    metrics = {
        "accuracy": torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        ).to(device),
        "precision": torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device),
        "recall": torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device),
        "f1": torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        ).to(device),
    }

    results = evaluate(model, test_loader, criterion, metrics, device)
    print("\n🏁 Results")
    print("─" * 32)
    for name, value in results.items():
        print(f"  {name:<12}: {value:.4f}")

    all_true, all_pred = _collect_preds(model, test_loader, device)
    cm_path = str(exp_dir / "confusion_matrix.png")
    plot_confusion_matrix(
        all_true,
        all_pred,
        dataset.class_names,
        dataset_name=dataset_name,
        save_path=cm_path,
    )

    _save_sample_predictions(
        model,
        test_set,
        dataset.class_names,
        device,
        args.num_samples,
        str(exp_dir / "sample_predictions.png"),
    )

    print("\n🎬 Saving prediction clips...")
    clip_records = save_prediction_clips(
        model, test_set, dataset.class_names, device, exp_dir, args.num_samples
    )

    report = {
        "timestamp": timestamp,
        "dataset": dataset_name,
        "dataset_mode": "classification",
        "classes": dataset.class_names,
        "checkpoint": {"epoch": epoch, "saved_loss": round(ckpt_loss, 6)},
        "metrics": {k: round(v, 6) for k, v in results.items()},
        "artifacts": {"confusion_matrix": cm_path, "prediction_clips": clip_records},
    }
    json_path = exp_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"📄 Report → {json_path}")


if __name__ == "__main__":
    main()
