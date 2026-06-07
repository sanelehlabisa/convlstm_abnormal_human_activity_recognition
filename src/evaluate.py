"""
evaluate.py

Evaluation script for AHAR.

Author: Sanele Hlabisa

python -m src.evaluate \
    --dataset_dir "datasets/processed/videos_violence-detection-dataset" \
    --checkpoint_path "models/violence-detection-dataset_best_model.pth" \
    --custom_filters 32 64 8 256 \
    --experiments_dir "experiments" \
    --batch_size 32 \
    --sequence_length 64 \
    --height 64 \
    --width 64 \
    --num_workers 2 \
    --pin_memory \
    --num_samples 8
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import Dataset, DataLoader, random_split

from .dataset import AHARDataset
from .model import ConvLSTMModel, ConvLSTMPooledModel, ConvLSTMCustom
from .utils import load_model, plot_confusion_matrix, save_prediction_clips

parser = argparse.ArgumentParser(description="Evaluate ConvLSTM for AHAR")
parser.add_argument("--dataset_dir", type=str, default="datasets/abnormal_activities")
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--experiments_dir", type=str, default="experiments")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--sequence_length", type=int, default=32)
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--height", type=int, default=128)
parser.add_argument(
    "--custom_filters",
    type=int,
    nargs="+",
    default=[32, 64, 4, 256],
    help="List of 4 integers for ConvLSTMCustom: [td_conv, convlstm, conv_post, fc1]",
)
parser.add_argument("--train_ratio", type=float, default=0.7)
parser.add_argument("--val_ratio", type=float, default=0.1)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--pin_memory", action="store_true")
parser.add_argument("--num_samples", type=int, default=8)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    metrics: dict,
    device: torch.device,
):
    """
    Evaluates the model performance on a given dataset loader using specified metrics.

    Parameters:
        model (torch.nn.Module): The trained PyTorch model to evaluate.
        loader (DataLoader): The DataLoader providing the evaluation data batches.
        criterion (torch.nn.Module): The loss function used to calculate the error.
        metrics (dict): A dictionary of TorchMetrics objects to compute.
        device (torch.device): The hardware device to run inference on.

    Returns:
        results (dict): A dictionary containing the computed loss and metric scores.
    """
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
def _collect_preds(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """
    Runs inference over a dataloader to collect all true and predicted labels.

    Parameters:
        model (torch.nn.Module): The trained PyTorch model.
        loader (DataLoader): The DataLoader providing the evaluation data.
        device (torch.device): The hardware device to run inference on.

    Returns:
        labels (tuple): A tuple containing a list of true labels and a list of predicted labels.
    """
    model.eval()
    all_true, all_pred = [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        preds = model(X).argmax(dim=1).cpu().tolist()
        all_pred.extend(preds)
        all_true.extend(y.tolist())
    return all_true, all_pred


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

    model = ConvLSTMCustom(
        num_classes,
        input_shape=(3, args.height, args.width),
        filters=args.custom_filters,
    ).to(device)

    # Initialize defaults in case checkpoint loading is skipped or fails
    epoch = 0
    ckpt_loss = float("inf")

    if args.checkpoint_path and Path(args.checkpoint_path).is_file():
        try:
            model, _, epoch, ckpt_loss = load_model(
                model, checkpoint_path=args.checkpoint_path, map_location=device
            )
            print(f"📂 Checkpoint → epoch={epoch}, loss={ckpt_loss:.4f}")
        except RuntimeError as e:
            print(f"❌ Architecture mismatch: {e}")
            print("⚠️ Falling back to default initialized weights.")
    else:
        print(
            "⚠️ No valid checkpoint path provided or found. Using default initialized weights."
        )

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

    print("\n🎬 Saving prediction clips...")
    clip_records = save_prediction_clips(
        model, test_set, dataset.class_names, device, exp_dir, args.num_samples
    )

    report = {
        "timestamp": timestamp,
        "dataset": dataset_name,
        "dataset_mode": "classification",
        "classes": dataset.class_names,
        "checkpoint": {
            "epoch": epoch,
            "saved_loss": round(ckpt_loss, 6) if ckpt_loss != float("inf") else None,
        },
        "metrics": {k: round(v, 6) for k, v in results.items()},
        "artifacts": {"confusion_matrix": cm_path, "prediction_clips": clip_records},
    }
    json_path = exp_dir / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"📄 Report → {json_path}")


if __name__ == "__main__":
    main()
