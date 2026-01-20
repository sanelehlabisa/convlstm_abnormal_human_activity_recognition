"""
train.py

Training script for ConvLSTM-based Abnormal Human Activity Recognition (AHAR).
Handles dataset splitting, training, validation, checkpointing, and visualization.

Author: Sanele Hlabisa
"""

from __future__ import annotations

import argparse
import random
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torchmetrics

from .dataset import AHARDataset
from .model import ConvLSTMModel
from .utils import (
    save_model,
    plot_training_curves,
    display_video_grid,
)

# ============================================================
# Argument parser
# ============================================================

parser = argparse.ArgumentParser(description="Train ConvLSTM model for AHAR")

parser.add_argument("--dataset_dir", type=str, default="dataset")
parser.add_argument("--model_dir", type=str, default="models")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)

parser.add_argument("--sequence_length", type=int, default=16)
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--height", type=int, default=128)

parser.add_argument("--train_ratio", type=float, default=0.7)
parser.add_argument("--val_ratio", type=float, default=0.15)

parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--pin_memory", action="store_true")

# ============================================================
# Train / Validate loops
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    accuracy_fn: torchmetrics.Metric,
    device: torch.device,
) -> tuple[float, float]:

    model.train()
    accuracy_fn.reset()

    total_loss = 0.0

    for X, y in tqdm(loader, leave=False):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        accuracy_fn.update(logits.softmax(dim=1), y)

    avg_loss = total_loss / len(loader)
    avg_acc = accuracy_fn.compute().item()

    return avg_loss, avg_acc


@torch.inference_mode()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    accuracy_fn: torchmetrics.Metric,
    device: torch.device,
) -> tuple[float, float]:

    model.eval()
    accuracy_fn.reset()

    total_loss = 0.0

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(X)
        loss = criterion(logits, y)

        total_loss += loss.item()
        accuracy_fn.update(logits.softmax(dim=1), y)

    avg_loss = total_loss / len(loader)
    avg_acc = accuracy_fn.compute().item()

    return avg_loss, avg_acc


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥 Using device: {device}")

    # ---------------- Dataset ----------------
    dataset = AHARDataset(
        dataset_dir=args.dataset_dir,
        sequence_length=args.sequence_length,
        frame_size=(args.width, args.height),
    )

    num_classes = dataset.num_classes
    print(f"📦 Dataset: {len(dataset)} samples | {num_classes} classes")

    # ---------------- Split ----------------
    n_total = len(dataset)
    n_train = int(args.train_ratio * n_total)
    n_val = int(args.val_ratio * n_total)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"📊 Split → Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # ---------------- Loaders ----------------
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_acc_fn = torchmetrics.Accuracy(
        task="multiclass", num_classes=num_classes
    ).to(device)
    val_acc_fn = train_acc_fn.clone()
    test_acc_fn = train_acc_fn.clone()

    # ---------------- Training ----------------
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("🚀 Starting training...")
    start_time = timer()

    for epoch in range(args.epochs):
        print(f"\n🧠 Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, train_acc_fn, device
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, val_acc_fn, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"📉 Loss → Train: {train_loss:.4f} | Val: {val_loss:.4f} "
            f"🎯 Acc → Train: {train_acc:.4f} | Val: {val_acc:.4f}"
        )

        save_model(model, optimizer, epoch, val_loss, base_path=args.model_dir)

    total_time = timer() - start_time
    print(f"\n⏱ Training finished in {total_time:.2f}s")

    # ---------------- Curves ----------------
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs
    )

    # ---------------- Test ----------------
    test_loss, test_acc = validate_one_epoch(
        model, test_loader, criterion, test_acc_fn, device
    )
    print(f"\n🏁 Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # ---------------- Visualization ----------------
    idx = random.randint(0, len(test_set) - 1)
    frames, true_label = test_set[idx]
    frames = frames.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(frames.unsqueeze(0))
        pred_label = logits.argmax(dim=1).item()

    display_video_grid(
        video=frames.cpu(),
        class_names=dataset.class_names,
        true_label=true_label,
        pred_label=pred_label,
    )


if __name__ == "__main__":
    main()
