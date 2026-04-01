"""
train.py

Training script for ConvLSTM-based Abnormal Human Activity Recognition (AHAR).
Handles dataset splitting, training, validation, checkpointing, and visualization.

Author: Sanele Hlabisa

python -m src.train \
    --dataset_dir "datasets/abnormal_activities" \
    --model_dir "models" \
    --batch_size 32 \
    --epochs 64 \
    --sequence_length 32 \
    --height 64 \
    --width 64 \
    --num_workers 2 \
    --pin_memory
"""

from __future__ import annotations

import argparse
from pathlib import Path
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .dataset import AHARDataset
from .model import ConvLSTMModel
from .utils import plot_training_curves, save_model, save_prediction_clips

parser = argparse.ArgumentParser(description="Train ConvLSTM for AHAR")
parser.add_argument("--dataset_dir", type=str, default="datasets/abnormal_activities")
parser.add_argument("--model_dir", type=str, default="models")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--sequence_length", type=int, default=32)
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--height", type=int, default=128)
parser.add_argument("--train_ratio", type=float, default=0.7)
parser.add_argument("--val_ratio", type=float, default=0.15)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--pin_memory", action="store_true")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    accuracy_fn: torchmetrics.Metric,
    device: torch.device,
) -> tuple[float, float]:

    model.train()

    total_loss = total_acc = 0.0
    for X, y in tqdm(loader, leave=False):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits: torch.Tensor = model(X)
        loss: torch.Tensor = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += accuracy_fn(logits.argmax(dim=1), y).item()
    return total_loss / len(loader), total_acc / len(loader)


@torch.inference_mode()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    accuracy_fn: torchmetrics.Metric,
    device: torch.device,
) -> tuple[float, float]:

    model.eval()

    total_loss = total_acc = 0.0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits: torch.Tensor = model(X)
        loss: torch.Tensor = criterion(logits, y)
        total_loss += loss.item()
        total_acc += accuracy_fn(logits.argmax(dim=1), y).item()
    return total_loss / len(loader), total_acc / len(loader)


def main() -> None:
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Using device: {device}")

    dataset = AHARDataset(
        args.dataset_dir, args.sequence_length, (args.width, args.height)
    )
    dataset_name = Path(args.dataset_dir).name
    num_classes = dataset.num_classes
    print(f"📦 {len(dataset)} samples | {num_classes} classes")

    n_total = len(dataset)
    n_train = int(args.train_ratio * n_total)
    n_val = int(args.val_ratio * n_total)
    n_test = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"📊 Train: {n_train} | Val: {n_val} | Test: {n_test}")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    model = ConvLSTMModel(num_classes, input_shape=(3, args.height, args.width)).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    acc_fn = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(
        device
    )

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_loss = float("inf")

    print("🚀 Training...")
    start = timer()

    for epoch in range(args.epochs):
        print(f"\n🧠 Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, acc_fn, device
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, acc_fn.clone(), device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(
            f"  Loss → Train: {train_loss:.4f} Val: {val_loss:.4f} | Acc → Train: {train_acc:.4f} Val: {val_acc:.4f}"
        )

        # Save every epoch; also copy to best_model.pth if improved
        ckpt_path = str(Path(args.model_dir) / f"checkpoint_epoch{epoch+1:03d}.pth")
        save_model(model, optimizer, epoch, val_loss, checkpoint_path=ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            import shutil

            shutil.copy2(ckpt_path, str(Path(args.model_dir) / "best_model.pth"))
            print(f"  ⭐ New best model saved (val_loss={val_loss:.4f})")

    print(f"\n⏱  Done in {timer() - start:.1f}s")

    plot_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        dataset_name=dataset_name,
        save_dir=args.model_dir,
        show=False,
    )

    test_loss, test_acc = validate_one_epoch(
        model, test_loader, criterion, acc_fn.clone(), device
    )
    print(f"\n🏁 Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    out_dir = Path("outputs") / "train_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_prediction_clips(
        model, test_set, dataset.class_names, device, out_dir, num_samples=1
    )


if __name__ == "__main__":
    main()
