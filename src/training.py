"""

w
""""""
train.py

Training script for ConvLSTM-based Abnormal Human Activity Recognition (AHAR) project.
Handles dataset splitting, training, validation, checkpointing, and evaluation.

Author: Sanele Hlabisa
"""

from __future__ import annotations
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from src.dataset import AHARDataset
from src.model import ConvLSTMModel
from src.utils import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    DATASET_SPLIT,
    save_model,
    load_model,
    display_frames,
    get_device,
    plot_training_curves,
)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train model for one epoch."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y.argmax(dim=1)).sum().item()
        total += X.size(0)

    return total_loss / total, correct / total


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on validation or test data."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)

            total_loss += loss.item() * X.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y.argmax(dim=1)).sum().item()
            total += X.size(0)

    return total_loss / total, correct / total


def plot_metrics(train_loss, val_loss, train_acc, val_acc):
    """Plot loss and accuracy curves."""
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train Acc")
    plt.plot(epochs, val_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")

    os.makedirs("outputs", exist_ok=True)
    plt.tight_layout()
    plt.savefig("outputs/training_curves.png")
    plt.show()


def main() -> None:
    print("🚀 Starting training pipeline...")

    # Device setup
    device = get_device()
    print(f"🖥 Using device: {device}")

    # --- Load dataset and split ---
    full_dataset = AHARDataset()
    n_total = len(full_dataset)
    n_train = int(DATASET_SPLIT[0] * n_total)
    n_val = int(DATASET_SPLIT[-1] * n_total)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    print(f"📦 Dataset split → Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # --- Model setup ---
    model = ConvLSTMModel(num_classes=full_dataset.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_loss = float("inf")
    start_epoch = 0

    # Resume if checkpoint exists
    ckpt_path = "checkpoints/best_model.pth"
    if os.path.exists(ckpt_path):
        model, optimizer, start_epoch, best_val_loss = load_model(model, optimizer, ckpt_path, map_location=device)
        print(f"🔁 Resuming training from epoch {start_epoch + 1}")

    # --- Training loop ---
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(start_epoch, EPOCHS):
        print(f"\n🧠 Epoch {epoch + 1}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"📊 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            f"| Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}",
            f"| LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, epoch, best_val_loss)

    # --- Plot training metrics ---
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    # --- Evaluate best model on test set ---
    print("\n🔍 Loading best model for testing...")
    model, _, _, _ = load_model(model, optimizer, ckpt_path, map_location=device)
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"🏁 Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # --- Predict random sequence ---
    print("\n🎞 Predicting random sample from test set...")
    model.eval()
    random_idxs: list[int] = random.sample(range(len(test_set)), k=5)
    for idx in random_idxs:
        frames, label_vec = test_set[idx]
        label_name = full_dataset.class_names[label_vec.argmax()]

        seq_tensor = frames.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(seq_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            pred_class = full_dataset.class_names[pred_idx]

        print(f"🎯 True: {label_name}, Predicted: {pred_class}, Probs: {probs[0].cpu().numpy()}")
        display_frames(frames.cpu().permute(0, 2, 3, 1).numpy(), label_name, pred_class, probs[0].cpu().numpy())


if __name__ == "__main__":
    main()
