"""
utils.py

Utility functions for the ConvLSTM-based Abnormal Human Activity Recognition (AHAR) project.
Includes visualization, training plots, and reproducible model persistence.

Author: Sanele Hlabisa
"""

from __future__ import annotations

import os
import json
import shutil
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================================================
# Visualization
# ============================================================

def display_video_grid(
    video: torch.Tensor,
    class_names: Optional[list[str]] = None,
    true_label: Optional[int] = None,
    pred_label: Optional[int] = None,
    max_cols: int = 5,
    frame_size: int = 3,
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Display video frames in a grid with optional true/predicted labels.

    Args:
        video: Tensor (T, C, H, W)
        class_names: List of class names
        true_label: Ground truth label index
        pred_label: Predicted label index
        max_cols: Maximum number of columns in grid
        frame_size: Size multiplier for each frame
        show: Whether to show the figure
        save_path: Optional path to save the image
    """
    assert video.ndim == 4, "Expected video shape (T, C, H, W)"

    T = video.size(0)
    cols = min(max_cols, T)
    rows = int(np.ceil(T / cols))

    video_np = video.permute(0, 2, 3, 1).cpu().numpy()
    video_np = np.clip(video_np, 0, 1)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * frame_size, rows * frame_size),
        squeeze=False,
    )

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < T:
            ax.imshow(video_np[i])
        ax.axis("off")

    # ---- Title logic ----
    title = ""
    color = "black"

    if true_label is not None and class_names:
        title += f"True: {class_names[true_label]}"

    if pred_label is not None and class_names:
        if title:
            title += " | "
        title += f"Pred: {class_names[pred_label]}"
        if true_label is not None:
            color = "green" if pred_label == true_label else "red"

    if title:
        fig.suptitle(title, fontsize=14, color=color)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"📸 Saved visualization to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# Training curves
# ============================================================

def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training vs validation loss and accuracy.

    Args:
        train_losses, val_losses: Loss history
        train_accs, val_accs: Accuracy history
        show: Whether to display the plot
        save_path: Optional path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train")
    plt.plot(epochs, val_accs, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"📊 Saved training curves to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# Model persistence
# ============================================================

def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    base_path: str = "models",
    src_dir: str = "src",
    extra_meta: Optional[dict] = None,
) -> None:
    """
    Save model checkpoint with reproducible source snapshot.
    """
    os.makedirs(base_path, exist_ok=True)

    folder_name = f"model_{loss:.4f}"
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    ckpt_path = os.path.join(folder_path, "checkpoint.pth")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        ckpt_path,
    )

    # Copy source code snapshot
    src_dst = os.path.join(folder_path, "src")
    if os.path.exists(src_dst):
        shutil.rmtree(src_dst)
    shutil.copytree(src_dir, src_dst)

    # Save metadata
    meta = {
        "epoch": epoch,
        "loss": loss,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "timestamp": datetime.now().isoformat(),
    }
    if extra_meta:
        meta.update(extra_meta)

    with open(os.path.join(folder_path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Update best model
    best_path = os.path.join(base_path, "best_model.pth")
    shutil.copy2(ckpt_path, best_path)

    print(f"✅ Saved checkpoint → {ckpt_path}")
    print(f"⭐ Updated best model → {best_path}")


def load_model(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    base_path: str = "models",
    map_location: str = "cpu",
) -> tuple[torch.nn.Module, Optional[torch.optim.Optimizer], int, float]:
    """
    Load best model checkpoint if available.
    """
    best_model_path = os.path.join(base_path, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("⚠️ No checkpoint found — starting from scratch.")
        return model, optimizer, 0, float("inf")

    checkpoint = torch.load(best_model_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))

    print(f"✅ Loaded best model (epoch={epoch}, loss={loss:.4f})")
    return model, optimizer, epoch, loss
