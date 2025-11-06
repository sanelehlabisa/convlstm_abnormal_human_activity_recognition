"""
utils.py

Utility functions for the ConvLSTM-based Abnormal Human Activity Recognition (AHAR) project.
Includes visualization and model persistence utilities.

Author: Sanele Hlabisa
"""

from __future__ import annotations
import os
import torch
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


# ========================
# Configuration constants
# ========================
BATCH_SIZE: int = 8
SEQUENCE_LENGTH: int = 32
WIDTH: int = 128
HEIGHT: int = 128
LEARNING_RATE: float = 1e-3
EPOCHS: int = 64
NUM_CHANNELS: int = 3
DATASET_SPLIT: tuple[float, float, float] = (0.7, 0.2, 0.1)
DATASET_DIR: str = "dataset"
OUTPUTS_DIR: str = "outputs"


# ========================
# Display Utility
# ========================
def display_frames(
    frames: list[np.ndarray],
    label: str,
    prediction: Optional[str] = None,
    probs: Optional[np.ndarray] = None,
) -> None:
    """
    Display a grid of frames with an optional label, prediction, and probability vector.

    Args:
        frames (list[np.ndarray]): List of image frames (BGR or RGB).
        label (str): Ground truth label.
        prediction (Optional[str]): Predicted class name (default: None).
        probs (Optional[np.ndarray]): Probability distribution vector (default: None).

    Returns:
        None
    """
    num_frames = len(frames)
    cols = min(num_frames, 5)
    rows = (num_frames + cols - 1) // cols

    plt.figure(figsize=(cols * 5, rows * 5))

    for i, frame in enumerate(frames):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis("off")

    # Construct title
    title = f"Label: {label}"
    if prediction is not None:
        title += f" | Prediction: {prediction}"
    if probs is not None:
        title += f"\nProbabilities: {np.round(probs, 3)}"

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUTS_DIR}/{label}frames.png")

def get_device() -> torch.device:
    """Return available device (CUDA if possible)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_training_curves(train_loss, val_loss, train_acc, val_acc, save_path=f"{OUTPUTS_DIR}/training_curves.png"):
    """Plot training vs validation loss and accuracy."""
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Val")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, val_acc, label="Val")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)


# ========================
# Model Persistence
# ========================
def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    base_path: str = "models",
    src_dir: str = "src",
) -> None:
    """
    Save model checkpoint along with a reproducible snapshot of the code.

    Creates a subfolder within `models/` named `model_{val_loss:.4f}` for this checkpoint,
    copies the current `src/` directory into it, and saves the model and optimizer state.

    Also updates `best_model.pth` in the base folder for resuming training easily.

    Args:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        epoch (int): Current epoch number.
        loss (float): Validation loss (used in folder name).
        base_path (str): Root folder to save model checkpoints.
        src_dir (str): Source code folder to copy for reproducibility.

    Returns:
        None
    """
    os.makedirs(base_path, exist_ok=True)

    # Folder for this specific checkpoint
    folder_name = f"model_{loss:.4f}"
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Save checkpoint in this folder
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

    # Copy src folder for reproducibility
    src_copy_path = os.path.join(folder_path, "src")
    if os.path.exists(src_copy_path):
        shutil.rmtree(src_copy_path)
    shutil.copytree(src_dir, src_copy_path)

    print(f"✅ Saved checkpoint at {ckpt_path} with source code snapshot.")

    # Update best_model.pth in base_path for easy resume
    best_model_path = os.path.join(base_path, "best_model.pth")
    shutil.copy2(ckpt_path, best_model_path)
    print(f"✅ Updated best_model.pth at {best_model_path}")

def load_model(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    base_path: str = "models",
    map_location: str = "cpu",
) -> tuple[torch.nn.Module, Optional[torch.optim.Optimizer], int, float]:
    """
    Load model and optimizer state from the latest best checkpoint if available.

    Args:
        model (torch.nn.Module): Model to load weights into.
        optimizer (Optional[torch.optim.Optimizer]): Optimizer to load state into (optional).
        base_path (str): Base folder where models/best_model.pth is stored.
        map_location (str): Device mapping for model.

    Returns:
        tuple: (model, optimizer, start_epoch, best_loss)
    """
    best_model_path = os.path.join(base_path, "best_model.pth")

    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_loss = checkpoint.get("loss", float("inf"))

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"✅ Loaded checkpoint from {best_model_path} (epoch {start_epoch}, loss {best_loss:.4f})")
    else:
        start_epoch = 0
        best_loss = float("inf")
        print(f"⚠️ No existing checkpoint found at {best_model_path}. Starting fresh.")

    return model, optimizer, start_epoch, best_loss
