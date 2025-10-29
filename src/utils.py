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
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


# ========================
# Configuration constants
# ========================
BATCH_SIZE: int = 8
SEQUENCE_LENGTH: int = 15
WIDTH: int = 320
HEIGHT: int = 320
LEARNING_RATE: float = 1e-4
EPOCHS: int = 32
NUM_CHANNELS: int = 3
DATASET_SPLIT: tuple[float, float, float] = (0.7, 0.2, 0.1)


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

    plt.figure(figsize=(cols * 3, rows * 3))

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
    plt.show()


# ========================
# Model Persistence
# ========================
def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str = "checkpoints/best_model.pth",
) -> None:
    """
    Save model weights, optimizer state, current epoch, and loss.

    Args:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        epoch (int): Current epoch number.
        loss (float): Best loss achieved.
        path (str): Path to save the checkpoint.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        path,
    )
    print(f"✅ Model checkpoint saved at: {path}")


def load_model(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    path: str = "checkpoints/best_model.pth",
    map_location: str = "cpu",
) -> tuple[torch.nn.Module, Optional[torch.optim.Optimizer], int, float]:
    """
    Load model and optimizer state from a checkpoint.

    Args:
        model (torch.nn.Module): Model to load weights into.
        optimizer (Optional[torch.optim.Optimizer]): Optimizer to load (optional).
        path (str): Path to the checkpoint file.
        map_location (str): Device to map the model (default: "cpu").

    Returns:
        tuple: (model, optimizer, epoch, loss)
    """
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))

    print(f"✅ Loaded checkpoint from {path} (epoch {epoch}, loss {loss:.4f})")
    return model, optimizer, epoch, loss
