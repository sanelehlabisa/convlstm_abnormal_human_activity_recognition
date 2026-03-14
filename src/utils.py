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
import random
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix as sk_cm
import numpy as np
from mlxtend.plotting import plot_confusion_matrix as mlxt_plot_cm
from mlxtend.evaluate import confusion_matrix as mlxt_cm


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

# ============================================================
# Confusion matrix
# ============================================================

def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    # force all classes to appear even if absent from this split
    cm = mlxt_cm(
        y_target=y_true,
        y_predicted=y_pred,
        binary=False,
        positive_label=1,   # ignored when binary=False, but required kwarg in some versions
    )

    # cm may still be smaller if mlxtend infers classes from data — fall back to sklearn
    if cm.shape[0] != len(class_names):
        cm = sk_cm(y_true, y_pred, labels=list(range(len(class_names))))

    fig, ax = mlxt_plot_cm(
        conf_mat=cm,
        class_names=class_names,
        colorbar=True,
        figsize=(max(6, len(class_names) * 1.5), max(5, len(class_names) * 1.4)),
    )
    ax.set_title("Confusion Matrix", fontsize=14, pad=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"📊 Saved confusion matrix → {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

# ============================================================
# Save prediction clips
# ============================================================

def save_prediction_clips(
    model: torch.nn.Module,
    dataset,
    class_names: list[str],
    device: torch.device,
    exp_dir: Path,
    num_samples: int = 8,
    fps: int = 8,
) -> list[dict]:
    """
    Sample random clips from the dataset, run inference, and save each as an MP4
    into experiments/<timestamp>/correct/ or experiments/<timestamp>/wrong/
    with filename: <original_stem>_true-<cls>_pred-<cls>.mp4

    Args:
        model:       Trained model in eval mode.
        dataset:     Subset/dataset to sample from (items are (T,C,H,W) tensors).
        class_names: Ordered class name strings.
        device:      Inference device.
        exp_dir:     Root experiment directory (Path).
        num_samples: How many clips to save.
        fps:         Frame rate for saved MP4.

    Returns:
        List of dicts with keys: path, true_label, pred_label, correct.
    """
    correct_dir = exp_dir / "correct"
    wrong_dir   = exp_dir / "wrong"
    correct_dir.mkdir(exist_ok=True)
    wrong_dir.mkdir(exist_ok=True)

    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    saved = []

    with torch.inference_mode():
        for idx in indices:
            frames, true_label = dataset[idx]          # (T, C, H, W) float [0,1]

            logits = model(frames.unsqueeze(0).to(device))
            pred_label = logits.argmax(dim=1).item()

            true_name = class_names[true_label]
            pred_name = class_names[pred_label]
            correct   = pred_label == true_label

            # torchvision.io.write_video expects (T, H, W, C) uint8
            clip_uint8 = (frames * 255).byte().permute(0, 2, 3, 1).cpu()

            # Try to recover the original filename from the underlying dataset
            try:
                # random_split wraps in Subset; .dataset + .indices lets us trace back
                real_idx = dataset.indices[idx] if hasattr(dataset, "indices") else idx
                stem = Path(dataset.dataset.samples[real_idx][0]).stem
            except Exception:
                stem = f"sample_{idx:04d}"

            fname  = f"{stem}_true-{true_name}_pred-{pred_name}.mp4"
            out_path = (correct_dir if correct else wrong_dir) / fname

            torchvision.io.write_video(
                str(out_path),
                clip_uint8,
                fps=fps,
            )

            saved.append({
                "path":        str(out_path),
                "true_label":  true_name,
                "pred_label":  pred_name,
                "correct":     correct,
            })

            mark = "✅" if correct else "❌"
            print(f"  {mark} {fname}")

    return saved
