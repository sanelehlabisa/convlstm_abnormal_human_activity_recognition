"""
utils.py

Utility functions for AHAR: visualization, training plots, model persistence.

Author: Sanele Hlabisa
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torchvision
from mlxtend.evaluate import confusion_matrix as mlxt_cm
from mlxtend.plotting import plot_confusion_matrix as mlxt_plot_cm
from sklearn.metrics import confusion_matrix as sk_cm


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    dataset_name: str = "dataset",
    save_dir: str = "outputs",
    show: bool = False,
) -> str:
    """Plot and save loss + accuracy curves. Returns save path."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train")
    plt.plot(epochs, val_accs, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = str(Path(save_dir) / f"training_curves_{dataset_name}.png")
    plt.savefig(save_path, dpi=150)
    print(f"📊 Saved training curves → {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    dataset_name: str = "dataset",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot and save confusion matrix. Falls back to sklearn if mlxtend shape mismatch."""
    cm = mlxt_cm(y_target=y_true, y_predicted=y_pred, binary=False, positive_label=1)
    if cm.shape[0] != len(class_names):
        cm = sk_cm(y_true, y_pred, labels=list(range(len(class_names))))

    fig, ax = mlxt_plot_cm(
        conf_mat=cm,
        class_names=class_names,
        colorbar=True,
        figsize=(max(6, len(class_names) * 1.5), max(5, len(class_names) * 1.4)),
    )
    ax.set_title(f"Confusion Matrix — {dataset_name}", fontsize=14, pad=12)
    plt.tight_layout()

    if save_path:
        # Inject dataset name into filename
        p = Path(save_path)
        final_path = str(p.parent / f"{p.stem}_{dataset_name}{p.suffix}")
        plt.savefig(final_path, dpi=150)
        print(f"📊 Saved confusion matrix → {final_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_path: str,  # full path e.g. "models/checkpoint.pth"
    extra_meta: Optional[dict] = None,
) -> None:
    """Save model checkpoint to an explicit full path."""
    ckpt_path = Path(checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        ckpt_path,
    )

    meta = {
        "epoch": epoch,
        "loss": loss,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "timestamp": datetime.now().isoformat(),
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(ckpt_path.parent / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved checkpoint → {ckpt_path}")


def load_model(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    checkpoint_path: str = "models/best_model.pth",
    map_location: str = "cpu",
) -> tuple[torch.nn.Module, Optional[torch.optim.Optimizer], int, float]:
    """Load best_model.pth from base_path if available."""
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        print("⚠️  No checkpoint found — starting from scratch.")
        return model, optimizer, 0, float("inf")

    checkpoint = torch.load(ckpt, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))
    print(f"✅ Loaded checkpoint (epoch={epoch}, loss={loss:.4f})")
    return model, optimizer, epoch, loss


def save_prediction_clips(
    model: torch.nn.Module,
    dataset,
    class_names: list[str],
    device: torch.device,
    exp_dir: Path,
    num_samples: int = 8,
    fps: int = 8,
) -> list[dict]:
    """Save random sample clips into exp_dir/correct/ and exp_dir/wrong/."""
    (exp_dir / "correct").mkdir(exist_ok=True)
    (exp_dir / "wrong").mkdir(exist_ok=True)

    model.eval()
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    saved = []

    with torch.inference_mode():
        for idx in indices:
            frames, true_label = dataset[idx]
            logits = model(frames.unsqueeze(0).to(device))
            pred_label = logits.argmax(dim=1).item()
            correct = pred_label == true_label

            true_name = class_names[true_label]
            pred_name = class_names[pred_label]

            try:
                real_idx = dataset.indices[idx] if hasattr(dataset, "indices") else idx
                stem = Path(dataset.dataset.samples[real_idx][0]).stem
            except Exception:
                stem = f"sample_{idx:04d}"

            fname = f"{stem}_true-{true_name}_pred-{pred_name}.mp4"
            out_path = (exp_dir / "correct" if correct else exp_dir / "wrong") / fname
            clip = (frames * 255).byte().permute(0, 2, 3, 1).cpu()
            torchvision.io.write_video(str(out_path), clip, fps=fps)

            saved.append(
                {
                    "path": str(out_path),
                    "true": true_name,
                    "pred": pred_name,
                    "correct": correct,
                }
            )
            print(f"  {'✅' if correct else '❌'} {fname}")

    return saved
