"""
experiments.py

Grid search over model configurations to find best architecture.
Uses small image size for speed. Reports val loss, test acc, param count,
and overfitting gap (train_acc - val_acc). Saves results to JSON.

Author: Sanele Hlabisa

python -m src.experiments \
    --dataset_dir "datasets/processed/frames_abnormal_activities" \
    --epochs 20 \
    --sequence_length 16 \
    --height 32 \
    --width 32 \
    --aug_copies 2
"""

from __future__ import annotations

import argparse
import json
import itertools
from pathlib import Path
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .dataset import AHARDataset
from .model import ConvLSTMCustom

parser = argparse.ArgumentParser(description="Architecture search for ConvLSTM AHAR")
parser.add_argument("--dataset_dir", type=str, default="datasets/abnormal_activities")
parser.add_argument("--results_dir", type=str, default="experiments/grid_search")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--sequence_length", type=int, default=16)
parser.add_argument("--height", type=int, default=32)
parser.add_argument("--width", type=int, default=32)
parser.add_argument("--aug_copies", type=int, default=2)
parser.add_argument("--train_ratio", type=float, default=0.7)
parser.add_argument("--val_ratio", type=float, default=0.1)
parser.add_argument("--num_workers", type=int, default=2)


def _train(model, loader, criterion, optimizer, acc_fn, device):
    """
    Runs a single training epoch and calculates the average loss and accuracy.

    Parameters:
        model (torch.nn.Module): The neural network model being trained.
        loader (DataLoader): The data loader providing batches of training data.
        criterion (torch.nn.Module): The loss function used to calculate the error.
        optimizer (torch.optim.Optimizer): The optimizer updating the model weights.
        acc_fn (torchmetrics.Metric): The function used to calculate accuracy.
        device (torch.device): The hardware device (CPU or GPU) running the calculations.

    Returns:
        metrics (tuple): A tuple containing the average loss and average accuracy for the epoch.
    """
    model.train()
    total_loss = total_acc = 0.0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += acc_fn(logits.argmax(dim=1), y).item()
    return total_loss / len(loader), total_acc / len(loader)


@torch.inference_mode()
def _validate(model, loader, criterion, acc_fn, device):
    """
    Evaluates the model on a validation or test dataset without updating weights.

    Parameters:
        model (torch.nn.Module): The neural network model being evaluated.
        loader (DataLoader): The data loader providing batches of evaluation data.
        criterion (torch.nn.Module): The loss function used to calculate the error.
        acc_fn (torchmetrics.Metric): The function used to calculate accuracy.
        device (torch.device): The hardware device (CPU or GPU) running the calculations.

    Returns:
        metrics (tuple): A tuple containing the average loss and average accuracy.
    """
    model.eval()
    total_loss = total_acc = 0.0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item()
        total_acc += acc_fn(logits.argmax(dim=1), y).item()
    return total_loss / len(loader), total_acc / len(loader)


def _overfit_score(train_accs: list[float], val_accs: list[float]) -> float:
    """
    Calculates an overfitting score by analyzing the gap between training and validation accuracy.

    Parameters:
        train_accs (list[float]): A list of training accuracies over all epochs.
        val_accs (list[float]): A list of validation accuracies over all epochs.

    Returns:
        score (float): A positive score indicating the degree of overfitting (higher is worse).
    """
    gaps = [t - v for t, v in zip(train_accs, val_accs)]
    if len(gaps) < 2:
        return gaps[-1] if gaps else 0.0
    gradients = [gaps[i + 1] - gaps[i] for i in range(len(gaps) - 1)]
    return sum(gradients) / len(gradients)


def main() -> None:
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset = AHARDataset(
        args.dataset_dir, args.sequence_length, (args.width, args.height)
    )
    num_classes = dataset.num_classes
    print(
        f"Loaded {len(dataset)} samples | {num_classes} classes: {dataset.class_names}"
    )

    n_total = len(dataset)
    n_train = int(args.train_ratio * n_total)
    n_val = int(args.val_ratio * n_total)
    n_test = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    # Upgraded robust transform pipeline
    aug_ds = AHARDataset(
        args.dataset_dir,
        args.sequence_length,
        (args.width, args.height),
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomApply(
                    [
                        transforms.RandomAffine(
                            degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
                        )
                    ],
                    p=0.5,
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            ]
        ),
    )

    base_subset = torch.utils.data.Subset(dataset, train_set.indices)
    aug_subsets = [
        torch.utils.data.Subset(aug_ds, train_set.indices)
        for _ in range(args.aug_copies)
    ]
    combined = torch.utils.data.ConcatDataset([base_subset] + aug_subsets)

    loader_kw = dict(
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True
    )
    train_loader = DataLoader(combined, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kw)

    custom_configs = {
        "Balanced_Medium": [16, 32, 16, 128],  # ~2.1M params
        "Balanced_Small": [16, 32, 8, 128],  # ~1.0M params
        "BigBase_SmallHead": [32, 64, 8, 64],  # ~700k params
        "SmallBase_BigHead": [8, 16, 16, 128],  # ~2.1M params
        "Heavy_LSTM": [16, 128, 16, 64],  # ~1.7M params
        "Heavy_PostConv": [16, 32, 32, 64],  # ~2.1M params
        "Funnel": [32, 32, 8, 128],  # ~1.0M params
        "Bottleneck": [32, 64, 4, 256],  # ~1.3M params
    }

    print(f"\nRunning {len(custom_configs)} custom configurations...\n")

    all_results = []

    for i, (model_name, filters) in enumerate(custom_configs.items()):
        cfg = {
            "model_type": model_name,
            "filters": filters,
            "optimizer": "adam",
            "learning_rate": 0.001,
        }
        print(f"[{i+1}/{len(custom_configs)}] {cfg}")

        model = ConvLSTMCustom(
            num_classes, (3, args.height, args.width), filters=filters
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters())

        # Locked to Adam and 0.001
        opt = optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        acc_fn = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(
            device
        )

        train_accs, val_accs, val_losses = [], [], []
        t0 = timer()

        for epoch in tqdm(range(args.epochs), leave=False, desc="epochs"):
            _, tr_acc = _train(model, train_loader, criterion, opt, acc_fn, device)
            vl_loss, vl_acc = _validate(
                model, val_loader, criterion, acc_fn.clone(), device
            )
            train_accs.append(tr_acc)
            val_accs.append(vl_acc)
            val_losses.append(vl_loss)

        elapsed = timer() - t0
        best_val_loss = min(val_losses)
        best_val_acc = max(val_accs)
        overfit_score = _overfit_score(train_accs, val_accs)

        _, test_acc = _validate(model, test_loader, criterion, acc_fn.clone(), device)

        result = {
            "config": cfg,
            "num_params": num_params,
            "best_val_loss": round(best_val_loss, 6),
            "best_val_acc": round(best_val_acc, 4),
            "test_acc": round(test_acc, 4),
            "overfit_score": round(overfit_score, 4),
            "train_time_s": round(elapsed, 1),
        }
        all_results.append(result)

        print(
            f"  val_loss={best_val_loss:.4f}  val_acc={best_val_acc:.4f}  "
            f"test_acc={test_acc:.4f}  overfit={overfit_score:+.4f}  "
            f"params={num_params:,}  time={elapsed:.0f}s"
        )

    stable = [r for r in all_results if r["overfit_score"] < 0.05]
    ranked = sorted(
        stable if stable else all_results,
        key=lambda r: (-r["best_val_acc"], r["num_params"]),
    )

    print(f"\nTop 5 configurations (stable + best val acc):")
    print("-" * 80)
    for r in ranked[:5]:
        print(
            f"  val_acc={r['best_val_acc']:.4f}  test_acc={r['test_acc']:.4f}  "
            f"overfit={r['overfit_score']:+.4f}  params={r['num_params']:,}"
        )
        print(f"    {r['config']}")

    out_path = results_dir / "grid_search_results.json"
    with open(out_path, "w") as f:
        json.dump({"best": ranked[:5], "all": all_results}, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
