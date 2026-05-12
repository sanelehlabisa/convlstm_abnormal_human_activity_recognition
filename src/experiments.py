"""
experiments.py

Grid search over model configurations to find best architecture.
Uses small image size for speed. Reports val loss, test acc, param count,
and overfitting gap (train_acc - val_acc). Saves results to JSON.

Author: Sanele Hlabisa

python -m src.experiments \
    --dataset_dir "datasets/violence-detection-dataset" \
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
from .model import ConvLSTMModel, ConvLSTM2D

parser = argparse.ArgumentParser(description="Architecture search for ConvLSTM AHAR")
parser.add_argument("--dataset_dir",     type=str,   default="datasets/abnormal_activities")
parser.add_argument("--results_dir",     type=str,   default="experiments/grid_search")
parser.add_argument("--epochs",          type=int,   default=20)
parser.add_argument("--batch_size",      type=int,   default=16)
parser.add_argument("--sequence_length", type=int,   default=16)
parser.add_argument("--height",          type=int,   default=32)
parser.add_argument("--width",           type=int,   default=32)
parser.add_argument("--aug_copies",      type=int,   default=2)
parser.add_argument("--train_ratio",     type=float, default=0.7)
parser.add_argument("--val_ratio",       type=float, default=0.1)
parser.add_argument("--num_workers",     type=int,   default=2)


# Configurable model - extends base arch with variable layers

class ConfigurableConvLSTMModel(nn.Module):
    """
    Same arch as paper but with configurable:
      - convlstm_filters: filters in ConvLSTM2D layer
      - dense_units:      neurons in Dense(256) layer
      - dropout:          dropout rate at both dropout layers
      - extra_conv:       whether to add an extra TimeDistributed conv before ConvLSTM
    """

    def __init__(
        self,
        num_classes: int,
        input_shape: tuple[int, int, int] = (3, 32, 32),
        convlstm_filters: int = 64,
        dense_units: int = 256,
        dropout: float = 0.5,
        extra_conv: bool = False,
    ) -> None:
        super().__init__()

        C, H, W = input_shape

        # TimeDistributed Conv2D(16)
        self.td_conv   = nn.Conv2d(C, 16, kernel_size=3, padding=1)
        self.extra_conv = None

        td_out_channels = 16
        if extra_conv:
            self.extra_conv = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            td_out_channels = 32

        # ConvLSTM2D
        self.convlstm = ConvLSTM2D(in_channels=td_out_channels, filters=convlstm_filters, kernel_size=3)

        # BatchNorm → Conv2D(16) → Dropout → Flatten → Dense → Dropout → Output
        self.bn        = nn.BatchNorm2d(convlstm_filters)
        self.conv_post = nn.Conv2d(convlstm_filters, 16, kernel_size=3, padding=1)
        self.dropout1  = nn.Dropout(dropout)
        self.flatten   = nn.Flatten()
        self.fc1       = nn.Linear(16 * H * W, dense_units)
        self.dropout2  = nn.Dropout(dropout)
        self.fc2       = nn.Linear(dense_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        x = torch.relu(self.td_conv(x))
        if self.extra_conv is not None:
            x = torch.relu(self.extra_conv(x))
        x = x.view(B, T, -1, H, W)

        x = self.convlstm(x)
        x = self.bn(x)
        x = torch.relu(self.conv_post(x))
        x = self.dropout1(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


# Train / validate loops (minimal - no saving)

def _train(model, loader, criterion, optimizer, acc_fn, device):
    model.train()
    total_loss = total_acc = 0.0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        loss   = criterion(logits, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
        total_acc  += acc_fn(logits.argmax(dim=1), y).item()
    return total_loss / len(loader), total_acc / len(loader)


@torch.inference_mode()
def _validate(model, loader, criterion, acc_fn, device):
    model.eval()
    total_loss = total_acc = 0.0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(X)
        loss   = criterion(logits, y)
        total_loss += loss.item()
        total_acc  += acc_fn(logits.argmax(dim=1), y).item()
    return total_loss / len(loader), total_acc / len(loader)


# Overfitting score: average slope of (train_acc - val_acc)
# Positive and growing = overfitting

def _overfit_score(train_accs: list[float], val_accs: list[float]) -> float:
    gaps = [t - v for t, v in zip(train_accs, val_accs)]
    if len(gaps) < 2:
        return gaps[-1] if gaps else 0.0
    # Average gradient of the gap - rising gap = overfitting
    gradients = [gaps[i+1] - gaps[i] for i in range(len(gaps) - 1)]
    return sum(gradients) / len(gradients)


# Main

def main() -> None:
    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"🖥  Device: {device}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Dataset (shared across all configs) ----
    dataset     = AHARDataset(args.dataset_dir, args.sequence_length, (args.width, args.height))
    num_classes = dataset.num_classes
    print(f"📦 {len(dataset)} samples | {num_classes} classes: {dataset.class_names}")

    n_total = len(dataset)
    n_train = int(args.train_ratio * n_total)
    n_val   = int(args.val_ratio   * n_total)
    n_test  = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    # Augmented train set
    aug_ds = AHARDataset(
        args.dataset_dir, args.sequence_length, (args.width, args.height),
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.2)], p=0.5),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.3),
        ])
    )
    base_subset = torch.utils.data.Subset(dataset, train_set.indices)
    aug_subsets = [torch.utils.data.Subset(aug_ds, train_set.indices) for _ in range(args.aug_copies)]
    combined    = torch.utils.data.ConcatDataset([base_subset] + aug_subsets)

    loader_kw    = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(combined,  shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_set,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_set,  shuffle=False, **loader_kw)

    # ---- Search space ----
    search_space = {
        "convlstm_filters": [32, 64, 128],
        "dense_units":      [128, 256, 512],
        "dropout":          [0.3, 0.5],
        "extra_conv":       [False, True],
        "optimizer":        ["adam", "sgd"],
        "learning_rate":    [1e-3, 1e-4],
    }

    configs = list(itertools.product(*search_space.values()))
    keys    = list(search_space.keys())
    print(f"\n🔬 Running {len(configs)} configurations...\n")

    all_results = []

    for i, values in enumerate(configs):
        cfg = dict(zip(keys, values))
        print(f"[{i+1}/{len(configs)}] {cfg}")

        model = ConfigurableConvLSTMModel(
            num_classes=num_classes,
            input_shape=(3, args.height, args.width),
            convlstm_filters=cfg["convlstm_filters"],
            dense_units=cfg["dense_units"],
            dropout=cfg["dropout"],
            extra_conv=cfg["extra_conv"],
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters())

        if cfg["optimizer"] == "adam":
            opt = optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-4)
        else:
            opt = optim.SGD(model.parameters(), lr=cfg["learning_rate"], momentum=0.9, weight_decay=1e-4)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        acc_fn    = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)

        train_accs, val_accs, val_losses = [], [], []
        t0 = timer()

        for epoch in tqdm(range(args.epochs), leave=False, desc="epochs"):
            _, tr_acc  = _train(model, train_loader, criterion, opt, acc_fn, device)
            vl_loss, vl_acc = _validate(model, val_loader, criterion, acc_fn.clone(), device)
            train_accs.append(tr_acc)
            val_accs.append(vl_acc)
            val_losses.append(vl_loss)

        elapsed       = timer() - t0
        best_val_loss = min(val_losses)
        best_val_acc  = max(val_accs)
        overfit_score = _overfit_score(train_accs, val_accs)

        # Test accuracy with best val epoch weights (we use final weights as proxy)
        _, test_acc = _validate(model, test_loader, criterion, acc_fn.clone(), device)

        result = {
            "config":         cfg,
            "num_params":     num_params,
            "best_val_loss":  round(best_val_loss, 6),
            "best_val_acc":   round(best_val_acc, 4),
            "test_acc":       round(test_acc, 4),
            "overfit_score":  round(overfit_score, 4),  # lower = less overfitting
            "train_time_s":   round(elapsed, 1),
        }
        all_results.append(result)

        print(
            f"  val_loss={best_val_loss:.4f}  val_acc={best_val_acc:.4f}  "
            f"test_acc={test_acc:.4f}  overfit={overfit_score:+.4f}  "
            f"params={num_params:,}  time={elapsed:.0f}s"
        )

    # ---- Sort and report ----
    # Best = highest val_acc among configs with overfit_score < 0.05
    stable = [r for r in all_results if r["overfit_score"] < 0.05]
    ranked = sorted(stable if stable else all_results,
                    key=lambda r: (-r["best_val_acc"], r["num_params"]))

    print(f"\n🏆 Top 5 configurations (stable + best val acc):")
    print("─" * 80)
    for r in ranked[:5]:
        print(
            f"  val_acc={r['best_val_acc']:.4f}  test_acc={r['test_acc']:.4f}  "
            f"overfit={r['overfit_score']:+.4f}  params={r['num_params']:,}")
        print(f"    {r['config']}")

    # Save all results
    out_path = results_dir / "grid_search_results.json"
    with open(out_path, "w") as f:
        json.dump({"best": ranked[:5], "all": all_results}, f, indent=2)
    print(f"\n📄 Full results → {out_path}")


if __name__ == "__main__":
    main()