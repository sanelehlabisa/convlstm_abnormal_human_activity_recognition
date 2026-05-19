"""
train.py

Training script for ConvLSTM-based Abnormal Human Activity Recognition (AHAR).

Author: Sanele Hlabisa

python -m src.train \
    --dataset_dir "datasets/violence-detection-dataset" \
    --model_dir "models" \
    --checkpoint_path "models/best_model.pth" \
    --resume \
    --finetune_full \
    --batch_size 32 \
    --epochs 64 \
    --sequence_length 32 \
    --height 64 \
    --width 64 \
    --aug_copies 4 \
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
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

from .dataset import AHARDataset, CachedAHARDataset
from .model import ConvLSTMModel
from .utils import plot_training_curves, save_model, save_prediction_clips

parser = argparse.ArgumentParser(description="Train ConvLSTM for AHAR")
parser.add_argument("--dataset_dir", type=str, default="datasets/abnormal_activities")
parser.add_argument("--model_dir", type=str, default="models")
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume training same dataset, no layer changes",
)
parser.add_argument(
    "--finetune_last", action="store_true", help="Freeze all except last layer (fc2)"
)
parser.add_argument(
    "--finetune_full",
    action="store_true",
    help="Load weights, unfreeze everything, train all layers",
)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--sequence_length", type=int, default=32)
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--height", type=int, default=128)
parser.add_argument("--aug_copies", type=int, default=4)
parser.add_argument("--train_ratio", type=float, default=0.7)
parser.add_argument("--val_ratio", type=float, default=0.1)
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
    torch.backends.cudnn.benchmark = True
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

    # Augmentation for training only
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.3, 0.05)], p=0.6
            ),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.3),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            # transforms.TrivialAugmentWide(),  # strong random single aug on top
        ]
    )

    # Use cached dataset if small enough (decodes once, serves from RAM)
    DatasetClass = CachedAHARDataset if len(dataset) <= 2000 else AHARDataset
    if DatasetClass is CachedAHARDataset:
        print(f"⚡ Small dataset detected - using RAM cache for fast loading")

    # Replace the three dataset constructions
    dataset = DatasetClass(
        args.dataset_dir, args.sequence_length, (args.width, args.height)
    )
    base_ds = DatasetClass(
        args.dataset_dir, args.sequence_length, (args.width, args.height)
    )
    aug_ds = DatasetClass(
        args.dataset_dir,
        args.sequence_length,
        (args.width, args.height),
        transform=train_transform,
    )

    train_indices = train_set.indices
    clean_subset = torch.utils.data.Subset(base_ds, train_indices)
    aug_subsets = [
        torch.utils.data.Subset(aug_ds, train_indices) for _ in range(args.aug_copies)
    ]
    combined_train = torch.utils.data.ConcatDataset([clean_subset] + aug_subsets)
    print(f"📈 Train expanded: {len(train_indices)} - {len(combined_train)} samples")

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(combined_train, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kw)

    model = ConvLSTMModel(num_classes, input_shape=(3, args.height, args.width)).to(
        device
    )

    # ---- Checkpoint loading ----
    if args.checkpoint_path and Path(args.checkpoint_path).is_file():
        import zipfile

        if not zipfile.is_zipfile(args.checkpoint_path):
            print(f"❌ Checkpoint corrupted - starting fresh")
        else:
            print(f"⏳ Loading: {args.checkpoint_path}")
            checkpoint = torch.load(
                args.checkpoint_path, map_location=device, weights_only=True
            )
            ckpt_classes = checkpoint["model_state_dict"]["fc2.weight"].shape[0]

            loaded_model = ConvLSTMModel(
                ckpt_classes, input_shape=(3, args.height, args.width)
            ).to(device)
            loaded_model.load_state_dict(checkpoint["model_state_dict"])
            total_params = sum(p.numel() for p in loaded_model.parameters())
            print(
                f"✅ Loaded epoch={checkpoint['epoch']} | classes={ckpt_classes} | params={total_params:,}"
            )

            # Replace output layer if dataset has different classes
            if ckpt_classes != num_classes:
                loaded_model.fc2 = nn.Linear(
                    loaded_model.fc2.in_features, num_classes
                ).to(device)
                print(
                    f"🔁 Output layer replaced: {ckpt_classes} - {num_classes} classes"
                )

            if args.resume:
                # Continue training everything as-is, no freezing
                for p in loaded_model.parameters():
                    p.requires_grad = True
                print("▶️  Resuming - all layers trainable")

            elif args.finetune_last:
                # Freeze all except fc2
                fc2_ids = {id(p) for p in loaded_model.fc2.parameters()}
                for p in loaded_model.parameters():
                    p.requires_grad = id(p) in fc2_ids
                frozen = sum(
                    1 for p in loaded_model.parameters() if not p.requires_grad
                )
                trainable = sum(1 for p in loaded_model.parameters() if p.requires_grad)
                print(f"🔒 Frozen: {frozen} | 🔓 Trainable (fc2 only): {trainable}")

            elif args.finetune_full:
                # Load weights, unfreeze everything
                for p in loaded_model.parameters():
                    p.requires_grad = True
                print("🔓 Fine-tuning all layers")

            model = loaded_model
    else:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"⚠️  No checkpoint - scratch | params={total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    acc_fn = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(
        device
    )

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_loss = float("inf")
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    print("🚀 Training...")
    start = timer()

    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n🧠 Epoch {epoch+1}/{args.epochs}  lr={current_lr:.2e}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, acc_fn, device
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, acc_fn.clone(), device
        )
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(
            f"  Loss -> Train: {train_loss:.4f} Val: {val_loss:.4f} | Acc -> Train: {train_acc:.4f} Val: {val_acc:.4f}"
        )

        # Save only when val loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = str(
                Path(args.model_dir) / f"{Path(args.dataset_dir).name}_best_model.pth"
            )
            save_model(model, optimizer, epoch, val_loss, checkpoint_path=best_path)
            print(f"  ⭐ Best model updated (val_loss={val_loss:.4f})")

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
        model, test_set, dataset.class_names, device, out_dir, num_samples=8
    )


if __name__ == "__main__":
    main()
