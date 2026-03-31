"""
demo.py

Quick demo: randomly samples N clips from a dataset, runs inference,
prints multiclass prediction + binary verdict, and saves a visual grid.

Author: Sanele Hlabisa

python demo.py \
    --dataset_dir "datasets/abnormal_activities" \
    --model_dir   "models" \
    --num_samples 8 \
    --sequence_length 32 \
    --height 64 \
    --width  64
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torchvision

from src.dataset import AHARDataset
from src.model import ConvLSTMModel
from src.utils import load_model

# ============================================================
# Argument parser
# ============================================================

parser = argparse.ArgumentParser(description="Demo: random clip predictions")

parser.add_argument("--dataset_dir", type=str, default="dataset_clean")
parser.add_argument("--model_dir", type=str, default="models")
parser.add_argument("--num_samples", type=int, default=8)
parser.add_argument("--sequence_length", type=int, default=32)
parser.add_argument("--height", type=int, default=64)
parser.add_argument("--width", type=int, default=64)
parser.add_argument("--fps", type=int, default=8)
parser.add_argument(
    "--seed", type=int, default=None, help="Fix random seed for reproducible demo picks"
)

# ============================================================
# Main
# ============================================================


def main() -> None:
    """
    Main function

    Args:
        None

    Returns:
        None
    """
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Using device: {device}")

    # ---------------- Dataset ----------------
    dataset = AHARDataset(
        dataset_dir=args.dataset_dir,
        sequence_length=args.sequence_length,
        frame_size=(args.height, args.width),
    )

    # ---------------- Model ----------------
    model = ConvLSTMModel(
        num_classes=dataset.num_classes,
        input_shape=(3, args.height, args.width),
    ).to(device)

    model, _, epoch, ckpt_loss = load_model(
        model,
        base_path=args.model_dir,
        map_location=device,
    )
    print(f"📂 Checkpoint → epoch={epoch}, saved_loss={ckpt_loss:.4f}")

    model.eval()

    # ---------------- Sample & predict ----------------
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    out_dir = Path("outputs") / "demo_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n🎬 Predictions")
    print("─" * 48)

    with torch.inference_mode():
        for sample_idx in indices:
            frames, true_label = dataset[sample_idx]  # (T, C, H, W)
            logits = model(frames.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = probs.argmax().item()
            pred_conf = probs[pred_idx].item()

            true_name = dataset.class_names[true_label]
            pred_name = dataset.class_names[pred_idx]
            verdict = dataset.registry.detection_verdict(pred_name) or "unknown"
            correct = pred_idx == true_label

            # Console output
            mark = "✅" if correct else "❌"
            print(
                f"  {mark} true: {true_name:<18} "
                f"pred: {pred_name:<18} "
                f"conf: {pred_conf:.2f}  verdict: {verdict.upper()}"
            )

            # Save clip
            try:
                stem = Path(dataset.samples[sample_idx][0]).stem
            except Exception:
                stem = f"sample_{sample_idx:04d}"

            correct_str = "correct" if correct else "wrong"
            fname = f"{stem}_true-{true_name}_pred-{pred_name}_{correct_str}.mp4"
            out_path = out_dir / fname

            clip_uint8 = (frames * 255).byte().permute(0, 2, 3, 1).cpu()
            torchvision.io.write_video(str(out_path), clip_uint8, fps=args.fps)
            print(f"       💾 {out_path}")

    print(f"\n📁 Clips saved → {out_dir}")

if __name__ == "__main__":
    main()
