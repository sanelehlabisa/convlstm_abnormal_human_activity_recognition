"""
demo.py

Quick demo: samples clips, runs inference, saves output videos.

Author: Sanele Hlabisa

python demo.py \
    --dataset_dir "datasets/abnormal_activities" \
    --checkpoint_path "models/best_model.pth" \
    --num_samples 8 \
    --sequence_length 32 \
    --height 64 \
    --width 64
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

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default="datasets/abnormal_activities")
parser.add_argument("--checkpoint_path", type=str, default="models/best_model.pth")
parser.add_argument("--num_samples", type=int, default=8)
parser.add_argument("--sequence_length", type=int, default=32)
parser.add_argument("--height", type=int, default=64)
parser.add_argument("--width", type=int, default=64)
parser.add_argument("--fps", type=int, default=8)
parser.add_argument("--seed", type=int, default=None)


def main() -> None:
    args = parser.parse_args()
    if args.seed:
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AHARDataset(
        args.dataset_dir, args.sequence_length, (args.height, args.width)
    )
    model = ConvLSTMModel(
        dataset.num_classes, input_shape=(3, args.height, args.width)
    ).to(device)
    model, _, epoch, loss = load_model(
        model, checkpoint_path=args.checkpoint_path, map_location=device
    )
    print(f"📂 Checkpoint → epoch={epoch}, loss={loss:.4f}")
    model.eval()

    out_dir = Path("outputs") / "demo_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    print("\n🎬 Predictions\n" + "─" * 48)

    with torch.inference_mode():
        for idx in indices:
            frames, true_label = dataset[idx]
            probs = torch.softmax(model(frames.unsqueeze(0).to(device)), dim=1)[0]
            pred_label = probs.argmax().item()
            pred_conf = probs[pred_label].item()

            true_name = dataset.class_names[true_label]
            pred_name = dataset.class_names[pred_label]
            correct = pred_label == true_label

            print(
                f"  {'✅' if correct else '❌'} true: {true_name:<18} pred: {pred_name:<18} conf: {pred_conf:.2f}"
            )

            stem = Path(dataset.samples[idx][0]).stem
            fname = f"{stem}_true-{true_name}_pred-{pred_name}_{'correct' if correct else 'wrong'}.mp4"
            clip = (frames * 255).byte().permute(0, 2, 3, 1).cpu()
            torchvision.io.write_video(str(out_dir / fname), clip, fps=args.fps)
            print(f"       💾 {out_dir / fname}")

    print(f"\n📁 Done → {out_dir}")


if __name__ == "__main__":
    main()
