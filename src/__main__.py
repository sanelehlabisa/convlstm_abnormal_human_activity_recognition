from pathlib import Path
import subprocess
import shutil

dataset_dir = Path("dataset")
noisy_dir = Path("dataset_noisy")
clean_dir = Path("dataset_clean")

clean_dir.mkdir(exist_ok=True)
noisy_dir.mkdir(exist_ok=True)

for video in dataset_dir.glob("*/*"):
    cls = video.parent.name
    (clean_dir / cls).mkdir(parents=True, exist_ok=True)
    (noisy_dir / cls).mkdir(parents=True, exist_ok=True)

    out_path = clean_dir / cls / video.name

    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", str(video),
        "-map", "0:v:0",
        "-vsync", "0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-y",
        str(out_path)
    ]

    result = subprocess.run(cmd, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"❌ Corrupted → {video.name}")
        shutil.move(video, noisy_dir / cls / video.name)
    else:
        print(f"✅ Cleaned → {video.name}")
