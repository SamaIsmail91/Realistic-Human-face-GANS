"""
evaluate.py
===========
Quantitative + visual evaluation of a trained GAN.

1. FID (Frechet Inception Distance) -- the standard metric for GAN image
   quality. Lower is better. Roughly:
       < 10   near-indistinguishable from real data (rare, needs lots of data+compute)
       10-30  clearly face-like, some artifacts
       30-60  recognizable faces but noticeable blur/artifacts
       > 60   still noisy / early training
   FID needs a reasonable sample size (>= 1-2k images on each side) to be
   meaningful -- with too few images the number is noisy and misleadingly high.

2. Loss curve plot from the CSV log written by train.py.

3. A "progress grid" that stitches together saved sample grids from
   different points in training so you can visually see the model improve.

Examples
--------
python evaluate.py --mode fid --checkpoint outputs/checkpoints/ckpt_epoch0100.pt \
    --real_dir data/raw_faces --num_samples 2000

python evaluate.py --mode losses --csv outputs/logs/losses.csv

python evaluate.py --mode progress --samples_dir outputs/samples
"""

import os
import sys
import glob
import argparse

import torch
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data import IMG_EXTENSIONS
from src.utils import plot_losses


def compute_fid(checkpoint_path, real_dir, num_samples=2000, batch_size=32, device=None):
    """
    Computes FID between real images in `real_dir` and images freshly
    sampled from the trained generator. Uses torchmetrics' FID
    implementation (InceptionV3 features under the hood).
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    from generate import load_generator

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G, latent_dim = load_generator(checkpoint_path, device)

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # ---- feed real images ----
    paths = []
    for ext in IMG_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(real_dir, "**", f"*{ext}"), recursive=True))
    if len(paths) == 0:
        raise FileNotFoundError(f"No images found under {real_dir}")
    paths = paths[:num_samples]

    tfm = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    print(f"Feeding {len(paths)} real images into FID...")
    with torch.no_grad():
        batch = []
        for p in paths:
            img = tfm(Image.open(p).convert("RGB"))
            batch.append(img)
            if len(batch) == batch_size:
                fid.update(torch.stack(batch).to(device), real=True)
                batch = []
        if batch:
            fid.update(torch.stack(batch).to(device), real=True)

    # ---- feed generated images ----
    print(f"Sampling {len(paths)} fake images from the generator...")
    resize = transforms.Resize((299, 299))
    with torch.no_grad():
        remaining = len(paths)
        while remaining > 0:
            n = min(batch_size, remaining)
            z = torch.randn(n, latent_dim, device=device)
            fake = G(z)
            fake = (fake + 1) / 2  # [-1,1] -> [0,1]
            fake = torch.stack([resize(img) for img in fake])
            fid.update(fake.to(device), real=False)
            remaining -= n

    score = fid.compute().item()
    print(f"\nFID score: {score:.3f}  (lower is better)")
    return score


def build_progress_grid(samples_dir, out_path, max_panels=8):
    """Stitch a handful of saved sample grids (evenly spaced over training)
    into one wide image so you can see quality improve left -> right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(glob.glob(os.path.join(samples_dir, "step_*.png")))
    if not files:
        raise FileNotFoundError(f"No sample grids found in {samples_dir}")

    if len(files) > max_panels:
        idx = [int(round(i * (len(files) - 1) / (max_panels - 1))) for i in range(max_panels)]
        files = [files[i] for i in sorted(set(idx))]

    fig, axes = plt.subplots(1, len(files), figsize=(4 * len(files), 4))
    if len(files) == 1:
        axes = [axes]
    for ax, f in zip(axes, files):
        step = os.path.basename(f).replace("step_", "").replace(".png", "")
        ax.imshow(Image.open(f))
        ax.set_title(f"step {int(step):,}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training-progress figure -> {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained face GAN")
    p.add_argument("--mode", choices=["fid", "losses", "progress"], required=True)
    p.add_argument("--checkpoint", help="[fid] path to a .pt checkpoint")
    p.add_argument("--real_dir", help="[fid] folder of real images to compare against")
    p.add_argument("--num_samples", type=int, default=2000, help="[fid] images to use on each side")
    p.add_argument("--csv", default="outputs/logs/losses.csv", help="[losses] path to losses.csv")
    p.add_argument("--samples_dir", default="outputs/samples", help="[progress] folder of saved sample grids")
    p.add_argument("--output_dir", default="outputs/eval")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "fid":
        if not args.checkpoint or not args.real_dir:
            raise SystemExit("--checkpoint and --real_dir are required for --mode fid")
        compute_fid(args.checkpoint, args.real_dir, num_samples=args.num_samples)

    elif args.mode == "losses":
        out = plot_losses(args.csv, os.path.join(args.output_dir, "loss_curve.png"))
        print(f"Saved loss curve -> {out}")

    elif args.mode == "progress":
        build_progress_grid(args.samples_dir, os.path.join(args.output_dir, "training_progress.png"))


if __name__ == "__main__":
    main()
