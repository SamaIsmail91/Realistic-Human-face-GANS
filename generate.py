"""
generate.py
===========
Generate new synthetic faces from a trained checkpoint.

Examples
--------
# 64 random faces as a single grid image
python generate.py --checkpoint outputs/checkpoints/ckpt_epoch0100.pt --num_images 64

# 16 individual PNG files instead of one grid
python generate.py --checkpoint outputs/checkpoints/ckpt_epoch0100.pt --num_images 16 --individual

# smooth latent-space interpolation between two random faces (great for a demo GIF)
python generate.py --checkpoint outputs/checkpoints/ckpt_epoch0100.pt --interpolate --steps 30

# reproducible output
python generate.py --checkpoint outputs/checkpoints/ckpt_epoch0100.pt --seed 123
"""

import os
import sys
import argparse

import torch
from torchvision.utils import make_grid, save_image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.models import Generator
from src.utils import set_seed


def load_generator(checkpoint_path, device, use_ema=True):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]

    G = Generator(
        latent_dim=cfg["latent_dim"],
        base_channels=cfg["g_base_channels"],
        image_size=cfg["image_size"],
        channels=cfg["channels"],
        use_self_attention=cfg["use_self_attention"],
    ).to(device)

    if use_ema and ckpt.get("ema") is not None:
        G.load_state_dict(ckpt["ema"])
        print("Loaded EMA (smoothed) generator weights.")
    else:
        G.load_state_dict(ckpt["generator"])
        print("Loaded raw (non-EMA) generator weights.")

    G.eval()
    return G, cfg["latent_dim"]


@torch.no_grad()
def generate_grid(G, latent_dim, num_images, device, out_path, nrow=None):
    z = torch.randn(num_images, latent_dim, device=device)
    imgs = G(z).cpu()
    imgs = (imgs + 1) / 2
    grid = make_grid(imgs, nrow=nrow or int(num_images ** 0.5))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_image(grid, out_path)
    return out_path


@torch.no_grad()
def generate_individual(G, latent_dim, num_images, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    z = torch.randn(num_images, latent_dim, device=device)
    imgs = G(z).cpu()
    imgs = (imgs + 1) / 2
    paths = []
    for i, img in enumerate(imgs):
        p = os.path.join(out_dir, f"face_{i:04d}.png")
        save_image(img, p)
        paths.append(p)
    return paths


@torch.no_grad()
def generate_interpolation(G, latent_dim, steps, device, out_path):
    """Spherical-ish linear interpolation between two random latent points."""
    z1 = torch.randn(1, latent_dim, device=device)
    z2 = torch.randn(1, latent_dim, device=device)
    alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1)
    z = (1 - alphas) * z1 + alphas * z2
    imgs = G(z).cpu()
    imgs = (imgs + 1) / 2
    grid = make_grid(imgs, nrow=steps)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_image(grid, out_path)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Generate faces from a trained GAN checkpoint")
    p.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint from train.py")
    p.add_argument("--num_images", type=int, default=64)
    p.add_argument("--output_dir", default="outputs/generated")
    p.add_argument("--individual", action="store_true", help="save separate PNGs instead of one grid")
    p.add_argument("--interpolate", action="store_true", help="latent-space interpolation instead of random samples")
    p.add_argument("--steps", type=int, default=30, help="frames for --interpolate")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no_ema", action="store_true", help="use raw generator weights instead of the EMA copy")
    args = p.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G, latent_dim = load_generator(args.checkpoint, device, use_ema=not args.no_ema)

    if args.interpolate:
        out_path = os.path.join(args.output_dir, "interpolation.png")
        path = generate_interpolation(G, latent_dim, args.steps, device, out_path)
        print(f"Saved interpolation grid -> {path}")
    elif args.individual:
        paths = generate_individual(G, latent_dim, args.num_images, device, args.output_dir)
        print(f"Saved {len(paths)} images -> {args.output_dir}/")
    else:
        out_path = os.path.join(args.output_dir, "generated_grid.png")
        path = generate_grid(G, latent_dim, args.num_images, device, out_path)
        print(f"Saved grid of {args.num_images} faces -> {path}")


if __name__ == "__main__":
    main()
