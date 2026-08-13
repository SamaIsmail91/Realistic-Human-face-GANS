"""
utils.py
========
Small reusable helpers used by train.py / evaluate.py / generate.py:
  - set_seed              reproducibility
  - EMA                   exponential moving average of generator weights
  - save_checkpoint / load_checkpoint
  - save_sample_grid      write a PNG grid of generated faces
  - CSVLogger             plain-text loss log (in addition to TensorBoard)
  - plot_losses           loss-curve PNG from the CSV log
"""

import os
import csv
import random
import copy

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    """
    Keeps a shadow copy of the generator whose weights are an exponential
    moving average of the training weights:
        shadow = decay * shadow + (1 - decay) * current

    GAN training is noisy step-to-step; sampling from the EMA generator
    instead of the raw (still-oscillating) generator gives visibly sharper,
    more stable-looking faces. This is standard practice in StyleGAN,
    BigGAN, etc.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for shadow_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            shadow_p.mul_(self.decay).add_(model_p.detach(), alpha=1 - self.decay)
        for shadow_b, model_b in zip(self.shadow.buffers(), model.buffers()):
            shadow_b.copy_(model_b)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


def save_checkpoint(path, epoch, global_step, generator, discriminator,
                     ema, opt_g, opt_d, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "config": vars(config) if not isinstance(config, dict) else config,
    }, path)


def load_checkpoint(path, generator, discriminator, ema=None, opt_g=None, opt_d=None,
                     map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    generator.load_state_dict(ckpt["generator"])
    discriminator.load_state_dict(ckpt["discriminator"])
    if ema is not None and ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])
    if opt_g is not None and ckpt.get("opt_g") is not None:
        opt_g.load_state_dict(ckpt["opt_g"])
    if opt_d is not None and ckpt.get("opt_d") is not None:
        opt_d.load_state_dict(ckpt["opt_d"])
    return ckpt.get("epoch", 0), ckpt.get("global_step", 0)


def find_latest_checkpoint(checkpoints_dir):
    if not os.path.isdir(checkpoints_dir):
        return None
    ckpts = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pt")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoints_dir, f)))
    return os.path.join(checkpoints_dir, ckpts[-1])


@torch.no_grad()
def save_sample_grid(generator, fixed_noise, path, nrow=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    generator.eval()
    imgs = generator(fixed_noise).cpu()
    generator.train()
    imgs = (imgs + 1) / 2  # [-1,1] -> [0,1]
    grid = make_grid(imgs, nrow=nrow, padding=2)
    save_image(grid, path)
    return path


class CSVLogger:
    """Plain-text alternative/companion to TensorBoard -- easy to `pandas.read_csv` later."""

    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path)
        self.file = open(path, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if write_header:
            self.writer.writeheader()
            self.file.flush()

    def log(self, **kwargs):
        self.writer.writerow(kwargs)
        self.file.flush()

    def close(self):
        self.file.close()


def plot_losses(csv_path, out_path):
    """Read the CSV loss log and save a loss-curve PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    plt.figure(figsize=(9, 5))
    plt.plot(df["global_step"], df["loss_d"], label="Discriminator loss", alpha=0.8)
    plt.plot(df["global_step"], df["loss_g"], label="Generator loss", alpha=0.8)
    plt.xlabel("Generator step")
    plt.ylabel("Loss")
    plt.title("GAN Training Losses")
    plt.legend()
    plt.grid(alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
