"""
diffaugment.py
===============
Differentiable Augmentation for Data-Efficient GAN Training.

Idea (Zhao et al., NeurIPS 2020): apply light, differentiable augmentations
(color jitter, translation, cutout) to BOTH real and fake images before they
reach the discriminator. Because the same random augmentation is applied to
both distributions, the discriminator can't "cheat" by keying off the
augmentation itself, but the extra data variety greatly reduces overfitting
and helps the GAN converge with far fewer training images than a vanilla
DCGAN would need -- important for face datasets smaller than CelebA's ~200k.

Usage:
    x = DiffAugment(x, policy='color,translation,cutout')
"""

import torch
import torch.nn.functional as F


def rand_brightness(x):
    factor = (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5)
    return x + factor


def rand_saturation(x):
    factor = torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2
    x_mean = x.mean(dim=1, keepdim=True)
    return (x - x_mean) * factor + x_mean


def rand_contrast(x):
    factor = torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - x_mean) * factor + x_mean


def rand_translation(x, ratio=0.125):
    B, C, H, W = x.shape
    shift_h = int(H * ratio + 0.5)
    shift_w = int(W * ratio + 0.5)
    translation_h = torch.randint(-shift_h, shift_h + 1, (B,), device=x.device)
    translation_w = torch.randint(-shift_w, shift_w + 1, (B,), device=x.device)

    grid_b, grid_h, grid_w = torch.meshgrid(
        torch.arange(B, device=x.device),
        torch.arange(H, device=x.device),
        torch.arange(W, device=x.device),
        indexing="ij",
    )
    grid_h = torch.clamp(grid_h + translation_h.view(-1, 1, 1) + 1, 0, H + 1)
    grid_w = torch.clamp(grid_w + translation_w.view(-1, 1, 1) + 1, 0, W + 1)

    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1)[grid_b, grid_h, grid_w].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, ratio=0.5):
    B, C, H, W = x.shape
    cutout_h = int(H * ratio + 0.5)
    cutout_w = int(W * ratio + 0.5)

    offset_h = torch.randint(0, H + (1 - cutout_h % 2), (B,), device=x.device)
    offset_w = torch.randint(0, W + (1 - cutout_w % 2), (B,), device=x.device)

    grid_b, grid_h, grid_w = torch.meshgrid(
        torch.arange(B, device=x.device),
        torch.arange(cutout_h, device=x.device),
        torch.arange(cutout_w, device=x.device),
        indexing="ij",
    )
    grid_h = torch.clamp(grid_h + offset_h.view(-1, 1, 1) - cutout_h // 2, 0, H - 1)
    grid_w = torch.clamp(grid_w + offset_w.view(-1, 1, 1) - cutout_w // 2, 0, W - 1)

    mask = torch.ones(B, H, W, device=x.device)
    mask[grid_b, grid_h, grid_w] = 0
    return x * mask.unsqueeze(1)


AUGMENT_FNS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}


def DiffAugment(x, policy="color,translation,cutout"):
    if policy:
        for p in policy.split(","):
            for fn in AUGMENT_FNS[p.strip()]:
                x = fn(x)
        x = x.contiguous()
    return x
