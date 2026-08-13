"""
losses.py
=========
Three interchangeable adversarial loss formulations (set via config.loss_type):

  hinge    - Hinge loss (Lim & Ye 2017 / used in SAGAN, BigGAN).
             Empirically the most stable of the three for this architecture;
             the default.
  bce      - Classic DCGAN non-saturating BCE loss with one-sided label
             smoothing on the real labels (0.9 instead of 1.0) to keep the
             discriminator from becoming overconfident.
  wgan_gp  - Wasserstein loss with gradient penalty (Gulrajani et al. 2017).
             No spectral norm needed at the same time in theory, but we keep
             it on here for extra stability; d_steps_per_g_step should be
             raised (e.g. 5) if you switch to this mode.
"""

import torch
import torch.nn.functional as F


def discriminator_loss(real_logits, fake_logits, loss_type="hinge", label_smoothing=0.9):
    if loss_type == "hinge":
        loss_real = F.relu(1.0 - real_logits).mean()
        loss_fake = F.relu(1.0 + fake_logits).mean()
        return loss_real + loss_fake

    elif loss_type == "bce":
        real_targets = torch.full_like(real_logits, label_smoothing)
        fake_targets = torch.zeros_like(fake_logits)
        loss_real = F.binary_cross_entropy_with_logits(real_logits, real_targets)
        loss_fake = F.binary_cross_entropy_with_logits(fake_logits, fake_targets)
        return loss_real + loss_fake

    elif loss_type == "wgan_gp":
        return fake_logits.mean() - real_logits.mean()

    raise ValueError(f"Unknown loss_type: {loss_type}")


def generator_loss(fake_logits, loss_type="hinge"):
    if loss_type == "hinge" or loss_type == "wgan_gp":
        return -fake_logits.mean()
    elif loss_type == "bce":
        real_targets = torch.ones_like(fake_logits)
        return F.binary_cross_entropy_with_logits(fake_logits, real_targets)
    raise ValueError(f"Unknown loss_type: {loss_type}")


def gradient_penalty(discriminator, real_images, fake_images, device):
    """R1-style interpolated gradient penalty for WGAN-GP."""
    batch_size = real_images.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
    d_interpolates = discriminator(interpolates)

    grad_outputs = torch.ones_like(d_interpolates, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty
