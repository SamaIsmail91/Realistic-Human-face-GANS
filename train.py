"""
train.py
========
Main training loop.

Example usage
-------------
# quick 64x64 run on your own folder of face images
python train.py --data_dir data/raw_faces --image_size 64 --epochs 100

# resume automatically from the latest checkpoint (default behaviour)
python train.py --data_dir data/raw_faces

# higher-quality 128x128 run with a bigger batch (needs a decent GPU)
python train.py --data_dir data/raw_faces --image_size 128 --batch_size 32 --epochs 200

# auto-download CelebA instead of using a local folder
python train.py --dataset celeba --data_dir data/celeba --image_size 64

Monitor training in real time with:
    tensorboard --logdir outputs/logs
"""

import os
import sys
import time
import argparse
import dataclasses

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config
from src.data import get_dataloader
from src.models import Generator, Discriminator, weights_init
from src.diffaugment import DiffAugment
from src.losses import discriminator_loss, generator_loss, gradient_penalty
from src.utils import (set_seed, EMA, save_checkpoint, load_checkpoint,
                        find_latest_checkpoint, save_sample_grid, CSVLogger)


def parse_args():
    defaults = Config()
    p = argparse.ArgumentParser(description="Train a face-generating GAN")
    for f in dataclasses.fields(defaults):
        if not f.init:
            continue
        val = getattr(defaults, f.name)
        arg_name = f"--{f.name}"
        if isinstance(val, bool):
            p.add_argument(arg_name, type=lambda x: str(x).lower() == "true", default=val)
        else:
            p.add_argument(arg_name, type=type(val), default=val)
    p.add_argument("--dataset", choices=["folder", "celeba"], default="folder")
    return p.parse_args()


def build_models(cfg, device):
    G = Generator(
        latent_dim=cfg.latent_dim,
        base_channels=cfg.g_base_channels,
        image_size=cfg.image_size,
        channels=cfg.channels,
        use_self_attention=cfg.use_self_attention,
    ).to(device)
    D = Discriminator(
        base_channels=cfg.d_base_channels,
        image_size=cfg.image_size,
        channels=cfg.channels,
        use_self_attention=cfg.use_self_attention,
        use_spectral_norm=cfg.use_spectral_norm,
    ).to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    return G, D


def train(cfg):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.mixed_precision and device.type == "cuda"
    print(f"Device: {device} | mixed precision: {use_amp}")

    os.makedirs(cfg.samples_dir, exist_ok=True)
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    os.makedirs(cfg.logs_dir, exist_ok=True)

    loader, dataset = get_dataloader(
        cfg.data_dir, image_size=cfg.image_size, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, dataset=getattr(cfg, "dataset", "folder"),
    )
    print(f"Dataset: {len(dataset)} images | {len(loader)} batches / epoch")

    G, D = build_models(cfg, device)
    ema = EMA(G, decay=cfg.ema_decay)

    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))

    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    fixed_noise = torch.randn(cfg.num_fixed_samples, cfg.latent_dim, device=device)

    start_epoch, global_step = 0, 0
    if cfg.resume:
        latest = find_latest_checkpoint(cfg.checkpoints_dir)
        if latest:
            start_epoch, global_step = load_checkpoint(
                latest, G, D, ema=ema, opt_g=opt_g, opt_d=opt_d, map_location=device)
            print(f"Resumed from {latest} (epoch {start_epoch}, step {global_step})")

    writer = SummaryWriter(log_dir=cfg.logs_dir)
    csv_logger = CSVLogger(
        os.path.join(cfg.logs_dir, "losses.csv"),
        fieldnames=["epoch", "global_step", "loss_d", "loss_g", "d_real", "d_fake"],
    )

    aug_policy = cfg.diffaugment_policy if cfg.use_diffaugment else ""

    def augment(x):
        return DiffAugment(x, policy=aug_policy) if aug_policy else x

    print(f"Starting training: {cfg.epochs} epochs from epoch {start_epoch}")
    for epoch in range(start_epoch, cfg.epochs):
        epoch_start = time.time()
        running_d, running_g, n_batches = 0.0, 0.0, 0

        for i, (real, _) in enumerate(loader):
            real = real.to(device, non_blocking=True)
            bsz = real.size(0)

            # =========================================================
            # 1) Train Discriminator (d_steps_per_g_step times)
            # =========================================================
            for _ in range(cfg.d_steps_per_g_step):
                opt_d.zero_grad(set_to_none=True)
                z = torch.randn(bsz, cfg.latent_dim, device=device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    fake = G(z).detach()
                    real_aug = augment(real)
                    fake_aug = augment(fake)
                    real_logits = D(real_aug)
                    fake_logits = D(fake_aug)
                    d_loss = discriminator_loss(
                        real_logits, fake_logits,
                        loss_type=cfg.loss_type, label_smoothing=cfg.label_smoothing)

                if cfg.loss_type == "wgan_gp":
                    # gradient penalty needs real fp32 grads -> compute outside autocast
                    gp = gradient_penalty(D, real_aug.float(), fake_aug.float(), device)
                    d_loss = d_loss + cfg.gp_lambda * gp

                scaler_d.scale(d_loss).backward()
                if cfg.grad_clip_norm > 0:
                    scaler_d.unscale_(opt_d)
                    nn.utils.clip_grad_norm_(D.parameters(), cfg.grad_clip_norm)
                scaler_d.step(opt_d)
                scaler_d.update()

            # =========================================================
            # 2) Train Generator
            # =========================================================
            opt_g.zero_grad(set_to_none=True)
            z = torch.randn(bsz, cfg.latent_dim, device=device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                fake = G(z)
                fake_aug = augment(fake)
                fake_logits_for_g = D(fake_aug)
                g_loss = generator_loss(fake_logits_for_g, loss_type=cfg.loss_type)

            scaler_g.scale(g_loss).backward()
            if cfg.grad_clip_norm > 0:
                scaler_g.unscale_(opt_g)
                nn.utils.clip_grad_norm_(G.parameters(), cfg.grad_clip_norm)
            scaler_g.step(opt_g)
            scaler_g.update()

            ema.update(G)

            global_step += 1
            running_d += d_loss.item()
            running_g += g_loss.item()
            n_batches += 1

            if global_step % cfg.log_every == 0:
                d_real_mean = torch.sigmoid(real_logits).mean().item() if cfg.loss_type == "bce" else real_logits.mean().item()
                d_fake_mean = torch.sigmoid(fake_logits).mean().item() if cfg.loss_type == "bce" else fake_logits.mean().item()
                print(f"[epoch {epoch+1}/{cfg.epochs}] step {global_step} "
                      f"| D_loss {d_loss.item():.3f} | G_loss {g_loss.item():.3f} "
                      f"| D(real) {d_real_mean:.3f} | D(fake) {d_fake_mean:.3f}")
                writer.add_scalar("loss/discriminator", d_loss.item(), global_step)
                writer.add_scalar("loss/generator", g_loss.item(), global_step)
                writer.add_scalar("logits/D_real", d_real_mean, global_step)
                writer.add_scalar("logits/D_fake", d_fake_mean, global_step)
                csv_logger.log(epoch=epoch + 1, global_step=global_step,
                                loss_d=d_loss.item(), loss_g=g_loss.item(),
                                d_real=d_real_mean, d_fake=d_fake_mean)

            if global_step % cfg.sample_every == 0:
                path = os.path.join(cfg.samples_dir, f"step_{global_step:07d}.png")
                save_sample_grid(ema.shadow, fixed_noise, path,
                                  nrow=int(cfg.num_fixed_samples ** 0.5))
                writer.add_images(
                    "samples/ema",
                    (ema.shadow(fixed_noise).clamp(-1, 1) + 1) / 2,
                    global_step,
                )

        elapsed = time.time() - epoch_start
        print(f"== Epoch {epoch+1}/{cfg.epochs} done in {elapsed:.1f}s | "
              f"avg D_loss {running_d/max(n_batches,1):.3f} | "
              f"avg G_loss {running_g/max(n_batches,1):.3f} ==")

        if (epoch + 1) % cfg.checkpoint_every == 0 or (epoch + 1) == cfg.epochs:
            ckpt_path = os.path.join(cfg.checkpoints_dir, f"ckpt_epoch{epoch+1:04d}.pt")
            save_checkpoint(ckpt_path, epoch + 1, global_step, G, D, ema, opt_g, opt_d, cfg)
            print(f"Saved checkpoint -> {ckpt_path}")

    writer.close()
    csv_logger.close()
    print("Training complete.")
    print(f"Samples: {cfg.samples_dir}")
    print(f"Checkpoints: {cfg.checkpoints_dir}")
    print(f"Loss log: {os.path.join(cfg.logs_dir, 'losses.csv')}")


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(**{k: v for k, v in vars(args).items() if k != "dataset"})
    cfg.dataset = args.dataset  # extra attribute, not part of the dataclass fields loop
    train(cfg)
