"""
models.py
=========
Generator and Discriminator networks.

Architecture summary (a DCGAN backbone upgraded with modern GAN tricks):
  - Transposed-conv Generator / strided-conv Discriminator (DCGAN base)
  - Self-Attention layer (SAGAN, Zhang et al. 2018) at a mid-resolution stage
    so the network can model long-range dependencies (eyes/nose/mouth
    symmetry) instead of only local texture.
  - Spectral Normalization (Miyato et al. 2018) on every Discriminator layer
    for a Lipschitz-bounded, much more stable critic.
  - Hinge loss compatible outputs (raw logits, no sigmoid).
"""

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


def weights_init(m):
    """DCGAN-style weight initialization (N(0, 0.02))."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class SelfAttention(nn.Module):
    """
    SAGAN self-attention block.
    Lets every spatial position attend to every other spatial position,
    so the generator can enforce global facial structure (symmetric eyes,
    consistent lighting) rather than only local patterns.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # start as identity, let the network learn to use it
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q = self.query(x).view(B, -1, N).permute(0, 2, 1)   # B x N x C'
        k = self.key(x).view(B, -1, N)                       # B x C' x N
        attn = self.softmax(torch.bmm(q, k))                 # B x N x N

        v = self.value(x).view(B, C, N)                      # B x C x N
        out = torch.bmm(v, attn.permute(0, 2, 1))             # B x C x N
        out = out.view(B, C, H, W)

        return self.gamma * out + x


class GenBlock(nn.Module):
    """One upsampling block: doubles spatial resolution."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DisBlock(nn.Module):
    """One downsampling block: halves spatial resolution."""

    def __init__(self, in_ch, out_ch, use_spectral_norm=True, use_bn=True):
        super().__init__()
        # BatchNorm is usually skipped when spectral norm is on (SN already
        # controls scale); kept as an option for experimentation. Only omit
        # the conv bias when BatchNorm is actually being applied afterwards.
        apply_bn = use_bn and not use_spectral_norm
        conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not apply_bn)
        if use_spectral_norm:
            conv = spectral_norm(conv)
        layers = [conv]
        if apply_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    """
    Maps a latent vector z ~ N(0, I) of shape (B, latent_dim) to an
    RGB image in [-1, 1] of shape (B, 3, image_size, image_size).

    Supports image_size in {64, 128}.
    """

    def __init__(self, latent_dim=128, base_channels=64, image_size=64,
                 channels=3, use_self_attention=True):
        super().__init__()
        assert image_size in (64, 128), "image_size must be 64 or 128"
        self.latent_dim = latent_dim
        self.image_size = image_size

        # channel multipliers for each resolution, largest -> smallest channel count
        # 64px path : 4 -> 8 -> 16 -> 32 -> 64
        # 128px path: 4 -> 8 -> 16 -> 32 -> 64 -> 128
        mult = 16 if image_size == 64 else 32

        self.project = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * mult, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base_channels * mult),
            nn.ReLU(inplace=True),
        )  # (B, base*mult, 4, 4)

        blocks = []
        ch = base_channels * mult
        # keep upsampling until we reach base_channels (i.e. image_size/... resolution)
        resolution = 4
        while resolution < image_size:
            out_ch = max(base_channels, ch // 2)
            blocks.append(GenBlock(ch, out_ch))
            ch = out_ch
            resolution *= 2
            if use_self_attention and resolution == image_size // 2:
                # insert attention at the second-to-last resolution: rich enough
                # features, still cheap enough to compute (attention is O(N^2)).
                blocks.append(SelfAttention(ch))

        self.blocks = nn.Sequential(*blocks)
        self.to_rgb = nn.Sequential(
            nn.Conv2d(ch, channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = z.view(z.size(0), self.latent_dim, 1, 1)
        x = self.project(x)
        x = self.blocks(x)
        return self.to_rgb(x)


class Discriminator(nn.Module):
    """
    Maps an RGB image to a single real-valued logit (higher = "more real").
    No sigmoid at the output: hinge / wgan losses want raw logits, and BCE
    loss uses `nn.BCEWithLogitsLoss` which applies sigmoid internally.
    """

    def __init__(self, base_channels=64, image_size=64, channels=3,
                 use_self_attention=True, use_spectral_norm=True):
        super().__init__()
        assert image_size in (64, 128), "image_size must be 64 or 128"

        first_conv = nn.Conv2d(channels, base_channels, kernel_size=4, stride=2, padding=1)
        if use_spectral_norm:
            first_conv = spectral_norm(first_conv)
        blocks = [first_conv, nn.LeakyReLU(0.2, inplace=True)]

        ch = base_channels
        resolution = image_size // 2
        while resolution > 4:
            out_ch = ch * 2
            blocks.append(DisBlock(ch, out_ch, use_spectral_norm=use_spectral_norm))
            ch = out_ch
            resolution //= 2
            if use_self_attention and resolution == image_size // 4:
                blocks.append(SelfAttention(ch))

        self.blocks = nn.Sequential(*blocks)
        final_conv = nn.Conv2d(ch, 1, kernel_size=4, stride=1, padding=0)
        if use_spectral_norm:
            final_conv = spectral_norm(final_conv)
        self.final = final_conv

    def forward(self, x):
        x = self.blocks(x)
        x = self.final(x)
        return x.view(x.size(0), -1).mean(dim=1)  # -> (B,) scalar logit per image


if __name__ == "__main__":
    # quick smoke test: run `python -m src.models` from the project root
    for size in (64, 128):
        g = Generator(latent_dim=128, image_size=size)
        d = Discriminator(image_size=size)
        z = torch.randn(2, 128)
        img = g(z)
        out = d(img)
        print(f"image_size={size}: generator output {tuple(img.shape)}, "
              f"discriminator output {tuple(out.shape)}")
