"""
config.py
=========
Single source of truth for every hyperparameter in the project.
Everything here can be overridden from the command line (see train.py --help).
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    # ---------------------------------------------------------------- data
    data_dir: str = "data/raw_faces"       # folder of face images (any nesting, jpg/png)
    image_size: int = 64                   # 64 (fast) or 128 (higher quality, slower)
    channels: int = 3
    batch_size: int = 64
    num_workers: int = 4

    # ------------------------------------------------------------- model
    latent_dim: int = 128                  # size of the random noise vector z
    g_base_channels: int = 64              # generator base feature width (ngf)
    d_base_channels: int = 64              # discriminator base feature width (ndf)
    use_self_attention: bool = True        # SAGAN-style self-attention layers
    use_spectral_norm: bool = True         # spectral norm on discriminator (stabilizes training)

    # ---------------------------------------------------------- training
    epochs: int = 100
    lr_g: float = 1e-4                     # generator learning rate
    lr_d: float = 4e-4                     # discriminator learning rate (TTUR: D learns faster than G)
    beta1: float = 0.0                     # Adam beta1 (0.0 recommended for SAGAN/hinge-loss GANs)
    beta2: float = 0.9                     # Adam beta2
    loss_type: str = "hinge"               # "hinge" | "bce" | "wgan_gp"
    gp_lambda: float = 10.0                # gradient penalty weight (only used if loss_type == wgan_gp)
    label_smoothing: float = 0.9           # only used if loss_type == "bce"
    d_steps_per_g_step: int = 1            # discriminator updates per generator update
    use_diffaugment: bool = True           # differentiable augmentation (helps with small datasets)
    diffaugment_policy: str = "color,translation,cutout"
    ema_decay: float = 0.999               # exponential moving average of generator weights
    mixed_precision: bool = True           # torch.cuda.amp (ignored automatically on CPU)
    grad_clip_norm: float = 0.0            # 0 disables gradient clipping

    # ------------------------------------------------------------- misc
    seed: int = 42
    num_fixed_samples: int = 64            # how many images in the tracked sample grid
    sample_every: int = 200                # save a sample grid every N generator steps
    checkpoint_every: int = 1              # save a checkpoint every N epochs
    log_every: int = 50                    # print / log loss every N steps
    resume: bool = True                    # auto-resume from latest checkpoint if present

    # ------------------------------------------------------------- paths
    output_dir: str = "outputs"
    samples_dir: str = field(init=False)
    checkpoints_dir: str = field(init=False)
    logs_dir: str = field(init=False)

    def __post_init__(self):
        self.samples_dir = f"{self.output_dir}/samples"
        self.checkpoints_dir = f"{self.output_dir}/checkpoints"
        self.logs_dir = f"{self.output_dir}/logs"
