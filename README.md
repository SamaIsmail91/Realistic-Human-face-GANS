# Realistic Human Face Generation with a GAN

A from-scratch PyTorch implementation of a Generative Adversarial Network that
learns to synthesize realistic human face images from random noise. Built as
a complete, runnable project rather than a single notebook: data pipeline,
model architecture, training loop, experiment tracking, quantitative
evaluation, inference, and an interactive demo.

> Every image the trained model produces is synthetic -- there is no real
> person behind any generated face.

**Two ways to use this project:**
- **`Face_GAN_Generation.ipynb`** -- a single, self-contained notebook covering every stage top to bottom (data -> models -> training -> monitoring -> results). Best for learning, presenting, or running in Google Colab/Kaggle.
- **The `.py` scripts** (`train.py`, `generate.py`, `evaluate.py`, `app.py` + `src/`) -- a modular, production-style version of the *same* project with extras the notebook doesn't include: resumable checkpoints, TensorBoard logging, FID scoring, and a Gradio demo app. Best for a real training run you might stop/restart, or for reusing the code elsewhere.

Both implement the same architecture (self-attention + spectral norm + hinge loss + DiffAugment + EMA) -- pick whichever fits how you want to work.

---

## 1. How it works (the short version)

A GAN is two networks locked in a contest:

- **Generator (G)** takes a vector of random noise `z` and tries to turn it
  into an image that looks like a real face.
- **Discriminator (D)** looks at an image (real or generated) and tries to
  guess which one it is.

They train together. G gets better at fooling D; D gets better at catching
G. At equilibrium, G has learned to map random noise onto the manifold of
"plausible face images."

```
random noise z (128-dim)                real face images
        │                                       │
        ▼                                       │
  ┌───────────┐        generated image          │
  │ Generator │ ───────────────┐                 │
  └───────────┘                ▼                 ▼
                          ┌───────────────────────────┐
                          │       Discriminator        │
                          │  "real" logit / "fake" logit│
                          └───────────────────────────┘
                                       │
                       gradients flow back to both networks
```

## 2. Architecture -- what makes this "advanced"

This isn't a bare-bones DCGAN tutorial. On top of the standard
transposed-conv Generator / strided-conv Discriminator backbone, it adds
techniques from the GAN literature that meaningfully improve face quality
and training stability:

| Technique | What it does | Where |
|---|---|---|
| **Self-Attention (SAGAN)** | Lets every pixel attend to every other pixel, so eyes/nose/mouth stay globally consistent instead of only locally coherent | `src/models.py` |
| **Spectral Normalization** | Constrains the Discriminator to be Lipschitz-continuous, the single biggest lever for stopping GAN training from diverging | `src/models.py` |
| **Hinge loss** (default, swappable) | More stable gradients than vanilla BCE, standard in SAGAN/BigGAN | `src/losses.py` |
| **WGAN-GP option** | Wasserstein loss + gradient penalty, an alternative stabilization strategy | `src/losses.py` |
| **DiffAugment** | Differentiable color/translation/cutout augmentation applied to real *and* fake images, so the model needs far less data before it stops overfitting/collapsing | `src/diffaugment.py` |
| **TTUR** (two time-scale updates) | Discriminator learns faster than the Generator (`lr_d=4e-4`, `lr_g=1e-4`) -- keeps the adversarial game balanced | `config.py` |
| **EMA of Generator weights** | Sampling from a smoothed average of recent generator weights instead of the raw noisy weights gives visibly sharper, more consistent output | `src/utils.py` |
| **Mixed precision (AMP)** | ~2x faster training on modern GPUs with no quality loss | `train.py` |
| **DCGAN weight init** | `N(0, 0.02)` initialization, empirically important for conv-GAN convergence | `src/models.py` |

All of it is configurable/toggleable from `config.py` or the command line, so
you can turn features off to see their individual effect -- a good way to
actually learn what each piece contributes.

## 3. Project structure

```
face_gan_project/
├── config.py              # every hyperparameter, in one place
├── requirements.txt
├── train.py                # main training loop
├── generate.py              # sample faces from a trained checkpoint
├── evaluate.py              # FID score, loss curves, training-progress grid
├── app.py                   # interactive Gradio demo
├── src/
│   ├── data.py               # dataset loading & preprocessing
│   ├── models.py              # Generator / Discriminator / SelfAttention
│   ├── losses.py               # hinge / bce / wgan-gp
│   ├── diffaugment.py           # differentiable augmentation
│   └── utils.py                  # EMA, checkpointing, logging, plotting
├── data/raw_faces/          # <- put your training images here
└── outputs/                 # created automatically during training
    ├── checkpoints/           # .pt files (resume + inference)
    ├── samples/                # PNG grids saved during training
    ├── logs/                    # TensorBoard + losses.csv
    └── eval/                      # FID / plots from evaluate.py
```

## 4. Setup

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

GPU strongly recommended. On CPU only, stick to `--image_size 64` and small
batch sizes -- it will still run, just slowly (fine for verifying everything
works before moving to a GPU machine or Colab).

## 5. Getting a face dataset

The code trains on **any folder of face images** -- nothing is hardcoded to
one dataset.

**Option A -- auto-download real faces, zero manual steps (easiest).**
```bash
pip install scikit-learn
python download_lfw.py
python train.py --data_dir data/raw_faces
```
This fetches ~13,000 real face photos ("Labeled Faces in the Wild" --
the same category of dataset as CelebA: public figures' photos,
long-established in the ML community and reused constantly for exactly
this task) via scikit-learn's built-in loader, and saves them straight into
`data/raw_faces/lfw/` in the layout `train.py` expects. No account, no
Kaggle, no manual download. (The notebook version has the same thing built
in -- just set `AUTO_DOWNLOAD_LFW = True` in Section 2.)

**Option B -- your own images.** Drop `.jpg`/`.png` files into
`data/raw_faces/` (subfolders are fine, they're scanned recursively). A few
hundred reasonably similar, front-facing face photos is a usable minimum
thanks to DiffAugment; a few thousand+ is much better.

**Option C -- CelebA (the standard face-GAN benchmark, ~200k images, more variety than LFW).**
The most reliable way to get it is Kaggle's mirror (the original Google
Drive link is frequently rate-limited):
1. Download the `img_align_celeba` split from Kaggle
   (search "CelebA Kaggle", or `kaggle datasets download -d jessicali9530/celeba-dataset`).
2. Unzip so the images live under `data/raw_faces/img_align_celeba/*.jpg`.
3. Train normally with `--data_dir data/raw_faces`.

Auto-download is also wired up if you'd rather let torchvision try:
```bash
python train.py --dataset celeba --data_dir data/celeba
```
(this calls `torchvision.datasets.CelebA(download=True)`, which pulls from
Google Drive and can fail if Google rate-limits the request -- Option A/C
above are the dependable paths).

**Option D -- FFHQ** (higher quality, aligned 1024px faces): download from
the [official FFHQ repo](https://github.com/NVlabs/ffhq-dataset) and point
`--data_dir` at the image folder. Great if you want to push past 128px later.

## 6. Training

```bash
# fast baseline: 64x64, good for iterating on ideas
python train.py --data_dir data/raw_faces --image_size 64 --epochs 100

# higher fidelity (needs more VRAM / time)
python train.py --data_dir data/raw_faces --image_size 128 --batch_size 32 --epochs 200

# resume is automatic -- rerunning the same command picks up the latest checkpoint
python train.py --data_dir data/raw_faces

# every hyperparameter in config.py is also a CLI flag, e.g.:
python train.py --data_dir data/raw_faces --lr_g 2e-4 --loss_type wgan_gp --d_steps_per_g_step 5
```

Watch it train in real time:
```bash
tensorboard --logdir outputs/logs
```

What to expect while watching the losses: they will **not** monotonically
decrease -- that's normal for adversarial training. What you want to see is
both losses oscillating in a roughly stable band (neither one collapsing to
~0 while the other explodes). `D(real)` and `D(fake)` converging toward each
other over time is a healthier signal than the raw loss values.

Rough face-quality timeline on CelebA at 64px with a single modern GPU:
- **~5-10 epochs**: blurry color blobs, rough face-shaped silhouette
- **~20-40 epochs**: recognizable facial structure, eyes/nose/mouth placed correctly, artifacts remain
- **~80-150 epochs**: coherent faces, occasional asymmetry/background artifacts
- **150+ epochs**: diminishing returns without also scaling the architecture (see §9)

## 7. Evaluating results

**Loss curves:**
```bash
python evaluate.py --mode losses --csv outputs/logs/losses.csv
```

**Visual training progression** (stitches saved sample grids side by side so you can see the model improve over time):
```bash
python evaluate.py --mode progress --samples_dir outputs/samples
```

**FID score** (Frechet Inception Distance -- the standard quantitative face-quality metric, lower is better; needs ≥1-2k images on each side to be a meaningful number):
```bash
python evaluate.py --mode fid \
    --checkpoint outputs/checkpoints/ckpt_epoch0100.pt \
    --real_dir data/raw_faces --num_samples 2000
```

Rough FID intuition: 30-60 = recognizable faces with visible artifacts (typical for a DCGAN-family model at 64px on a modest dataset/budget), 10-30 = clearly good, <10 = excellent (usually needs StyleGAN-scale architecture and data).

## 8. Generating new faces / demo

```bash
# grid of 64 random faces
python generate.py --checkpoint outputs/checkpoints/ckpt_epoch0100.pt --num_images 64

# individual PNG files
python generate.py --checkpoint outputs/checkpoints/ckpt_epoch0100.pt --num_images 16 --individual

# latent-space interpolation between two random faces (nice for a demo GIF)
python generate.py --checkpoint outputs/checkpoints/ckpt_epoch0100.pt --interpolate --steps 30
```

Or launch the interactive browser demo:
```bash
python app.py --checkpoint outputs/checkpoints/ckpt_epoch0100.pt
```

## 9. Pushing further

This project is deliberately a strong-but-tractable DCGAN+SAGAN hybrid --
trainable on a single GPU in hours, not the multi-week / multi-GPU budgets
that true state-of-the-art face models need. Natural next steps, roughly in
order of effort:

1. **Train longer / on more data.** The single highest-leverage lever.
2. **Progressive resizing.** Train at 64px to convergence, then use those
   weights to warm-start a 128px run (freeze early layers initially).
3. **Conditional generation.** Concatenate a one-hot attribute vector
   (CelebA ships 40 binary attributes: smiling, glasses, hair color...) to
   `z` so you can control generated attributes.
4. **Swap in StyleGAN2/3.** For genuinely photoreal, high-resolution faces,
   NVIDIA's StyleGAN family (style-based generator, mapping network,
   adaptive instance norm) is the actual state of the art; this project's
   data pipeline and evaluation scripts (FID, DiffAugment) transfer
   directly if you swap the model in `src/models.py`.
5. **Diffusion models.** For 2024+-era quality, a DDPM/latent-diffusion
   model outperforms GANs on fidelity, at the cost of much slower sampling
   (this is a genuinely different architecture, not a drop-in swap).

## 10. Troubleshooting

- **Discriminator loss → 0, Generator loss → huge, samples turn to noise**
  ("mode collapse" / D overpowering G): lower `--lr_d`, make sure
  `--d_steps_per_g_step` is 1, and confirm `--use_spectral_norm true` and
  `--use_diffaugment true`.
- **Losses NaN**: lower the learning rate, enable gradient clipping
  (`--grad_clip_norm 1.0`), and double check images are actually loading in
  `[-1, 1]` range (`python -m src.data --data_dir data/raw_faces`).
- **All generated faces look nearly identical** (mode collapse): usually
  means D got too weak too early -- try `--loss_type wgan_gp
  --d_steps_per_g_step 5`, or reduce `--lr_g`.
- **Out of memory**: lower `--batch_size`, or use `--image_size 64` instead
  of 128.
- **Checkpoints eating disk space**: each checkpoint stores G, D, EMA, and
  both optimizer states (needed for exact resume) -- raise
  `--checkpoint_every` if you don't need every epoch saved.

## 11. References

- Goodfellow et al., *Generative Adversarial Networks*, 2014
- Radford et al., *Unsupervised Representation Learning with Deep
  Convolutional GANs (DCGAN)*, 2015
- Zhang et al., *Self-Attention Generative Adversarial Networks (SAGAN)*, 2018
- Miyato et al., *Spectral Normalization for Generative Adversarial
  Networks*, 2018
- Gulrajani et al., *Improved Training of Wasserstein GANs (WGAN-GP)*, 2017
- Zhao et al., *Differentiable Augmentation for Data-Efficient GAN
  Training (DiffAugment)*, 2020
- Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a
  Local Nash Equilibrium (FID metric, TTUR)*, 2017
- Liu et al., *Deep Learning Face Attributes in the Wild (CelebA)*, 2015
