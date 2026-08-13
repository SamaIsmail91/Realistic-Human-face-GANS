"""
data.py
=======
Dataset loading & preprocessing for face images.

Two ways to supply data:
  1. Point --data_dir at any folder containing face images (any nesting,
     .jpg/.jpeg/.png). This is the recommended path -- works with CelebA,
     FFHQ, your own photos, a Kaggle download, etc.
  2. Set --dataset celeba to have torchvision download & manage CelebA for
     you automatically (first run only). Note: torchvision serves CelebA
     from Google Drive, which occasionally rate-limits; if it fails, download
     the "img_align_celeba" split manually (Kaggle mirror is the most
     reliable) and use option 1 instead.

Preprocessing pipeline (applied to every image):
  Resize (short side) -> CenterCrop (square, centered on the face region)
  -> Resize to target resolution -> RandomHorizontalFlip -> ToTensor
  -> Normalize to [-1, 1]  (matches the Generator's Tanh output range)
"""

import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_transform(image_size, train=True):
    # CelebA faces are roughly centered, so a modest center-crop before
    # resizing removes background / hair edges and focuses on the face,
    # which is standard practice for face-GAN preprocessing.
    crop_size = int(image_size * 1.15)
    ops = [
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms.Resize((image_size, image_size)),
    ]
    if train:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # -> [-1, 1]
    ]
    return transforms.Compose(ops)


class FlatFaceFolder(Dataset):
    """Recursively loads every image under `root`, no class-label folders required."""

    def __init__(self, root, transform=None):
        self.paths = []
        for ext in IMG_EXTENSIONS:
            self.paths.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        self.paths.sort()
        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"No images found under '{root}'. Populate this folder with face "
                f"images (jpg/png), or pass --dataset celeba to auto-download CelebA."
            )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label kept for API symmetry with labeled datasets


def get_dataloader(data_dir, image_size=64, batch_size=64, num_workers=4,
                    dataset="folder"):
    """
    dataset: "folder" (default, use `data_dir`) or "celeba" (auto-download).
    """
    transform = build_transform(image_size, train=True)

    if dataset == "celeba":
        ds = datasets.CelebA(root=data_dir, split="all", download=True, transform=transform)
    else:
        ds = FlatFaceFolder(data_dir, transform=transform)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return loader, ds


if __name__ == "__main__":
    # quick smoke test: python -m src.data --data_dir path/to/faces
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/raw_faces")
    p.add_argument("--image_size", type=int, default=64)
    args = p.parse_args()

    loader, ds = get_dataloader(args.data_dir, image_size=args.image_size, batch_size=4)
    batch, _ = next(iter(loader))
    print(f"Loaded {len(ds)} images. Batch shape: {tuple(batch.shape)}, "
          f"range [{batch.min():.2f}, {batch.max():.2f}]")
