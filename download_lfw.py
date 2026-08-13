"""
download_lfw.py
================
Fetches ~13,000 REAL face photos automatically -- no Kaggle account, no
manual download, no API key. Just run this once, then start training.

Uses "Labeled Faces in the Wild" (LFW) via scikit-learn's built-in loader.
LFW is the same category of dataset as CelebA (public figures' photos,
originally collected for face-recognition research, long-established and
extremely widely reused across the ML community for exactly this purpose:
training *unconditional* face-generation GANs, not identifying or
recreating any specific person).

Usage
-----
    pip install scikit-learn
    python download_lfw.py
    python train.py --data_dir data/raw_faces

Notes
-----
- First run downloads and caches the raw dataset under
  ~/scikit_learn_data/ (a few hundred MB, one-time).
- This script then re-saves every image as a JPG under data/raw_faces/lfw/,
  in the flat-folder layout train.py already expects -- no other setup
  needed.
- This download happens on YOUR machine (wherever you run this script),
  not in any sandboxed environment -- make sure you actually have internet
  access here.
- Faces are provided as slightly-cropped rectangles (125x94 by default);
  train.py's preprocessing pipeline (resize -> center-crop -> resize)
  handles turning these into square images automatically.
- Want more diversity / higher resolution instead? See the README section
  "Getting a face dataset" for CelebA (Kaggle) and FFHQ alternatives.
"""

import os
import numpy as np
from PIL import Image

OUT_DIR = "data/raw_faces/lfw"


def main():
    from sklearn.datasets import fetch_lfw_people  # imported here so the rest
                                                      # of the project doesn't
                                                      # need scikit-learn installed

    print("Downloading/loading the LFW face dataset via scikit-learn "
          "(first run only -- this can take a few minutes)...")
    lfw = fetch_lfw_people(color=True, resize=1.0, min_faces_per_person=0)
    images = lfw.images  # float32 in [0, 1], shape (N, H, W, 3)
    n = images.shape[0]
    print(f"Got {n} face images ({images.shape[1]}x{images.shape[2]} pixels each, "
          f"{len(set(lfw.target.tolist()))} different people).")

    os.makedirs(OUT_DIR, exist_ok=True)
    for i, img in enumerate(images):
        arr = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(OUT_DIR, f"lfw_{i:05d}.jpg"), quality=95)
        if (i + 1) % 2000 == 0:
            print(f"  saved {i + 1}/{n}")

    print(f"\nDone -- {n} real face images saved to {OUT_DIR}/")
    print("Now run:  python train.py --data_dir data/raw_faces")


if __name__ == "__main__":
    main()
