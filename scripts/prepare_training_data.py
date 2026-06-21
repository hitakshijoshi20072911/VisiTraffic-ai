"""
Copy a small random subset of IDD and BDD100K images into a training folder.
Run once: python scripts/prepare_training_data.py
"""

import os
import random
import shutil
from pathlib import Path

# Configuration
IDD_IMAGES = Path("data/raw/IDD/JPEGImages")      # adjust to your actual path
IDD_ANNOTATIONS = Path("data/raw/IDD/Annotations") # adjust if annotations are separate

BDD_IMAGES = Path("data/raw/bdd100k_images_100k/train")
BDD_LABELS = Path("data/raw/bdd100k_labels")       # labels for BDD100K (if available)

TRAIN_IMAGES = Path("data/train_subset/images")
TRAIN_LABELS = Path("data/train_subset/labels")

NUM_IDD = 300
NUM_BDD = 200

def copy_files(src_dir, dst_dir, num_files, extensions=(".jpg", ".png")):
    dst_dir.mkdir(parents=True, exist_ok=True)
    all_files = [f for f in src_dir.iterdir() if f.suffix.lower() in extensions]
    selected = random.sample(all_files, min(num_files, len(all_files)))
    for f in selected:
        shutil.copy(f, dst_dir / f.name)
    return selected

def main():
    random.seed(42)
    TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
    TRAIN_LABELS.mkdir(parents=True, exist_ok=True)

    # IDD
    if IDD_IMAGES.exists():
        idd_files = copy_files(IDD_IMAGES, TRAIN_IMAGES, NUM_IDD)
        # Copy corresponding XML annotations if they exist
        for img in idd_files:
            xml_path = IDD_ANNOTATIONS / (img.stem + ".xml")
            if xml_path.exists():
                shutil.copy(xml_path, TRAIN_LABELS / xml_path.name)
        print(f"Copied {len(idd_files)} IDD images.")
    else:
        print("IDD folder not found. Skipping.")

    # BDD100K
    if BDD_IMAGES.exists():
        bdd_files = copy_files(BDD_IMAGES, TRAIN_IMAGES, NUM_BDD)
        print(f"Copied {len(bdd_files)} BDD100K images.")
    else:
        print("BDD100K folder not found. Skipping.")

    print(f"Training subset ready: {len(list(TRAIN_IMAGES.iterdir()))} images.")

if __name__ == "__main__":
    main()