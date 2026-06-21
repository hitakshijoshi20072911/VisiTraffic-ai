#!/usr/bin/env python3
"""
Dataset Downloader for VisiTraffic AI – Final Robust Version
- Uses Kaggle for BDD100K (requires Kaggle API key)
- Detects corrupt downloads (size < 1MB) and falls back to manual
- Treats ROD2021 as manual only
- Uses alternative public Kaggle dataset for Indian number plates
"""

import os
import sys
import zipfile
import tarfile
import shutil
from pathlib import Path

import requests
from tqdm import tqdm

# ---------- Configuration ----------
DATA_DIR = Path("data/raw")
# Minimum size for a valid dataset archive (1 MB)
MIN_ARCHIVE_SIZE = 1_000_000

DATASETS = {
    "BDD100K": {
        "type": "kaggle",
        "dataset": "bdd100k/bdd100k_images_100k",
        "extract_dir": "bdd100k_images_100k",
        "manual_url": "https://www.bdd100k.com/",
        "note": "BDD100K 100K images via Kaggle. Requires Kaggle API key setup."
    },
    "UA-DETRAC": {
        "type": "direct",
        "url": "https://detrac-db.rit.albany.edu/UA-DETRAC/DETRAC-train-data.zip",
        "file": "DETRAC-train-data.zip",
        "extract_dir": "DETRAC-train-data",
        "manual_url": "https://detrac-db.rit.albany.edu/download",
        "note": "If auto-download fails, use the manual link."
    },
    "ROD2021": {
        "type": "manual",
        "url": "https://www.cvlibs.net/datasets/rod2021/",
        "note": "Fill the form to get download link. Extract into data/raw/ROD2021/"
    },
    "IDD": {
        "type": "manual",
        "url": "https://idd.insaan.iiit.ac.in/",
        "note": "Register, download 'IDD Detection', extract into data/raw/IDD/"
    },
    "Helmet_Detection_Kaggle": {
        "type": "kaggle",
        "dataset": "andrewmvd/helmet-detection",
        "extract_dir": "helmet-detection"
    },
    "Indian_Plates_Alternative": {
        "type": "kaggle",
        "dataset": "aneesarom/rider-with-helmet-without-helmet-number-plate",
        "extract_dir": "indian-rider-plates"
    },
}

# ---------- Helper functions ----------
def download_file(url: str, dest: Path, desc: str = None, max_retries=2, timeout=60):
    """Download a file with progress bar, retries, and timeout."""
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            # Some servers redirect to login pages; check content-type
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                raise Exception("Server returned HTML, likely a login page.")
            total = int(response.headers.get('content-length', 0))
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, 'wb') as f, tqdm(
                desc=desc or dest.name,
                total=total,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
            # Check if file is large enough to be a real archive
            if dest.stat().st_size < MIN_ARCHIVE_SIZE:
                raise Exception(f"Downloaded file is too small ({dest.stat().st_size} bytes); likely corrupt.")
            return  # Success
        except Exception as e:
            print(f"   Attempt {attempt+1} failed: {e}")
            if attempt == max_retries:
                raise

def extract_archive(file_path: Path, extract_to: Path):
    """Extract zip or tar files."""
    if zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path, 'r') as zf:
            zf.extractall(extract_to)
    elif tarfile.is_tarfile(file_path):
        with tarfile.open(file_path, 'r') as tf:
            tf.extractall(extract_to)
    else:
        raise ValueError(f"Unknown archive format: {file_path}")

def handle_direct_dataset(name: str, info: dict):
    """Download and extract a direct URL dataset with fallback."""
    dest_file = DATA_DIR / info["file"]
    extract_dir = DATA_DIR / info["extract_dir"]
    if extract_dir.exists():
        print(f"✓ {name} already exists. Skipping.\n")
        return
    print(f"⏳ Downloading {name}...")
    try:
        download_file(info["url"], dest_file)
        print(f"📦 Extracting {name}...")
        extract_archive(dest_file, DATA_DIR)
        # Remove archive after extraction
        os.remove(dest_file)
        print(f"✅ {name} is ready.\n")
    except Exception as e:
        print(f"❌ Automatic download failed for {name}.")
        print(f"   Error: {e}")
        print(f"   👉 Please download manually from: {info.get('manual_url', info['url'])}")
        print(f"   Place the extracted files in: {extract_dir}\n")
        # Clean up any corrupted file
        if dest_file.exists():
            os.remove(dest_file)

def handle_kaggle_dataset(name: str, info: dict):
    """Download Kaggle dataset using kagglehub."""
    try:
        import kagglehub
    except ImportError:
        print("❌ kagglehub not installed. Run: pip install kagglehub")
        return

    extract_dir = DATA_DIR / info["extract_dir"]
    if extract_dir.exists():
        print(f"✓ {name} already exists. Skipping.\n")
        return
    print(f"⏳ Downloading {name} via kagglehub...")
    try:
        path = kagglehub.dataset_download(info["dataset"])
        shutil.move(path, extract_dir)
        print(f"✅ {name} downloaded to {extract_dir}.\n")
    except Exception as e:
        print(f"❌ Failed to download {name} from Kaggle: {e}")
        if "403" in str(e) or "unauthorized" in str(e).lower():
            print("   👉 You may need to set up your Kaggle API key:")
            print("      1. Go to kaggle.com/settings/account")
            print("      2. Click 'Create API Token'")
            print("      3. Place the downloaded kaggle.json in C:/Users/<you>/.kaggle/")
        print("   Check your internet or Kaggle API status.\n")

def handle_manual_dataset(name: str, info: dict):
    """Print manual download instructions."""
    print(f"ℹ️ {name} requires manual download.")
    print(f"   Visit: {info['url']}")
    print(f"   {info['note']}")
    print(f"   Place the extracted files in: {DATA_DIR / name}\n")

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=== VisiTraffic Dataset Downloader (Final) ===\n")

    # Check Kaggle API key existence
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_json.exists():
        print("⚠️ Kaggle API key not found. Please set it up before running this script.")
        print("   Instructions: https://www.kaggle.com/docs/api")
        print("   After setup, place kaggle.json in C:/Users/<you>/.kaggle/\n")
    else:
        print("✓ Kaggle API key found.\n")

    for ds_name, ds_info in DATASETS.items():
        if ds_info["type"] == "direct":
            handle_direct_dataset(ds_name, ds_info)
        elif ds_info["type"] == "kaggle":
            handle_kaggle_dataset(ds_name, ds_info)
        elif ds_info["type"] == "manual":
            handle_manual_dataset(ds_name, ds_info)

    print("\n🎉 All automatic downloads attempted.")
    print("If any failed, follow the manual instructions above.")

if __name__ == "__main__":
    main()