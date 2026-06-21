#!/bin/bash
# Docker-based dataset downloader for VisiTraffic AI
# Mounts data/raw and attempts to download datasets using wget.

DATA_DIR="/data/raw"
mkdir -p "$DATA_DIR"

# ---------- BDD100K (100K images) ----------
echo "--- Downloading BDD100K (100K images) ---"
if [ ! -d "$DATA_DIR/bdd100k_images_100k" ]; then
    wget -c "https://dl.cv.ethz.ch/bdd100k/data/bdd100k_images_100k.zip" -O /tmp/bdd100k.zip && \
    unzip -q /tmp/bdd100k.zip -d "$DATA_DIR" && \
    rm /tmp/bdd100k.zip && \
    echo "BDD100K done."
else
    echo "BDD100K already exists."
fi

# ---------- UA-DETRAC (train data) ----------
echo "--- Downloading UA-DETRAC ---"
if [ ! -d "$DATA_DIR/DETRAC-train-data" ]; then
    wget -c "https://detrac-db.rit.albany.edu/UA-DETRAC/DETRAC-train-data.zip" -O /tmp/detrac.zip && \
    unzip -q /tmp/detrac.zip -d "$DATA_DIR" && \
    rm /tmp/detrac.zip && \
    echo "UA-DETRAC done."
else
    echo "UA-DETRAC already exists."
fi

# ROD2021 and IDD require manual form; skip here.
echo "ROD2021 and IDD require manual download. See instructions below."

echo "All automated downloads attempted."