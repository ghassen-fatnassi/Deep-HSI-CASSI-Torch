#!/bin/bash

# Setup Git LFS
git lfs install

echo "Creating data directories..."
mkdir -p /root/Deep-HSI-CASSI-Torch/data/raw
mkdir -p /root/Deep-HSI-CASSI-Torch/data/processed

cd /root/Deep-HSI-CASSI-Torch/data/raw

echo "Downloading CAVE dataset..."
git clone https://huggingface.co/datasets/danaroth/cave
rm -rf cave/.git cave/README.md

echo "Downloading KAIST dataset..."
git clone https://huggingface.co/datasets/danaroth/kaist-hyperspectral
rm -rf kaist-hyperspectral/.git

echo "Downloading Harvard dataset..."
git clone https://huggingface.co/datasets/danaroth/harvard
rm -rf harvard/.git

echo "Data download complete!"