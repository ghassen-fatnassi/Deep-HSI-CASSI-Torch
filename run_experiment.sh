#!/bin/bash

# Activate virtual environment
source /root/hsi_env/bin/activate

# Set working directory
cd /root/hsi_autoencoder

# Check GPU availability
echo "==================================="
echo "Checking GPU Status..."
echo "==================================="
nvidia-smi
echo ""

# Step 1: Download data
echo "==================================="
echo "Step 1: Downloading datasets..."
echo "==================================="
chmod +x scripts/download_data.sh
./scripts/download_data.sh

# Step 2: Preprocess data
echo "==================================="
echo "Step 2: Preprocessing datasets..."
echo "==================================="
python3 scripts/preprocess_all_data.py

# Step 3: Count total patches
echo "==================================="
echo "Step 3: Verifying processed data..."
echo "==================================="
PATCH_COUNT=$(ls -1 data/processed/all_patches/*.pt 2>/dev/null | wc -l)
echo "Total patches found: $PATCH_COUNT"

if [ $PATCH_COUNT -eq 0 ]; then
    echo "ERROR: No patches found! Check preprocessing."
    exit 1
fi

# Step 4: Start training
echo "==================================="
echo "Step 4: Starting training..."
echo "==================================="

# Get WandB key (optional - replace with your key)
WANDB_KEY="48c8b8d2ffad22e15da2eb80cf917fb45c1fb543"  # Replace this!

# Run training with recommended parameters
python3 scripts/train_autoencoder.py \
    --data_path /root/Deep-HSI-CASSI-Torch/data/processed/all_patches \
    --batch_size 64 \
    --epochs 60 \
    --lr 5e-4 \
    --R 64 \
    --d 5 \
    --wandb_key $WANDB_KEY \
    --val_split 0.2

echo "==================================="
echo "Experiment complete!"
echo "Checkpoints saved in: /root/Deep-HSI-CASSI-Torch/checkpoints"
echo "==================================="