#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import glob
import wandb
import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
import random
import os
import argparse
from datetime import datetime
import torchvision.transforms as transforms
from torchvision.transforms.functional import rotate


from ConvAE import Encoder,Decoder,ConvAutoencoder,ae_loss_exact,psnr
# =============================
# Dataset Class
# =============================


class HyperspectralDataset(Dataset):
    def __init__(self, folder_path, normalize=True, augment=True, mode='train'):
        self.files = glob.glob(f"{folder_path}/*.pt")
        self.normalize = normalize
        self.augment = augment
        self.mode = mode
        
        # Compute normalization stats
        if self.normalize and len(self.files) > 0:
            self._compute_normalization_stats()
        
        # Define augmentations
        if self.augment and self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: self._random_flip(x)),
                transforms.Lambda(lambda x: self._random_rotate(x)),
            ])
        else:
            self.transform = None

    def _compute_normalization_stats(self):
        # Compute mean and std for normalization
        sample = torch.load(self.files[0])
        c = sample.size(0)
        
        # Compute mean and std
        mean = torch.zeros(c)
        std = torch.zeros(c)
        count = 0
        
        print("Computing normalization statistics...")
        for f in tqdm(self.files, desc="Computing stats"):
            x = torch.load(f)
            mean += x.view(c, -1).mean(dim=1)
            std += x.view(c, -1).std(dim=1)
            count += 1
        
        self.mean = mean / count
        self.std = std / count
        print(f"Normalization computed (mean: {self.mean.mean():.4f}, std: {self.std.mean():.4f})")

    def _random_flip(self, x):
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])  # Horizontal flip
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])  # Vertical flip
        return x

    def _random_rotate(self, x):
        if random.random() < 0.5:
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                # Rotate each channel separately
                rotated = []
                for i in range(x.size(0)):
                    rotated.append(rotate(x[i:i+1], angle))
                x = torch.cat(rotated, dim=0)
        return x

    def __getitem__(self, idx):
        x = torch.load(self.files[idx]).float()
        
        if self.normalize:
            x = (x - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-8)
        
        if self.augment and self.mode == 'train' and self.transform is not None:
            x = self.transform(x)
        
        return x


# =============================
# Training Function
# =============================
def train(model, train_loader, val_loader=None, device='cuda', epochs=60, 
          lr=1e-4, tau_w=1e-8, weight_decay=1e-8, grad_clip=1.0, 
          save_dir="/root/Deep-HSI-CASSI-Torch/checkpoints", wandb_key=None):
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    # Setup wandb
    if wandb_key:
        wandb.login(key=wandb_key)
    
    run_name = f"hsi_ae_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project="hyperspectral_autoencoder", name=run_name, config={
        "epochs": epochs,
        "batch_size": train_loader.batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "tau_w": tau_w,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "dataset_size": len(train_loader.dataset)
    })
    
    model = model.to(device)
    best_psnr = -1.0
    best_val_psnr = -1.0
    
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loader.dataset.training = True
        epoch_loss, epoch_psnr = 0.0, 0.0
        
        for batch_idx, x in enumerate(train_loader):
            x = x.to(device)
            
            optimizer.zero_grad()
            with autocast():
                recon, alpha = model(x)
                loss = ae_loss_exact(recon, x, tau_w=tau_w, model=model)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item() * x.size(0)
            epoch_psnr += psnr(recon, x).item() * x.size(0)
            
            # Log batch metrics
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} PSNR: {psnr(recon, x).item():.2f}dB")
        
        scheduler.step()
        epoch_loss /= len(train_loader.dataset)
        epoch_psnr /= len(train_loader.dataset)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loader.dataset.training = False
            val_loss, val_psnr = 0.0, 0.0
            
            with torch.no_grad():
                for x in val_loader:
                    x = x.to(device)
                    recon, alpha = model(x)
                    loss = ae_loss_exact(recon, x, tau_w=tau_w, model=model)
                    val_loss += loss.item() * x.size(0)
                    val_psnr += psnr(recon, x).item() * x.size(0)
            
            val_loss /= len(val_loader.dataset)
            val_psnr /= len(val_loader.dataset)
            
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_psnr': best_val_psnr,
                }, os.path.join(save_dir, 'best_val_model.pth'))
        
        # Save best training model
        if epoch_psnr > best_psnr:
            best_psnr = epoch_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_psnr': best_psnr,
            }, os.path.join(save_dir, 'best_model.pth'))
        
        # Log to wandb
        log_dict = {
            "train_loss": epoch_loss,
            "train_psnr": epoch_psnr,
            "lr": optimizer.param_groups[0]["lr"],
            "best_psnr": best_psnr,
        }
        
        if val_loader is not None:
            log_dict.update({
                "val_loss": val_loss,
                "val_psnr": val_psnr,
                "best_val_psnr": best_val_psnr,
            })
        
        wandb.log(log_dict)
        
        # Visual logging
        if (epoch+1) % 32 == 0:
            x_vis = x[:1].detach().cpu()
            recon_vis = recon[:1].detach().cpu()
            input_rgb = (x_vis[0, :3].permute(1,2,0).numpy() * 255).clip(0,255).astype("uint8")
            recon_rgb = (recon_vis[0, :3].permute(1,2,0).numpy() * 255).clip(0,255).astype("uint8")
            wandb.log({
                "input_rgb": wandb.Image(input_rgb, caption=f"Input Epoch {epoch+1}"),
                "recon_rgb": wandb.Image(recon_rgb, caption=f"Recon Epoch {epoch+1}")
            })
        
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(f"  Train - Loss: {epoch_loss:.4f}, PSNR: {epoch_psnr:.2f}dB (best: {best_psnr:.2f})")
        if val_loader is not None:
            print(f"  Val   - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}dB (best: {best_val_psnr:.2f})")
        print("-" * 50)
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_psnr': epoch_psnr,
    }, os.path.join(save_dir, 'final_model.pth'))
    
    wandb.finish()
    print("\nTraining completed!")
    print(f"Best training PSNR: {best_psnr:.2f}dB")
    if val_loader is not None:
        print(f"Best validation PSNR: {best_val_psnr:.2f}dB")

# =============================
# Main Function
# =============================
def main():
    parser = argparse.ArgumentParser(description='Train Hyperspectral Autoencoder')
    parser.add_argument('--data_path', type=str, default='/root/Deep-HSI-CASSI-Torch/data/processed/all_patches',
                        help='Path to processed patches')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--R', type=int, default=64, help='Number of features')
    parser.add_argument('--d', type=int, default=5, help='Network depth')
    parser.add_argument('--wandb_key', type=str, default=None, help='WandB API key')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Create dataset
    print(f"Loading dataset from {args.data_path}")
    full_dataset = HyperspectralDataset(args.data_path, normalize=True, augment=True)
    
    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create model
    encoder = Encoder(in_channels=31, R=args.R, d=args.d, use_batchnorm=True)
    decoder = Decoder(out_channels=31, R=args.R, d=args.d, use_batchnorm=True, output_activation='sigmoid')
    model = ConvAutoencoder(encoder, decoder)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train
    train(model, train_loader, val_loader, 
          device='cuda' if torch.cuda.is_available() else 'cpu',
          epochs=args.epochs, lr=args.lr, wandb_key=args.wandb_key)

if __name__ == "__main__":
    main()