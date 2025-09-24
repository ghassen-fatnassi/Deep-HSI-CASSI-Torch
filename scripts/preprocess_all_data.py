import os
import sys
import glob
import torch
import numpy as np
import imageio.v3 as iio
import scipy.io as sio
import OpenEXR, Imath
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


# ===========================
# Configuration
# ===========================
RAW_DATA_ROOT = '/root/Deep-HSI-CASSI-Torch/data/raw'
PROCESSED_DATA_ROOT = '/root/Deep-HSI-CASSI-Torch/data/processed'
PATCH_SIZE = 96
STRIDE = 96

# ===========================
# CAVE Dataset Processing
# ===========================
def read_png_folder(folder_path, fix_rgba=True):
    """Reads a folder of 31 PNG images and returns a (C,H,W) torch tensor."""
    png_files = sorted(glob.glob(os.path.join(folder_path, '*.png')))
    if len(png_files) != 31:
        raise ValueError(f"Expected 31 PNG files, got {len(png_files)} in {folder_path}")
    
    images = []
    for f in png_files:
        img = iio.imread(f).astype(np.float32) / 65535.0
        if fix_rgba and img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3].mean(axis=2)
        images.append(img)
    
    tensor = torch.tensor(np.stack(images, axis=0))
    return tensor

def save_patches(tensor, save_dir, scene_name, patch_size=96, stride=32):
    """Splits a (C,H,W) tensor into overlapping patches."""
    os.makedirs(save_dir, exist_ok=True)
    _, H, W = tensor.shape
    patch_idx = 0
    
    patches_saved = []
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = tensor[:, i:i+patch_size, j:j+patch_size]
            patch = patch.contiguous()
            patch_path = os.path.join(save_dir, f"{scene_name}_patch{patch_idx}.pt")
            torch.save(patch, patch_path)
            patches_saved.append(patch_path)
            patch_idx += 1
    
    return patches_saved

def preprocess_cave(cave_root, save_folder):
    """Preprocess all scenes in CAVE dataset."""
    print("\n=== Processing CAVE Dataset ===")
    os.makedirs(save_folder, exist_ok=True)
    
    scene_folders = sorted([os.path.join(cave_root, d) for d in os.listdir(cave_root)
                           if os.path.isdir(os.path.join(cave_root, d))])
    
    total_patches = 0
    for folder in tqdm(scene_folders, desc="CAVE scenes"):
        try:
            tensor = read_png_folder(folder)
            scene_name = os.path.basename(folder)
            patches = save_patches(tensor, save_folder, scene_name, PATCH_SIZE, STRIDE)
            total_patches += len(patches)
        except Exception as e:
            print(f"Skipping {folder}: {e}")
    
    print(f"CAVE: Saved {total_patches} patches")
    return total_patches

# ===========================
# KAIST Dataset Processing
# ===========================
def read_exr_file_openexr(exr_path, skip_first_n=3):
    """Reads an EXR file using OpenEXR."""
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()
    dw = header['dataWindow']
    H, W = dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1
    
    channel_names = sorted(header['channels'].keys())
    channel_names = channel_names[skip_first_n:]
    
    if len(channel_names) != 31:
        raise ValueError(f"Expected 31 channels, got {len(channel_names)}")
    
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = []
    for c in channel_names:
        raw = exr_file.channel(c, FLOAT)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(H, W)
        channels.append(arr)
    
    tensor = torch.tensor(np.stack(channels, axis=0), dtype=torch.float32)
    return tensor

def process_exr_file(exr_path):
    try:
        tensor = read_exr_file_openexr(exr_path, skip_first_n=3)
        base_name = os.path.splitext(os.path.basename(exr_path))[0]
        return tensor, base_name
    except Exception as e:
        print(f"Skipping {exr_path}: {e}")
        return None, None

def preprocess_kaist(exr_folder, save_folder):
    """Preprocess all EXR files in KAIST dataset using multiprocessing"""
    print("\n=== Processing KAIST Dataset ===")
    os.makedirs(save_folder, exist_ok=True)
    
    exr_files = sorted(glob.glob(os.path.join(exr_folder, '*.exr')))
    total_patches = 0
    
    # Use multiprocessing for EXR reading
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_exr_file, exr_files), 
                          total=len(exr_files), desc="Processing EXR files"))
    
    # Process results and save patches
    for tensor, base_name in results:
        if tensor is not None:
            patches = save_patches(tensor, save_folder, base_name, PATCH_SIZE, STRIDE)
            total_patches += len(patches)
    
    print(f"KAIST: Saved {total_patches} patches")
    return total_patches

# ===========================
# Harvard Dataset Processing
# ===========================
def read_mat_file(mat_path):
    """Reads a MATLAB .mat hyperspectral file."""
    mat = sio.loadmat(mat_path)
    
    if 'ref' not in mat:
        raise ValueError(f"'ref' variable not found in {mat_path}")
    
    hsi = mat['ref']  # usually (H,W,C)
    if hsi.shape[2] != 31:
        raise ValueError(f"Expected 31 channels, got {hsi.shape[2]}")
    
    hsi = np.transpose(hsi, (2,0,1))  # (C,H,W)
    tensor = torch.tensor(hsi, dtype=torch.float32)
    return tensor

def preprocess_harvard(mat_folders, save_folder):
    """Preprocess all .mat files in Harvard dataset."""
    print("\n=== Processing Harvard Dataset ===")
    os.makedirs(save_folder, exist_ok=True)
    
    total_patches = 0
    for mat_root in mat_folders:
        mat_files = sorted(glob.glob(os.path.join(mat_root, '*.mat')))
        
        for mat_path in tqdm(mat_files, desc=f"Harvard {os.path.basename(mat_root)}"):
            try:
                tensor = read_mat_file(mat_path)
                base_name = os.path.splitext(os.path.basename(mat_path))[0]
                patches = save_patches(tensor, save_folder, base_name, PATCH_SIZE, STRIDE)
                total_patches += len(patches)
            except Exception as e:
                print(f"Skipping {mat_path}: {e}")
    
    print(f"Harvard: Saved {total_patches} patches")
    return total_patches


def check_patch_consistency(folder_path):
    """Check all patches have consistent shape and channels"""
    files = glob.glob(f"{folder_path}/*.pt")
    if not files:
        return
    
    sample = torch.load(files[0])
    expected_shape = sample.shape
    expected_channels = expected_shape[0]
    
    inconsistent_files = []
    for f in tqdm(files, desc="Checking patch consistency"):
        patch = torch.load(f)
        if patch.shape != expected_shape:
            inconsistent_files.append(f)
    
    if inconsistent_files:
        print(f"Found {len(inconsistent_files)} inconsistent files:")
        for f in inconsistent_files[:5]:  # Show first 5
            patch = torch.load(f)
            print(f"  {f}: expected {expected_shape}, got {patch.shape}")
        # Remove inconsistent files
        for f in inconsistent_files:
            os.remove(f)
        print(f"Removed {len(inconsistent_files)} inconsistent files")
    else:
        print("All patches have consistent shape")
    
    return expected_shape

# ===========================
# Main Processing Function
# ===========================
def main():
    print("Starting data preprocessing...")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Stride: {STRIDE}")
    
    # Create main processed directory
    all_patches_dir = os.path.join(PROCESSED_DATA_ROOT, 'all_patches')
    os.makedirs(all_patches_dir, exist_ok=True)
    
    total_patches = 0
    
    # Process CAVE dataset
    cave_root = os.path.join(RAW_DATA_ROOT, 'cave')
    if os.path.exists(cave_root):
        patches = preprocess_cave(cave_root, all_patches_dir)
        total_patches += patches
    else:
        print(f"CAVE dataset not found at {cave_root}")
    
    # Process KAIST dataset
    kaist_root = os.path.join(RAW_DATA_ROOT, 'kaist-hyperspectral', 'exr')
    if os.path.exists(kaist_root):
        patches = preprocess_kaist(kaist_root, all_patches_dir)
        total_patches += patches
    else:
        print(f"KAIST dataset not found at {kaist_root}")
    
    # Process Harvard dataset
    harvard_root = os.path.join(RAW_DATA_ROOT, 'harvard')
    if os.path.exists(harvard_root):
        cz_hsdb_natural = os.path.join(harvard_root, 'CZ_hsdb')
        cz_hsdb_artificial = os.path.join(harvard_root, 'CZ_hsdbi')
        patches = preprocess_harvard([cz_hsdb_natural, cz_hsdb_artificial], all_patches_dir)
        total_patches += patches
    else:
        print(f"Harvard dataset not found at {harvard_root}")
    
    print(f"\n=== Preprocessing Complete ===")
    check_patch_consistency(all_patches_dir)
    print(f"Total patches saved: {total_patches}")
    print(f"All patches saved to: {all_patches_dir}")
    

if __name__ == "__main__":
    main()