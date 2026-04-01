# -*- coding: utf-8 -*-
import numpy as np
import os
from skimage import io, transform
from tqdm import tqdm


# Define paths
img_path = "Path to endoscopic RGB images"
mask_path = "Path to segmentation masks"
output_path = "./data/npy/Endo_SEOUL"
os.makedirs(os.path.join(output_path, "imgs"), exist_ok=True)
os.makedirs(os.path.join(output_path, "gts"), exist_ok=True)

# Parameters
image_size = 1024
voxel_num_thre2d = 100  # Minimum pixel count for mask regions

# Load image/mask names
img_names = sorted(os.listdir(img_path))
mask_names = sorted(os.listdir(mask_path))

# Ensure image-mask pairing is correct
valid_names = [name for name in img_names if name in mask_names]
print(f"Total valid image-mask pairs: {len(valid_names)}")

# Process each image
for name in tqdm(valid_names):
    # Load image (RGB)
    img = io.imread(os.path.join(img_path, name))[:,:,0:3]
    if img.shape[-1] != 3:
        raise ValueError(f"Image {name} does not have 3 channels (RGB). Found shape: {img.shape}")
    
    # Load mask (Grayscale)
    mask = io.imread(os.path.join(mask_path, name), as_gray=True)
    mask = np.uint8(mask > 0)  # Convert to binary if needed
    
    # Remove small components in mask
    # mask_clean = cc3d.dust(mask, threshold=voxel_num_thre2d, connectivity=8, in_place=True)
    mask_clean=mask
    
    # Resize image and mask
    img_resized = transform.resize(img, (image_size, image_size), order=3, preserve_range=True, mode="constant", anti_aliasing=True)
    mask_resized = transform.resize(mask_clean, (image_size, image_size), order=0, preserve_range=True, mode="constant", anti_aliasing=False)
    mask_resized = np.uint8(mask_resized)
    
    # Normalize image to [0,1]
    img_resized = img_resized / 255.0
    
    # Save compressed NPZ file
    np.savez_compressed(os.path.join(output_path, name.replace('.png', '.npz').replace('.jpg', '.npz')), 
                         imgs=img_resized, gts=mask_resized)
    
    # Save each slice as individual .npy
    np.save(os.path.join(output_path, "imgs", name.replace('.png', '.npy').replace('.jpg', '.npy')), img_resized)
    np.save(os.path.join(output_path, "gts", name.replace('.png', '.npy').replace('.jpg', '.npy')), mask_resized)
    
print("Processing complete!")
