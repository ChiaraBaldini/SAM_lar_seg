#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_nnunet_from_png.py  (grayscale version)
-----------------------------------------------
Convert a dataset of PNG images and masks into nnU-Net v1 format (2D, single-channel).
Each image must have a matching mask with the same filename.
Images and masks are expected as grayscale (single channel).

Usage:
  export nnUNet_raw_data_base=/path/to/nnUNet_raw_data_base
  python prepare_nnunet_from_png.py \
      --images-dir /path/images \
      --masks-dir /path/masks \
      --task-id 501 \
      --task-name LARSeg
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import imageio.v3 as iio
import nibabel as nib
from tqdm import tqdm


def read_grayscale_png(path: Path) -> np.ndarray:
    """Read a PNG and return a 2D float32 numpy array."""
    arr = iio.imread(path)
    if arr.ndim == 3:
        # convert RGB/RGBA to grayscale using luminance
        arr = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    return arr.astype(np.float32)


def save_image_nifti(image: np.ndarray, out_path: Path, dry_run: bool = False) -> None:
    # image: (H, W) -> (H, W, 1)
    if image.ndim == 2:
        image = image[:, :, None]
    affine = np.eye(4, dtype=np.float32)
    nii = nib.Nifti1Image(image.astype(np.float32), affine)
    if not dry_run:
        nib.save(nii, str(out_path))

def save_mask_nifti(mask_png: Path, out_path: Path, dry_run: bool = False, multiclass: bool = False) -> None:
    m = iio.imread(mask_png)
    if m.ndim == 3:
        m = m[..., 0]
    m = m.astype(np.int16) if multiclass else (m > 0).astype(np.uint8)
    # mask: (H, W) -> (H, W, 1)
    if m.ndim == 2:
        m = m[:, :, None]
    affine = np.eye(4, dtype=np.float32)
    nii = nib.Nifti1Image(m, affine)
    if not dry_run:
        nib.save(nii, str(out_path))


def build_dataset_json(task_folder: Path, training_entries, labels_map):
    """Generate a nnU-Net compliant dataset.json file."""
    dataset_json = {
        "name": task_folder.name.split("_", 1)[1] if "_" in task_folder.name else task_folder.name,
        "description": "Dataset converted from PNG (2D grayscale images and masks).",
        "reference": "",
        "licence": "",
        "release": "1.0",
        "tensorImageSize": "2D",
        "modality": {"0": "Gray"},
        "labels": labels_map,
        "numTraining": len(training_entries),
        "numTest": 0,
        "training": training_entries,
        "test": []
    }

    with open(task_folder / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Convert PNG (grayscale) dataset to nnU-Net format.")
    parser.add_argument("--images-dir", required=True, type=Path, help="Folder containing PNG images")
    parser.add_argument("--masks-dir", required=True, type=Path, help="Folder containing PNG masks")
    parser.add_argument("--task-id", required=True, type=int, help="Numeric task ID (e.g., 501)")
    parser.add_argument("--task-name", required=True, type=str, help="Short task name (e.g., LARSeg)")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without writing files")
    parser.add_argument("--multiclass", action="store_true", help="Keep integer labels instead of binarizing masks")
    args = parser.parse_args()

    raw_base = "/work/cbaldini/medSAM/code/nnUNet/nnUNet_raw_data_base/"
    if not raw_base:
        raise SystemExit("Error: you must export nnUNet_raw_data_base in the environment.")

    task_folder = Path(raw_base) / f"Task{args.task_id:03d}_{args.task_name}"
    imagesTr = task_folder / "imagesTr"
    imagesTs = task_folder / "imagesTs"
    labelsTr = task_folder / "labelsTr"
    labelsTs = task_folder / "labelsTs"

    for d in (imagesTr, imagesTs, labelsTr, labelsTs):
        if not args.dry_run:
            d.mkdir(parents=True, exist_ok=True)

    mask_map = {p.stem: p for p in Path(args.masks_dir).glob("*.png")}
    img_paths = sorted(Path(args.images_dir).glob("*.png"))

    if not img_paths:
        raise SystemExit(f"No images found in {args.images_dir}")
    if not mask_map:
        raise SystemExit(f"No masks found in {args.masks_dir}")

    labels_map = {"0": "background", "1": "object"}
    training_entries = []
    missing = 0
    pairs = 0

    print(f"\nCreating structure in: {task_folder}\n")
    for img_path in tqdm(img_paths, desc="Convert PNG -> NIfTI"):
        stem = img_path.stem
        mask_png = mask_map.get(stem)
        if mask_png is None:
            missing += 1
            continue

        arr = read_grayscale_png(img_path)
        out_img = imagesTr / f"{stem}.nii.gz"
        out_img1 = imagesTr / f"{stem}_0000.nii.gz"
        save_image_nifti(arr, out_img1, dry_run=args.dry_run)

        lab_out = labelsTr / f"{stem}.nii.gz"
        lab_out1 = labelsTr / f"{stem}_0000.nii.gz"
        save_mask_nifti(mask_png, lab_out, dry_run=args.dry_run, multiclass=args.multiclass)

        training_entries.append({
            "image": f"./imagesTr/{out_img.name}",
            "label": f"./labelsTr/{lab_out.name}"
        })
        pairs += 1

    print(f"Images found: {len(img_paths)} | Missing masks: {missing} | Valid pairs: {pairs}")

    if pairs == 0:
        raise SystemExit("No valid image+mask pairs found. Check file names.")

    if not args.dry_run:
        build_dataset_json(task_folder, training_entries, labels_map)
        print(f"Created: {task_folder/'dataset.json'}")

    print("\nExpected final structure:")
    print(f"{task_folder}/")
    print("  imagesTr/   (case_0000.nii.gz)")
    print("  labelsTr/   (case.nii.gz)")
    print("  imagesTs/   (optional)")
    print("  labelsTs/   (optional)")
    print("  dataset.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
