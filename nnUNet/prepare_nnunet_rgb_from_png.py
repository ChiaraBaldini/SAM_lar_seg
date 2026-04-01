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
from typing import Dict, List


def read_png(path: Path) -> np.ndarray:
    arr = iio.imread(path)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            return arr[..., :3].astype(np.float32)
        else:
            return arr[..., 0].astype(np.float32)
    raise ValueError(f"Unsupported PNG format: {path} with shape {arr.shape}")

def is_rgb(arr: np.ndarray) -> bool:
    return arr.ndim == 3 and arr.shape[2] == 3

def save_image_nifti_channels(
    arr: np.ndarray,
    out_prefix: Path,
    dry_run: bool = False
) -> List[Path]:
    """Save image as one or three NIfTI files (_0000, _0001, _0002)"""
    written = []
    affine = np.eye(4, dtype=np.float32)

    if arr.ndim == 2:
        img = nib.Nifti1Image(arr, affine)
        p = Path(f"{out_prefix}_0000.nii.gz")
        if not dry_run:
            nib.save(img, str(p))
        written.append(p)
    elif is_rgb(arr):
        for ch in range(3):
            arr_single=arr[..., ch]
            if arr_single.ndim == 2:
                arr_single = arr_single[:, :, None]
            img = nib.Nifti1Image(arr_single, affine)
            p = Path(f"{out_prefix}_{ch:04d}.nii.gz")
            if not dry_run:
                nib.save(img, str(p))
            written.append(p)
    else:
        img = nib.Nifti1Image(arr[..., 0], affine)
        p = Path(f"{out_prefix}_0000.nii.gz")
        if not dry_run:
            nib.save(img, str(p))
        written.append(p)
    return written

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


def build_dataset_json(
    task_folder: Path,
    training_entries: List[Dict[str, str]],
    use_rgb: bool,
    labels_map: Dict[str, str]
) -> None:
    """Create a valid nnU-Net dataset.json"""
    modality = {"0": "Red", "1": "Green", "2": "Blue"} if use_rgb else {"0": "Gray"}

    ds = {
        "name": task_folder.name.split("_", 1)[1] if "_" in task_folder.name else task_folder.name,
        "description": "Dataset converted from PNG (2D images and masks).",
        "reference": "",
        "licence": "",
        "release": "1.0",
        "tensorImageSize": "2D",
        "modality": modality,
        "labels": labels_map,
        "numTraining": len(training_entries),
        "numTest": 0,
        "training": training_entries,
        "test": []
    }
    with open(task_folder / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(ds, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Convert PNG (grayscale) dataset to nnU-Net format.")
    parser.add_argument("--images-dir", required=True, type=Path, help="Folder containing PNG images")
    parser.add_argument("--masks-dir", required=True, type=Path, help="Folder containing PNG masks")
    parser.add_argument("--task-id", required=True, type=int, help="Numeric task ID (e.g., 501)")
    parser.add_argument("--task-name", required=True, type=str, help="Short task name (e.g., LARSeg)")
    parser.add_argument("--force-rgb", action="store_true", help="Force 3-channel output even if images are grayscale")
    parser.add_argument("--allow-grayscale", action="store_true", help="Allow mixed dataset; if one RGB → dataset is RGB")
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
    
    # Determine whether dataset should be RGB or grayscale
    any_rgb = False
    for p in img_paths[:min(10, len(img_paths))]:
        arr = read_png(p)
        if is_rgb(arr):
            any_rgb = True
            break

    use_rgb = args.force_rgb or any_rgb
    if args.allow_grayscale and any_rgb:
        use_rgb = True

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

        arr = read_png(img_path)
        if use_rgb:
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[..., :3]
        else:
            if arr.ndim == 3 and arr.shape[2] >= 3:
                arr = arr.mean(axis=2)
        mask=iio.imread(mask_png)
        print("Image arr.shape:", arr.shape)
        print("Mask png shape:", mask.shape)
        out_prefix = imagesTr / stem
        img_out = imagesTr / f"{stem}.nii.gz"
        written_imgs = save_image_nifti_channels(arr, out_prefix, dry_run=args.dry_run)
        for img_path in written_imgs:
            nii_img = nib.load(str(img_path))
            print(f"Saved NIfTI file: {img_path}, shape: {nii_img.shape}")

        lab_out = labelsTr / f"{stem}.nii.gz"
        save_mask_nifti(mask_png, lab_out, dry_run=args.dry_run, multiclass=args.multiclass)
        nii_mask = nib.load(str(lab_out))
        print(f"Saved NIfTI mask: {lab_out}, shape: {nii_mask.shape}")

        first = next((p for p in written_imgs if p.name.endswith("_0000.nii.gz")), written_imgs[0])

        training_entries.append({
            "image": f"./imagesTr/{img_out.name}",
            "label": f"./labelsTr/{lab_out.name}"
        })
        pairs += 1

    print(f"Images found: {len(img_paths)} | Missing masks: {missing} | Valid pairs: {pairs}")

    if pairs == 0:
        raise SystemExit("No valid image+mask pairs found. Check file names.")

    if not args.dry_run:
        build_dataset_json(task_folder, training_entries, use_rgb=True, labels_map=labels_map)
        print(f"Created: {task_folder/'dataset.json'}")

    print("\nExpected final structure:")
    print(f"{task_folder}/")
    print("  imagesTr/   (case_0000.nii.gz [, _0001/_0002 if RGB])")
    print("  labelsTr/   (case.nii.gz)")
    print("  imagesTs/   (optional)")
    print("  labelsTs/   (optional)")
    print("  dataset.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
