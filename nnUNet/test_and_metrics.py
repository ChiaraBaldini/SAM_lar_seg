#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_nnunet.py
------------------
Evaluate a trained nnU-Net v1 model:
- (optional) run nnUNet_predict to generate predictions
- compute IoU and Dice vs ground-truth masks
- supports binary (default) and multi-class
- writes a CSV with per-case metrics and prints a summary

Requirements: nibabel, numpy, pandas (for CSV)
pip install nibabel numpy pandas tqdm

Examples:

# 1) Run prediction (2D model) + evaluate against labels
python evaluate_nnunet.py \
  --images-dir /path/nnUNet_raw_data_base/Task501_LARSeg/imagesTr \
  --labels-dir /path/nnUNet_raw_data_base/Task501_LARSeg/labelsTr \
  --preds-dir /tmp/preds_Task501_LARSeg \
  --task-name Task501_LARSeg \
  --model 2d --trainer nnUNetTrainerV2 --folds 0 \
  --run-predict \
  --nnUNet_raw_data_base /path/nnUNet_raw_data_base \
  --nnUNet_preprocessed /path/nnUNet_preprocessed \
  --RESULTS_FOLDER /path/nnUNet_trained_models

# 2) Evaluate only (predictions already computed)
python evaluate_nnunet.py \
  --labels-dir /path/nnUNet_raw_data_base/Task501_LARSeg/labelsTr \
  --preds-dir /tmp/preds_Task501_LARSeg

Notes:
- For 2D nnU-Net inputs, images are case_0000.nii.gz; predictions will be named case.nii.gz
- Binary evaluation uses foreground=1; multi-class averages over classes {1..K}
"""

import os
import csv
import json
import glob
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib
from tqdm import tqdm
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# ------------------ metrics ------------------ #

def dice_coef(a: np.ndarray, b: np.ndarray) -> float:
    """Binary Dice on {0,1} arrays."""
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return (2.0 * inter / denom) if denom > 0 else 1.0  # perfect if both empty

def iou_coef(a: np.ndarray, b: np.ndarray) -> float:
    """Binary IoU on {0,1} arrays."""
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return (inter / union) if union > 0 else 1.0  # perfect if both empty

def per_class_metrics(pred: np.ndarray, gt: np.ndarray, classes: List[int]) -> Dict[int, Tuple[float,float]]:
    """Return {class_id: (dice, iou)} for given classes (exclude 0 typically)."""
    out = {}
    for c in classes:
        p = (pred == c)
        g = (gt == c)
        out[c] = (dice_coef(p, g), iou_coef(p, g))
    return out


# ------------------ helpers ------------------ #

def run_cmd(cmd: List[str], env: Optional[Dict[str, str]] = None):
    print("\n>>> Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

def ensure_env(args) -> Dict[str,str]:
    env = os.environ.copy()
    for var in ("nnUNet_raw_data_base", "nnUNet_preprocessed", "RESULTS_FOLDER"):
        val = getattr(args, var, None)
        if val:
            env[var] = val
    return env

def infer_case_stem_from_image_path(p: Path) -> str:
    """
    For nnU-Net 2D images: images are usually <case>_0000.nii.gz.
    The prediction will be <case>.nii.gz
    """
    name = p.name
    if name.endswith(".nii.gz") and name[:-7].endswith("_0000"):
        return name[:-7].rsplit("_", 1)[0]  # remove "_0000"
    if name.endswith(".nii.gz"):
        return name[:-7]  # without .nii.gz
    return p.stem

def pair_preds_labels(preds_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path, str]]:
    """
    Build (pred_path, label_path, case_id) pairs by matching stems.
    Expects predictions named <case>.nii.gz and labels <case>.nii.gz.
    """
    preds = {Path(p).stem: Path(p) for p in glob.glob(str(preds_dir / "*.nii.gz"))}
    labels = {Path(p).stem: Path(p) for p in glob.glob(str(labels_dir / "*.nii.gz"))}
    pairs = []
    for case in sorted(labels.keys()):
        if case in preds:
            pairs.append((preds[case], labels[case], case))
    return pairs

def load_nii(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata()
    # accept (H,W) or (H,W,1) or (Z,H,W) or (H,W,Z). Reduce to 2D or 3D consistently.
    if data.ndim == 3 and 1 in data.shape:
        data = np.squeeze(data)  # collapse singleton
    return data

def maybe_argmax_if_prob(pred_arr: np.ndarray) -> np.ndarray:
    """
    If predictions are softmax probabilities (C,H,W[,(Z)]), take argmax.
    If predictions are label maps already, return as-int.
    """
    if pred_arr.ndim >= 3 and pred_arr.dtype.kind in "fc" and pred_arr.shape[0] in (2,3,4,5):
        # Heuristic: channel-first probs (C, H, W[, Z])
        return np.argmax(pred_arr, axis=0).astype(np.int16)
    return np.rint(pred_arr).astype(np.int16)  # round just in case


def _prepare_display_2d(arr: np.ndarray) -> np.ndarray:
    """
    Prepare a 2D array for display: if arr is 2D return as float32 normalized to [0,1].
    If arr is 3D, collapse via max-projection along the first axis to obtain a 2D image.
    Works for both intensity images and binary label maps.
    """
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim == 2:
        out = a.astype(np.float32)
    elif a.ndim == 3:
        # robust projection for volumes: use max projection along axis 0
        out = np.max(a, axis=0).astype(np.float32)
    elif a.ndim == 4:
        # channel-first probability maps (C,H,W,Z) or (N,C,H,W): take argmax across channel first
        out = np.max(a.reshape(-1, *a.shape[-2:]), axis=0).astype(np.float32)
    else:
        out = a.astype(np.float32)

    # normalize for visualization (if not binary)
    if out.dtype.kind in "fci":
        mn = float(np.min(out))
        mx = float(np.max(out))
        if mx > mn:
            out = (out - mn) / (mx - mn)
        else:
            out = out - mn
    return out


def _save_case_figure(case: str, img_arr: Optional[np.ndarray], gt_arr: np.ndarray, pred_arr: np.ndarray,
                      out_path: str, fmt: str = "png", alpha: float = 0.5) -> None:
    """Save a side-by-side figure for a single case.
    Layout: [real image | real+GT overlay | real+PRED overlay]
    If `img_arr` is None, layout becomes [GT | PRED].
    """
    try:
        fig = None
        base = _prepare_display_2d(img_arr) if img_arr is not None else None
        gt = _prepare_display_2d(gt_arr)
        pred = _prepare_display_2d(pred_arr)

        if base is None:
            cols = 2
            figsize = (8, 4)
        else:
            cols = 3
            figsize = (12, 4)

        fig, axes = plt.subplots(1, cols, figsize=figsize)
        if cols == 3:
            ax_img, ax_gt, ax_pred = axes
        else:
            ax_gt, ax_pred = axes

        if base is not None:
            ax_img.imshow(base, cmap="gray")
            ax_img.set_title(f"{case}: image")
            ax_img.axis("off")

            ax_gt.imshow(base, cmap="gray")
            ax_gt.imshow(gt, cmap="Reds", alpha=alpha)
            ax_gt.set_title("GT overlay")
            ax_gt.axis("off")

            ax_pred.imshow(base, cmap="gray")
            ax_pred.imshow(pred, cmap="Blues", alpha=alpha)
            ax_pred.set_title("Prediction overlay")
            ax_pred.axis("off")
        else:
            ax_gt.imshow(gt, cmap="Reds")
            ax_gt.set_title(f"{case}: GT")
            ax_gt.axis("off")

            ax_pred.imshow(pred, cmap="Blues")
            ax_pred.set_title("Prediction")
            ax_pred.axis("off")

        plt.tight_layout()
        # ensure extension
        out_file = str(out_path)
        if not out_file.lower().endswith(f".{fmt.lower()}"):
            out_file = out_file + "." + fmt
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        # don't crash evaluation for figure saving errors; print a warning
        print(f"Warning: failed to save figure for case {case}")

def load_image_nifti_channels(
    input_prefix: Path
) -> np.ndarray:
    """Load one or three NIfTI files (_0000, _0001, _0002) and return as RGB array."""
    channels = []

    for ch in range(3):
        file_path = Path(f"{input_prefix}_{ch:04d}.nii.gz")
        print(file_path)
        if file_path.exists():
            img = nib.load(str(file_path))
            data = img.get_fdata()
            if data.ndim == 3 and data.shape[2] == 1:
                data = data[:, :, 0]  # Remove singleton dimension
            channels.append(data)

    if len(channels) == 1:
        # Single channel (grayscale)
        return np.stack(channels, axis=-1)
    # elif len(channels) == 3:
    #     # RGB channels
    #     return np.stack(channels, axis=-1)
    else:
        raise ValueError(f"Expected 1 or 3 NIfTI files, but found {len(channels)}")

# ------------------ prediction ------------------ #

def run_nnunet_predict(args, env):
    """
    Call nnUNet_predict for a given folder of images.
    - For 2D: input images should be case_0000.nii.gz (nnU-Net requirement).
    - Outputs will be <case>.nii.gz into --preds-dir
    """
    cmd = [
        "nnUNet_predict",
        "-i", str(args.images_dir),
        "-o", str(args.preds_dir),
        "-t", args.task_name,      # e.g., Task501_LARgSeg (full name)
        "-m", args.model,          # 2d | 3d_fullres | 3d_lowres
        "-f", ",".join(map(str, args.folds))
    ]
    if args.trainer:
        cmd += ["-tr", args.trainer]
    if args.gpu is not None:
        env = env.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run_cmd(cmd, env)


# ------------------ main ------------------ #

def main():
    ap = argparse.ArgumentParser(description="Evaluate nnU-Net predictions with IoU and Dice.")
    # I/O
    ap.add_argument("--images-dir", type=Path, help="Folder with input images for prediction (NIfTI). Required if --run-predict.")
    ap.add_argument("--labels-dir", type=Path, required=True, help="Folder with ground-truth labels (NIfTI).")
    ap.add_argument("--preds-dir", type=Path, required=True, help="Folder for predictions (existing or to be created).")
    # nnU-Net cfg for prediction
    ap.add_argument("--task-name", type=str, help="Full task name, e.g. Task501_LARgSeg (required if --run-predict).")
    ap.add_argument("--model", type=str, default="2d", choices=["2d","3d_fullres","3d_lowres"])
    ap.add_argument("--trainer", type=str, default="nnUNetTrainerV2")
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--run-predict", action="store_true", help="Run nnUNet_predict before evaluation.")
    # env overrides
    ap.add_argument("--nnUNet_raw_data_base", type=str, default=None)
    ap.add_argument("--nnUNet_preprocessed", type=str, default=None)
    ap.add_argument("--RESULTS_FOLDER", type=str, default=None)
    # eval options
    ap.add_argument("--multiclass", action="store_true", help="Compute mean metrics over classes {1..K}")
    ap.add_argument("--csv-out", type=Path, default=None, help="Path to save per-case metrics as CSV.")
    # figure saving options
    ap.add_argument("--save-figs", action="store_true", help="Save per-case figures (image, GT overlay, prediction).")
    ap.add_argument("--figs-dir", type=Path, default=None, help="Folder to save per-case figures. Defaults to <preds_dir>/figs when --save-figs is used.")
    ap.add_argument("--fig-format", type=str, default="png", choices=["png","jpg","jpeg"], help="Output image format for saved figures.")
    ap.add_argument("--overlay-alpha", type=float, default=0.5, help="Alpha transparency for mask overlays (0..1).")
    args = ap.parse_args()

    # sanity
    if args.run_predict:
        if args.images_dir is None or args.task_name is None:
            raise SystemExit("--run-predict requires --images-dir and --task-name.")
        args.preds_dir.mkdir(parents=True, exist_ok=True)

    env = ensure_env(args)

    # run prediction if requested
    if args.run_predict:
        run_nnunet_predict(args, env)

    # pair predictions with labels
    if not args.preds_dir.exists():
        raise SystemExit(f"Predictions folder not found: {args.preds_dir}")

    pairs = pair_preds_labels(args.preds_dir, args.labels_dir)
    if not pairs:
        # try to help: maybe predictions have suffixes?
        raise SystemExit(f"No (pred,label) pairs found in {args.preds_dir} vs {args.labels_dir}. "
                         "Check filenames. Expected <case>.nii.gz in both.")

    rows = []
    all_dice = []
    all_iou = []
    all_mc_dice = []
    all_mc_iou = []

    for p_pred, p_lab, case in tqdm(pairs, desc="Evaluating"):
        pred = load_nii(p_pred)
        lab  = load_nii(p_lab)

        # ensure same spatial shape
        if pred.shape != lab.shape:
            # if predictions are probabilities (C,H,W...), take argmax first (may drop a dim)
            pred = maybe_argmax_if_prob(pred)
            if pred.shape != lab.shape:
                raise RuntimeError(f"Shape mismatch for {case}: pred {pred.shape} vs label {lab.shape}")

        # convert to int labels
        pred = maybe_argmax_if_prob(pred)
        lab  = np.rint(lab).astype(np.int16)
        print(f"Pred size: {pred.shape}, Label size: {lab.shape}")

        # optionally save a figure with image / GT overlay / prediction
        if args.save_figs:
            # determine output dir
            figs_dir = args.figs_dir if args.figs_dir is not None else (args.preds_dir / "figs")
            try:
                figs_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            img_arr = None
            # try to locate original image file in images_dir if provided
            if args.images_dir is not None:
                imgs = {Path(p).stem: Path(p) for p in glob.glob(str(args.images_dir / "*.nii.gz"))}
                print(imgs)
                # look for files starting with case or case_0000
                patterns = [str(args.images_dir / f"{case}*.nii*"), str(args.images_dir / f"{case}_*000*.nii*"),str(args.images_dir / f"{case}_0000.nii.gz")]
                found = []
                for pat in patterns:
                    found.extend(glob.glob(pat))
                if found:
                    print(found)
                    img_arr = load_nii(Path(found[0]))

                img_arr = load_nii(imgs[case.split("_0001.nii")[0].split('.nii')[0]+'_0000.nii'])
                # print(case)
                # print(case.split(".nii")[0])
                # print(case.split(".nii")[0]+'_0001.nii')
                # print(str(imgs[case.split(".nii")[0]+'_0001.nii']).split('_0001.nii')[0])
                # img_arr = load_image_nifti_channels(str(imgs[case.split(".nii")[0]+'_0001.nii']).split('_0001.nii')[0])

            out_file = figs_dir / case
            _save_case_figure(case, img_arr, lab, pred, out_file, fmt=args.fig_format, alpha=args.overlay_alpha)
            # _save_case_figure(case, img_arr[:,:,0], lab[:, :, 0], pred[:,:,0], out_file, fmt=args.fig_format, alpha=args.overlay_alpha)
        
            # # RGBBBBB
            # from matplotlib.colors import ListedColormap
            # # Normalize img_arr for visualization
            # img_norm = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min()) if img_arr.max() > img_arr.min() else img_arr
            # print(img_norm.shape)
            # # Define colormap for labels and predictions
            # cmap = ListedColormap(['black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta'])
            # # # Convert lab and pred to RGB
            # # lab_rgb = cmap(lab.astype(int))[:, :, :3,0]  # Drop alpha channel
            # binary_pred = (pred > 0.5).astype(int)
            # binary_lab = (lab > 0.5).astype(int)
            # # pred_rgb = cmap(pred.astype(int))[:, :, :3,0]
            # # print(lab_rgb.shape)
            # # print(pred_rgb.shape)
            # # Overlay lab and pred on img_arr
            # lab_overlay = (0.5 * img_norm + 0.5 * binary_lab).clip(0, 1)
            # pred_overlay = (0.5 * img_norm + 0.5 * binary_pred).clip(0, 1)
            # # Visualize and save
            # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            # axes[0].imshow(img_norm)
            # axes[0].set_title('Original Image')
            # axes[0].axis('off')
            # axes[1].imshow(lab_overlay)
            # axes[1].set_title('Ground Truth Overlay')
            # axes[1].axis('off')
            # axes[2].imshow(pred_overlay)
            # axes[2].set_title('Prediction Overlay')
            # axes[2].axis('off')
            # plt.tight_layout()
            # plt.savefig(figs_dir / f"{case}_rgb_visualization.png", dpi=150)
            # plt.close()
            # print(f"Saved RGB visualization for {case} to {figs_dir}")
        
        if args.multiclass:
            max_class = max(int(pred.max()), int(lab.max()))
            classes = [c for c in range(1, max_class+1)]
            if not classes:
                # no foreground at all -> define perfect
                mc_dice = 1.0
                mc_iou  = 1.0
                per_class = {}
            else:
                per_class = per_class_metrics(pred, lab, classes)
                mc_dice = float(np.mean([d for (d,_) in per_class.values()])) if per_class else 1.0
                mc_iou  = float(np.mean([i for (_,i) in per_class.values()])) if per_class else 1.0

            rows.append({
                "case": case,
                "mode": "multiclass",
                "dice_mean": mc_dice,
                "iou_mean": mc_iou,
                "per_class": json.dumps({str(k): {"dice": v[0], "iou": v[1]} for k,v in per_class.items()}),
            })
            all_mc_dice.append(mc_dice)
            all_mc_iou.append(mc_iou)
        else:
            # binary foreground == 1
            d = dice_coef(pred == 1, lab == 1)
            i = iou_coef(pred == 1, lab == 1)
            rows.append({"case": case, "mode": "binary", "dice": d, "iou": i})
            all_dice.append(d)
            all_iou.append(i)

    # summary
    if args.multiclass:
        mean_dice = float(np.mean(all_mc_dice)) if all_mc_dice else float("nan")
        mean_iou  = float(np.mean(all_mc_iou))  if all_mc_iou else float("nan")
        print(f"\n=== Summary (multiclass, foreground classes) ===")
        print(f"Cases: {len(rows)}")
        print(f"Mean Dice: {mean_dice:.4f}")
        print(f"Mean IoU : {mean_iou:.4f}")
    else:
        mean_dice = float(np.mean(all_dice)) if all_dice else float("nan")
        mean_iou  = float(np.mean(all_iou))  if all_iou else float("nan")
        print(f"\n=== Summary (binary, class=1) ===")
        print(f"Cases: {len(rows)}")
        print(f"Mean Dice: {mean_dice:.4f}")
        print(f"Mean IoU : {mean_iou:.4f}")

    # CSV
    if args.csv_out is None:
        args.csv_out = args.preds_dir / ("metrics_multiclass.csv" if args.multiclass else "metrics_binary.csv")
    df = pd.DataFrame(rows)
    df.to_csv(args.csv_out, index=False)
    print(f"\nSaved per-case metrics to: {args.csv_out}")

if __name__ == "__main__":
    main()
