#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nnU-Net v1 Python runner:
- (optional) plan & preprocess
- training on one or more folds
- collection and printing of validation metrics from summary.json files

Typical usage:
        python run_nnunet.py \
                --task-id 501 --task-name PNGSeg \
                --model 2d --trainer nnUNetTrainerV2 \
                --folds 0 1 2 3 4 \
                --do-preprocess \
                --nnUNet_raw_data_base /path/nnUNet_raw_data_base \
                --nnUNet_preprocessed /path/nnUNet_preprocessed \
                --RESULTS_FOLDER /path/nnUNet_trained_models \
                --gpu 0

Notes:
- This script DOES NOT modify your dataset. It assumes you have already created
    $nnUNet_raw_data_base/TaskXXX_NAME/ with dataset.json, imagesTr/ and labelsTr/.
- Validation metrics are read from the summary*.json files produced by nnU-Net
    (one per fold) and aggregated at the end of the run.
"""

import argparse
import json
import os
import sys
import time
import glob
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

# ----------------------------
# Utilità
# ----------------------------

def run(cmd: List[str], env: Optional[Dict[str, str]] = None) -> None:
    print("\n>>> Running:", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with return code {e.returncode}: {' '.join(cmd)}")
        sys.exit(e.returncode)

def ensure_env(args) -> Dict[str, str]:
    env = os.environ.copy()

    # set (or verify) the environment variables required by nnU-Net
    if args.nnUNet_raw_data_base:
        env["nnUNet_raw_data_base"] = args.nnUNet_raw_data_base
    elif "nnUNet_raw_data_base" not in env:
        sys.exit("[ERROR] you must specify --nnUNet_raw_data_base or export the environment variable.")

    if args.nnUNet_preprocessed:
        env["nnUNet_preprocessed"] = args.nnUNet_preprocessed
    elif "nnUNet_preprocessed" not in env:
        sys.exit("[ERROR] you must specify --nnUNet_preprocessed or export the environment variable.")

    if args.RESULTS_FOLDER:
        env["RESULTS_FOLDER"] = args.RESULTS_FOLDER
    elif "RESULTS_FOLDER" not in env:
        sys.exit("[ERROR] you must specify --RESULTS_FOLDER or export the environment variable.")

    # opzionale: forza GPU specifica
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    return env

def task_name_full(task_id: int, task_name: str) -> str:
    return f"Task{task_id:03d}_{task_name}"

def find_summary_jsons(results_folder: Path, task_full: str, trainer: str, model: str) -> List[Path]:
    """
        The results layout in nnU-Net v1 may vary depending on the plans.
        We search for summary.json files robustly using the pattern:
            RESULTS_FOLDER/**/{task_full}/**/fold_*/summary*.json
        and filter to those that belong to the requested 'model' (2d/3d_fullres/3d_lowres)
    """
    pattern = str(results_folder / "**" / task_full / "**" / "fold_*" / "summary*.json")
    candidates = [Path(p) for p in glob.glob(pattern, recursive=True)]

    # opzionale: tieni solo i path che includono /{model}/ e il trainer
    filtered: List[Path] = []
    for p in candidates:
        p_str = str(p)
        if f"/{model}/" in p_str.replace("\\", "/") and trainer in p_str:
            filtered.append(p)
    if not filtered:
        # se non trovato, restituisci i candidates grezzi (magari la struttura è diversa)
        return candidates
    return filtered

def load_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARNING] unable to read {path}: {e}")
        return None

def aggregate_metrics(summary_files: List[Path]) -> Dict[str, Any]:
    """
    Aggregate some common metrics (e.g. mean Dice) from summary*.json files.
    The format can vary: try keys like 'mean' / 'mean_foreground' or known dicts.
    """
    per_fold = []
    for s in summary_files:
        data = load_json(s)
        if not data:
            continue

        # euristiche comuni in nnU-Net summary.json
        fold_metrics = {
            "summary_file": str(s),
            "mean_dice": None,
            "class_dice": None
        }

        # Some summaries have 'mean' or 'average' or per-class 'Dice'.
        # Try multiple possible keys.
        possible_keys = [
            ("mean", float),
            ("mean_dice", float),
            ("average", float),
            ("foreground_mean", float),
            ("mean_foreground", float)
        ]
        for k, _t in possible_keys:
            if k in data and isinstance(data[k], (float, int)):
                fold_metrics["mean_dice"] = float(data[k])
                break

        # class-wise: look for dictionaries with dice per label (e.g. "0", "1", or "background", "organ", ...)
        class_keys = ["foreground_mean_Dice_per_class", "per_class", "per_class_Dice", "dice_per_class", "Dice"]
        for ck in class_keys:
            if ck in data and isinstance(data[ck], dict):
                fold_metrics["class_dice"] = data[ck]
                break

        per_fold.append(fold_metrics)

    # aggrega media sui fold per la mean_dice
    mean_over_folds = None
    vals = [f["mean_dice"] for f in per_fold if f["mean_dice"] is not None]
    if vals:
        mean_over_folds = sum(vals) / len(vals)

    return {"per_fold": per_fold, "mean_over_folds": mean_over_folds}

# ----------------------------
# Pipeline principali
# ----------------------------

def do_preprocess(env, task_id: int) -> None:
    cmd = ["nnUNet_plan_and_preprocess", "-t", "Task501_LARrgb1Seg", "--verify_dataset_integrity"]
    run(cmd, env)

def do_train(env, model: str, trainer: str, task_full: str, fold: int, npz: bool = False, deterministic: bool = False, pretrained_weights: Optional[str] = None) -> None:
    cmd = ["nnUNet_train", model, trainer, task_full, str(fold)]
    if npz:
        cmd.append("--npz")
    if deterministic:
        cmd.append("--deterministic")
    if pretrained_weights:
        cmd += ["--pretrained_weights", str(pretrained_weights)]
    run(cmd, env)

def main():
    parser = argparse.ArgumentParser(description="nnU-Net v1 Python runner (preprocess, training, validation).")
    parser.add_argument("--task-id", type=int, required=True, help="Numeric task ID (e.g. 501)")
    parser.add_argument("--task-name", type=str, required=True, help="Task name (e.g. PNGSeg)")
    parser.add_argument("--model", type=str, default="2d", choices=["2d", "3d_fullres", "3d_lowres"], help="nnU-Net model configuration")
    parser.add_argument("--trainer", type=str, default="nnUNetTrainerV2", help="Trainer (e.g. nnUNetTrainerV2)")
    parser.add_argument("--folds", type=int, nargs="+", default=[0], help="List of folds to train (e.g. 0 1 2 3 4)")
    parser.add_argument("--npz", action="store_true", help="Also save npz files during training")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic mode (slower)")
    parser.add_argument("--pretrained-weights", type=str, default=None, help="Path to pretrained weights")
    parser.add_argument("--do-preprocess", action="store_true", help="Run plan & preprocess before training")
    parser.add_argument("--skip-train", action="store_true", help="Skip training (only collect metrics)")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index for CUDA_VISIBLE_DEVICES (e.g. 0)")
    # override env vars
    parser.add_argument("--nnUNet_raw_data_base", type=str, default=None, help="Path for nnUNet_raw_data_base")
    parser.add_argument("--nnUNet_preprocessed", type=str, default=None, help="Path for nnUNet_preprocessed")
    parser.add_argument("--RESULTS_FOLDER", type=str, default=None, help="Path for RESULTS_FOLDER")
    args = parser.parse_args()

    env = ensure_env(args)
    task_full = task_name_full(args.task_id, args.task_name)
    results_folder = Path(env["RESULTS_FOLDER"])

    print(f"\n[INFO] Task: {task_full}")
    print(f"[INFO] Model: {args.model} | Trainer: {args.trainer} | Folds: {args.folds}")
    print(f"[INFO] RESULTS_FOLDER: {results_folder}")
    if args.gpu is not None:
        print(f"[INFO] CUDA_VISIBLE_DEVICES = {env.get('CUDA_VISIBLE_DEVICES')}")

    # 1) preprocess (optional)
    if args.do_preprocess:
        do_preprocess(env, args.task_id)

    # 2) training
    if not args.skip_train:
        for f in args.folds:
            t0 = time.time()
            print(f"\n===== TRAINING FOLD {f} =====")
            do_train(env, args.model, args.trainer, task_full, f, npz=args.npz, deterministic=args.deterministic)
            dt = time.time() - t0
            print(f"===== END FOLD {f} (duration: {dt/60:.1f} min) =====")

    # 3) raccolta metriche (validation) dai summary.json prodotti
    print("\n[INFO] Searching for summary*.json files with validation results...")
    summaries = find_summary_jsons(results_folder, task_full, args.trainer, args.model)
    if not summaries:
        print("[WARNING] No summary*.json files found. Check paths under RESULTS_FOLDER after training.")
        sys.exit(0)

    print(f"[INFO] Found {len(summaries)} summary files:")
    for s in summaries:
        print(" -", s)

    agg = aggregate_metrics(summaries)
    print("\n========== VALIDATION RESULTS ==========")
    if agg["mean_over_folds"] is not None:
        print(f"Mean Dice across folds: {agg['mean_over_folds']:.4f}")
    else:
        print("Mean Dice across folds: n/a")

    for i, fold in enumerate(agg["per_fold"]):
        print(f"\n--- Fold {i} ---")
        print(f"file: {fold['summary_file']}")
        print("mean_dice:", fold["mean_dice"])
        if isinstance(fold["class_dice"], dict):
            print("dice per class:")
            for k, v in fold["class_dice"].items():
                try:
                    print(f"  {k}: {float(v):.4f}")
                except Exception:
                    print(f"  {k}: {v}")

    print("\n[OK] Done.")

if __name__ == "__main__":
    main()
