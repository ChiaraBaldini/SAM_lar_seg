#!/usr/bin/env python3
import json, re
from pathlib import Path

ROOT = Path("/work/cbaldini/medSAM/code/nnUNet/nnUNet_raw_data_base/Task501_LARSeg")
imagesTr = ROOT / "imagesTr"
labelsTr = ROOT / "labelsTr"
out_json = ROOT / "dataset.json"

# extract the case_id by removing the channel suffix "_0000/_0001/_0002" before ".nii.gz"
chan_pat = re.compile(r"(.*)_\d{4}\.nii\.gz$")  # capture everything before the final channel suffix

# 1) find all cases from the labels folder
label_files = sorted(labelsTr.glob("*.nii.gz"))
cases = [lf.stem for lf in label_files]  # e.g. '17ff392d-frame_0000'

# 2) check whether RGB triplets exist (channels 0,1,2)
def has_rgb(case_id: str) -> bool:
    return all((imagesTr / f"{case_id}_{i:04d}.nii.gz").exists() for i in (0,1,2))

def has_gray(case_id: str) -> bool:
    return (imagesTr / f"{case_id}_0000.nii.gz").exists()

rgb_possible = all(has_gray(c) for c in cases) and any(has_rgb(c) for c in cases)
# If EVERY case has at least _0000 and AT LEAST ONE has _0001/_0002 -> assume 3-channel RGB
# Otherwise, consider single-channel (grayscale)

# 3) build training list
training = []
missing = []
for c in cases:
    img0 = imagesTr / f"{c}_0000.nii.gz"
    lab = labelsTr / f"{c}.nii.gz"
    if not img0.exists() or not lab.exists():
        missing.append(c)
        continue
    training.append({
        "image": f"./imagesTr/{img0.name}",   # ONLY _0000; nnU-Net will detect _0001/_0002 automatically
        "label": f"./labelsTr/{lab.name}"
    })

if missing:
    print("[WARNING] Missing files for cases:", missing)

# 4) create the correct dataset.json
modality = {"0":"Red","1":"Green","2":"Blue"} if rgb_possible else {"0":"Gray"}
dataset = {
    "name": "LARSeg",
    "description": "Larynx RGB segmentation (converted from PNG).",
    "reference": "",
    "licence": "",
    "release": "1.0",
    "tensorImageSize": "2D",
    "modality": modality,
    "labels": {"0":"background","1":"object"},
    "numTraining": len(training),
    "numTest": 0,
    "training": training,
    "test": []
}

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)
print(f"[OK] Rewrote {out_json} with {len(training)} cases, modality={list(modality.values())}")
