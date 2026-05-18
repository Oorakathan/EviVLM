"""
STEP 1 — Data Preparation
==========================
Reads raw .nii volumes + masks from your archive folder.
(.nii IS NIfTI format — they are the same thing)
Extracts 2D slices → PNG images + binary mask PNGs.
Generates text annotations Excel in LViT format (Image, Description columns).

Run once from EviVLM-main/ folder:
    python step1_prepare_data.py
"""

import sys
import numpy as np
import nibabel as nib
from PIL import Image
import openpyxl
import random
from pathlib import Path

BASE_DIR   = Path(__file__).parent
ARCHIVE    = BASE_DIR / "archive" / "MosMedData Chest CT Scans with COVID-19 Related Findings COVID19_1110 1.0"
MASK_DIR   = ARCHIVE / "masks"
STUDY_DIRS = {
    "CT-0": ARCHIVE / "studies" / "CT-0",
    "CT-1": ARCHIVE / "studies" / "CT-1",
    "CT-2": ARCHIVE / "studies" / "CT-2",
    "CT-3": ARCHIVE / "studies" / "CT-3",
    "CT-4": ARCHIVE / "studies" / "CT-4",
}
OUT_ROOT    = BASE_DIR / "datasets" / "MosMedData"
IMG_SIZE    = 224
SEED        = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10

# Text annotations matching LViT/EviVLM paper format for MosMedData+
# Column names must be "Image" and "Description" (what read_text() in utils_train.py expects)
SEVERITY_TEXT = {
    "CT-0": "No pulmonary infection, no infected areas, normal lung.",
    "CT-1": "Unilateral pulmonary infection, one infected area, lower right lung.",
    "CT-2": "Bilateral pulmonary infection, two infected areas, middle left lung and upper right lung.",
    "CT-3": "Bilateral pulmonary infection, multiple infected areas, all left lung and lower right lung.",
    "CT-4": "Bilateral pulmonary infection, extensive infected areas, all left lung and all right lung.",
}


def load_nii_slices(nii_path):
    """Load .nii volume → list of 2D axial slices."""
    vol = nib.load(str(nii_path)).get_fdata()
    return [vol[:, :, i] for i in range(vol.shape[2])]


def normalize_to_uint8(arr2d):
    arr = arr2d.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-5:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def save_image_png(arr2d, path):
    gray = normalize_to_uint8(arr2d)
    img  = Image.fromarray(gray, mode='L').convert('RGB')
    img  = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img.save(str(path))


def save_mask_png(arr2d, path):
    binary = (arr2d > 0).astype(np.uint8) * 255
    mask   = Image.fromarray(binary, mode='L')
    mask   = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    mask.save(str(path))


def write_split(pairs, split_folder, text_filename):
    img_dir  = OUT_ROOT / split_folder / "img"
    mask_dir = OUT_ROOT / split_folder / "labelcol"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Image", "Description"])   # exact column names utils_train.py expects

    print(f"\n  Writing {split_folder} ({len(pairs)} slices) ...")
    for i, (fname, img_slice, mask_slice, ct_grade) in enumerate(pairs):
        save_image_png(img_slice, img_dir / f"{fname}.png")
        save_mask_png(mask_slice, mask_dir / f"{fname}.png")
        ws.append([fname, SEVERITY_TEXT[ct_grade]])
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(pairs)}")

    wb.save(str(OUT_ROOT / split_folder / text_filename))
    print(f"    ✓ Text saved → {text_filename}")


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    mask_files = sorted(MASK_DIR.glob("*.nii"))
    print(f"Found {len(mask_files)} mask files in {MASK_DIR}")
    if not mask_files:
        sys.exit(f"ERROR: No .nii files found. Check path:\n  {MASK_DIR}")

    all_pairs = []
    for mf in mask_files:
        study_id = mf.stem.replace("_mask", "")

        vol_path = ct_grade = None
        for grade, sdir in STUDY_DIRS.items():
            cand = sdir / f"{study_id}.nii"
            if cand.exists():
                vol_path, ct_grade = cand, grade
                break

        if vol_path is None:
            print(f"  WARNING: no volume for {study_id}, skipping")
            continue

        print(f"  Loading {study_id} [{ct_grade}]")
        img_slices  = load_nii_slices(vol_path)
        mask_slices = load_nii_slices(mf)

        if len(img_slices) != len(mask_slices):
            print(f"  WARNING: depth mismatch for {study_id}, skipping")
            continue

        for idx in range(len(img_slices)):
            if mask_slices[idx].max() > 0:    # only slices that have lesion pixels
                fname = f"{study_id}_slice{idx:03d}"
                all_pairs.append((fname, img_slices[idx], mask_slices[idx], ct_grade))

    print(f"\nTotal lesion slices: {len(all_pairs)}")
    if not all_pairs:
        sys.exit("ERROR: No valid slices found.")

    random.shuffle(all_pairs)
    n       = len(all_pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    write_split(all_pairs[:n_train],           "Train_Folder", "Train_text.xlsx")
    write_split(all_pairs[n_train:n_train+n_val], "Val_Folder",  "Val_text.xlsx")
    write_split(all_pairs[n_train+n_val:],     "Test_Folder",  "Test_text.xlsx")

    print(f"\n✅ Done. Dataset at: {OUT_ROOT}")


if __name__ == "__main__":
    main()
