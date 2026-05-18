# EviVLM — Complete Setup, Training & Inference Guide
# =====================================================
# MosMedData+ COVID-19 CT Segmentation
# Works on: CPU (your i5-13500H) and GPU (friend's GTX)
# =====================================================


# ═══════════════════════════════════════════════════════
# SECTION 0 — PROJECT FOLDER STRUCTURE
# ═══════════════════════════════════════════════════════

# EviVLM-main/
# ├── archive/
# │   └── MosMedData Chest CT Scans with COVID-19.../
# │       ├── masks/          <- 50 .nii mask files
# │       └── studies/
# │           ├── CT-0/  CT-1/  CT-2/  CT-3/  CT-4/
# ├── code/
# │   ├── backbones/bert_model/
# │   │   ├── bert_config.json      <- already in repo
# │   │   └── Bio_ClinicalBERT/     <- YOU download this (Section 2)
# │   │       ├── pytorch_model.bin
# │   │       ├── config.json
# │   │       └── vocab.txt
# │   └── nets/EviVLM.py            <- original, do NOT edit
# ├── step1_prepare_data.py
# ├── step2a_cache.py
# ├── step2b_train.py
# ├── step3_infer.py
# ├── download_bert.py
# └── requirements.txt


# ═══════════════════════════════════════════════════════
# SECTION 1 — INSTALL DEPENDENCIES
# ═══════════════════════════════════════════════════════

# For GPU machine — run IN THIS ORDER:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#   pip install -r requirements.txt

# For CPU machine (your i5-13500H) — run IN THIS ORDER:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#   pip install -r requirements.txt

# NOTE: The code auto-detects GPU vs CPU automatically.
#       GPU -> batch_size=16, num_workers=4, pin_memory=True
#       CPU -> batch_size=2,  num_workers=0, pin_memory=False
#       Same scripts, same commands, work on both machines.


# ═══════════════════════════════════════════════════════
# SECTION 2 — DOWNLOAD BioClinicalBERT
# ═══════════════════════════════════════════════════════

#   python download_bert.py
#
# Downloads ~437 MB to:
#   code/backbones/bert_model/Bio_ClinicalBERT/
#       pytorch_model.bin   (436 MB)
#       config.json
#       vocab.txt
#
# NOTE: The TensorFlow .ckpt files you downloaded from GitHub
# are a different format and cannot be used here. This script
# downloads the correct PyTorch version from HuggingFace.


# ═══════════════════════════════════════════════════════
# SECTION 3 — STEP 1: PREPARE DATASET  (~5-10 min)
# ═══════════════════════════════════════════════════════

#   python step1_prepare_data.py
#
# What it does:
#   - Reads all 50 annotated .nii volumes from archive/
#   - Extracts 2D axial slices containing lesion pixels only
#   - Saves 224x224 RGB PNG images + binary mask PNGs
#   - Generates text annotation Excel files
#   - Splits: Train 70% / Val 10% / Test 20%
#
# Output:
#   datasets/MosMedData/
#   ├── Train_Folder/  img/  labelcol/  Train_text.xlsx
#   ├── Val_Folder/    img/  labelcol/  Val_text.xlsx
#   └── Test_Folder/   img/  labelcol/  Test_text.xlsx
#
# Text annotations (LViT/EviVLM paper format):
#   CT-0: "No pulmonary infection, no infected areas, normal lung."
#   CT-1: "Unilateral pulmonary infection, one infected area, lower right lung."
#   CT-2: "Bilateral pulmonary infection, two infected areas, middle left lung and upper right lung."
#   CT-3: "Bilateral pulmonary infection, multiple infected areas, all left lung and lower right lung."
#   CT-4: "Bilateral pulmonary infection, extensive infected areas, all left lung and all right lung."
#
# Run this ONCE on any machine. Copy datasets/ to friend's machine if needed.


# ═══════════════════════════════════════════════════════
# SECTION 4 — STEP 2a: CACHE ENCODER FEATURES  (~25-35 min)
# ═══════════════════════════════════════════════════════

#   python step2a_cache.py
#
# What it does:
#   - Loads BioClinicalBERT + UNet encoder
#   - Runs every train/val image through both encoders ONCE
#   - Saves output feature tensors to disk
#   - After this, heavy encoders NEVER run during training
#   - Images resized to 128x128 (decoder 3.5x faster vs 224x224)
#   - Auto-uses GPU if available for faster caching
#
# Output:
#   datasets/MosMedData/cache/
#       train_cache.pt   (~150 MB)
#       val_cache.pt     (~22 MB)
#
# PORTABILITY: Cache is always saved as CPU tensors.
# You can run step2a on YOUR CPU machine, copy the cache/ folder
# to your friend's GPU machine, and run step2b there.
# The cache works on any device.


# ═══════════════════════════════════════════════════════
# SECTION 5 — STEP 2b: TRAIN  (GPU: ~1-2 hrs / CPU: ~11 hrs)
# ═══════════════════════════════════════════════════════

#   python step2b_train.py
#
# What is TRAINABLE (gets updated every batch):
#   EAMG      : vision_nonlocal, text_nonlocal, cross_att
#   EDSL      : BVD loss (no extra params, computed in forward)
#   Decoder   : up4, up3, up2, up1
#   Prob heads: prob_alpha, prob_beta, prob_alpha_2
#   Text proj : global_embed, local_embed
#
# What is FROZEN (loaded from cache, never re-runs):
#   BioClinicalBERT 12-layer transformer body
#   UNet encoder (inc, down1, down2, down3, down4)
#
# Expected speed:
#   GTX 1060 6GB  -> ~1-2 min/epoch   ~2-3 hrs total
#   GTX 1080 8GB  -> ~45 sec/epoch    ~1-1.5 hrs total
#   i5-13500H CPU -> ~8 min/epoch     ~11 hrs total
#
# Output:
#   runs/YYYYMMDD_HHMMSS/
#   ├── checkpoints/
#   │   ├── best_model.pth      <- USE THIS FOR INFERENCE
#   │   ├── checkpoint_ep010.pth
#   │   └── checkpoint_ep020.pth ...
#   └── train.log
#
# Log format per epoch:
#   Epoch [  5/200] Train Loss:0.3421 Dice:0.7234 IoU:0.6891 |
#                   Val Loss:0.3801 Dice:0.7012 IoU:0.6543 |
#                   LR:3.0e-04 Time:112s
#
# Training auto-stops when Val Dice shows no improvement for 50 epochs.
# Best checkpoint is always saved when a new best Val Dice is hit.
#
# If you see CUDA out of memory error:
#   Open step2b_train.py, find line:
#       BATCH_SIZE = 16 if torch.cuda.is_available() else 2
#   Change 16 to 8, save, re-run.


# ═══════════════════════════════════════════════════════
# SECTION 6 — STEP 3: INFERENCE
# ═══════════════════════════════════════════════════════

# -- Single image --
#   python step3_infer.py \
#       --ckpt  runs/YYYYMMDD_HHMMSS/checkpoints/best_model.pth \
#       --image datasets/MosMedData/Test_Folder/img/study_0255_slice045.png \
#       --text  "Bilateral pulmonary infection, two infected areas, lower left lung." \
#       --out   output_mask.png
#
# Output:
#   output_mask.png             <- white pixels = detected lesion
#   output_mask_uncertainty.png <- brighter = model less confident

# -- Full test set + Dice score --
#   python step3_infer.py \
#       --ckpt     runs/YYYYMMDD_HHMMSS/checkpoints/best_model.pth \
#       --test_dir datasets/MosMedData/Test_Folder \
#       --out_dir  inference_results/
#
# Prints: "Test Dice: 77.64%  (over 125 samples)"
# Saves:  inference_results/study_XXXX_sliceYYY_pred.png  (one per test image)
#         inference_results/study_XXXX_sliceYYY_pred_uncertainty.png

# NOTE: Inference also auto-detects GPU. IMG_SIZE is read from the checkpoint
# automatically (128 if trained with step2b, 224 if trained with original code).


# ═══════════════════════════════════════════════════════
# SECTION 7 — SHARING FILES BETWEEN MACHINES
# ═══════════════════════════════════════════════════════

# Recommended workflow (you prepare, friend trains):
#   YOUR MACHINE:
#     python step1_prepare_data.py   -> datasets/
#     python step2a_cache.py         -> datasets/cache/
#
#   COPY TO FRIEND'S MACHINE (USB drive or zip):
#     datasets/                      (~372 MB total)
#     code/backbones/bert_model/Bio_ClinicalBERT/  (437 MB)
#     code/nets/EviVLM.py
#     code/backbones/bert_model/     (bert_config.json, TextEncoder.py, med.py)
#     step2b_train.py
#     step3_infer.py
#     requirements.txt
#
#   FRIEND'S MACHINE (GPU):
#     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#     pip install -r requirements.txt
#     python step2b_train.py
#     python step3_infer.py --ckpt ...


# ═══════════════════════════════════════════════════════
# SECTION 8 — TROUBLESHOOTING
# ═══════════════════════════════════════════════════════

# "Cache not found: ...train_cache.pt"
#   Run step2a_cache.py first.

# "No .nii files found"
#   Check archive/ folder path — it has spaces in the name, must match exactly.

# "CUDA out of memory"
#   In step2b_train.py change BATCH_SIZE from 16 to 8 on the GPU line.

# "Some weights not used when initializing BertModel" (long warning)
#   NORMAL. BERT was pretrained with MLM heads not used in EviVLM. Ignore it.

# "Some weights newly initialized: bert.encoder.layer.X.crossattention..."
#   NORMAL. Cross-attention layers are new and will be trained. Ignore it.

# UnicodeEncodeError on Windows console
#   Run: set PYTHONIOENCODING=utf-8  (before running any script)
#   Already handled inside setup_logger() but this env var is a safe backup.

# Val Dice stays near 0 for first 5-10 epochs
#   Normal — model needs warmup. Wait for epoch 15-20 before judging.

# Training very slow on CPU
#   Make sure you ran step2a_cache.py first.
#   If still slow, check Task Manager — Python should use all CPU cores.


# ═══════════════════════════════════════════════════════
# SECTION 9 — ALL COMMANDS QUICK REFERENCE
# ═══════════════════════════════════════════════════════

# GPU install:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#   pip install -r requirements.txt

# CPU install:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#   pip install -r requirements.txt

# Download BERT:
#   python download_bert.py

# Prepare data (once):
#   python step1_prepare_data.py

# Cache encoders (once):
#   python step2a_cache.py

# Train:
#   python step2b_train.py

# Infer single image:
#   python step3_infer.py --ckpt runs/.../best_model.pth --image path/img.png --text "..." --out mask.png

# Infer full test set:
#   python step3_infer.py --ckpt runs/.../best_model.pth --test_dir datasets/MosMedData/Test_Folder --out_dir results/
