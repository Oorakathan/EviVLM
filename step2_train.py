"""
STEP 2 — CPU Training (Freeze Encoders, Train EAMG + EDSL + Decoder)
=====================================================================
Uses the ORIGINAL EviVLM class from nets/EviVLM.py exactly as written.
Only two things are frozen:
  1. BioClinicalBERT weights (self.text_encoder.model)
  2. U-Net encoder layers (inc, down1, down2, down3, down4)

Everything else trains normally:
  - EAMG (vision_nonlocal, text_nonlocal, cross_att) ← the EAMG
  - EDSL loss (computed inside forward(), returned as loss_sim)  ← the EDSL
  - U-Net decoder (up4, up3, up2, up1)
  - Probability heads (prob_alpha, prob_beta, prob_alpha_2)
  - Text encoder projection heads (global_embed, local_embed)

Key CPU fixes vs original code:
  - device: cpu (not cuda:0)
  - No GradScaler / autocast (GPU only)
  - batch_size: 2 (not 32)
  - num_workers: 0 (not 8)
  - No pin_memory
  - No tensorboard (optional, remove if you want it)

BERT model path fix:
  Your downloaded clinicalBERT goes in:
  EviVLM-main/code/backbones/bert_model/Bio_ClinicalBERT/
  The script patches the hardcoded absolute path in TextEncoder_Bert automatically.

Usage (from EviVLM-main/ folder):
    python step2_train.py
"""

import os
import sys
import time
import random
import logging
import datetime
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torchvision.transforms import functional as TF
from scipy.ndimage import zoom
import cv2
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from scipy import ndimage

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
CODE_DIR    = BASE_DIR / "code"
DATASET_DIR = BASE_DIR / "datasets" / "MosMedData"
BERT_PATH   = str(CODE_DIR / "backbones" / "bert_model" / "Bio_ClinicalBERT")
BERT_CFG    = str(CODE_DIR / "backbones" / "bert_model" / "bert_config.json")

# Add code paths so original imports work
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "backbones" / "bert_model"))
sys.path.insert(0, str(CODE_DIR / "nets"))

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cpu")
BATCH_SIZE  = 2
LR          = 3e-4
WEIGHT_DECAY= 1e-4
EPOCHS      = 200
EARLY_STOP  = 50
SEED        = 42
IMG_SIZE    = 224
TEMPERATURE = 0.07

RUN_DIR  = BASE_DIR / "runs" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CKPT_DIR = RUN_DIR / "checkpoints"


# ─── Logger ───────────────────────────────────────────────────────────────────
def setup_logger():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    # FileHandler needs encoding="utf-8" on Windows to avoid cp1252 crash
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(str(RUN_DIR / "train.log"), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ]
    )
    # Force stdout to UTF-8 on Windows so special chars don't crash the console
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    return logging.getLogger()


# ─── Patch TextEncoder_Bert to use your local BERT path ───────────────────────
# The original code has absolute paths hardcoded to /root/data1/...
# We patch the class after import so it uses your actual path.
def patch_and_load_model():
    """
    Import original EviVLM, patch the BERT path, move to CPU.
    Returns the full model with frozen BERT + frozen U-Net encoder.
    """
    import backbones.bert_model.TextEncoder as te

    # Monkey-patch the path before __init__ runs
    original_init = te.TextEncoder_Bert.__init__

    def patched_init(self, tokenizer=None, emb_dim=768, output_dim=512,
                     hidden_dim=2048, freeze_bert=False):
        # Call original but intercept path attributes
        import torch.nn as nn_inner
        nn_inner.Module.__init__(self)
        self.Bio_ClinicalBERT_path = BERT_PATH
        self.last_n_layers = 1
        self.aggregate_method = "sum"
        self.embedding_dim = emb_dim
        self.output_dim = output_dim
        self.freeze_bert = freeze_bert
        self.agg_tokens = True

        from transformers import BertTokenizer, BertConfig
        from backbones.bert_model.med import BertModel

        self.config    = BertConfig.from_json_file(BERT_CFG)
        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        self.model     = BertModel.from_pretrained(BERT_PATH, config=self.config,
                                                    add_pooling_layer=False)
        if freeze_bert:
            for param in self.model.parameters():
                param.requires_grad = False

        self.idxtoword  = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.global_embed = te.GlobalEmbedding(emb_dim, hidden_dim, output_dim)
        self.local_embed  = te.LocalEmbedding(emb_dim, hidden_dim, output_dim)

    te.TextEncoder_Bert.__init__ = patched_init

    # Now import and build the full EviVLM
    from nets.EviVLM import EviVLM

    # Patch device inside EviVLM (original hardcodes cuda:0)
    original_evivlm_init = EviVLM.__init__

    def patched_evivlm_init(self, n_channels=3, n_classes=1):
        original_evivlm_init(self, n_channels, n_classes)
        self.device = DEVICE   # override cuda:0 → cpu

    EviVLM.__init__ = patched_evivlm_init

    model = EviVLM(n_channels=3, n_classes=1)
    model = model.to(DEVICE)

    # ── Freeze BERT weights ──
    for param in model.text_encoder.model.parameters():
        param.requires_grad = False

    # ── Freeze U-Net encoder layers ──
    for module in [model.inc, model.down1, model.down2, model.down3, model.down4]:
        for param in module.parameters():
            param.requires_grad = False

    frozen   = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable= sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Frozen params:    {frozen:,}")
    print(f"  Trainable params: {trainable:,}  ({trainable/1e6:.1f}M)")
    print(f"  Trainable modules: EAMG (vision_nonlocal, text_nonlocal, cross_att)")
    print(f"                     Decoder (up4,up3,up2,up1)")
    print(f"                     Prob heads (prob_alpha, prob_beta, prob_alpha_2)")
    print(f"                     Text proj (global_embed, local_embed)")

    return model


# ─── Dataset ──────────────────────────────────────────────────────────────────
# Simplified version of ImageToImage2D_val from Load_Dataset_val_SSL.py
# adapted for CPU (no pin_memory issues, no multiprocess file issues)

def read_text(filename):
    """Read text Excel — exactly as in utils_train.py."""
    df = pd.read_excel(filename)
    text = {}
    for i in df.index.values:
        count = len(df.Description[i].split())
        if count < 9:
            df.Description[i] = df.Description[i] + ' EOF XXX' * (9 - count)
        text[df.Image[i]] = df.Description[i]
    return text


def to_long_tensor(pic):
    img = torch.from_numpy(np.array(pic, np.uint8))
    return img.long()


class MosMedDataset(Dataset):
    """
    Loads PNG images + masks from a split folder.
    Returns {'image': [3,224,224], 'label': [224,224]} and filename.
    """
    def __init__(self, split_folder, augment=False):
        self.img_dir  = DATASET_DIR / split_folder / "img"
        self.mask_dir = DATASET_DIR / split_folder / "labelcol"
        self.augment  = augment
        self.files    = sorted([f.stem for f in self.img_dir.glob("*.png")])
        assert len(self.files) > 0, f"No images found in {self.img_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img   = np.array(Image.open(self.img_dir  / f"{fname}.png").convert("RGB"))
        mask  = np.array(Image.open(self.mask_dir / f"{fname}.png").convert("L"))

        # Normalize mask to 0/1
        mask = (mask > 127).astype(np.uint8)

        # Resize to IMG_SIZE
        h, w = img.shape[:2]
        if h != IMG_SIZE or w != IMG_SIZE:
            img  = zoom(img,  (IMG_SIZE/h, IMG_SIZE/w, 1), order=3)
            mask = zoom(mask, (IMG_SIZE/h, IMG_SIZE/w),    order=0)

        # Augmentation (train only)
        if self.augment and random.random() > 0.5:
            k    = np.random.randint(0, 4)
            img  = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()

        if self.augment and random.random() > 0.5:
            angle = np.random.randint(-20, 20)
            img   = ndimage.rotate(img,  angle, order=0, reshape=False)
            mask  = ndimage.rotate(mask, angle, order=0, reshape=False)

        img  = img.astype(np.uint8)
        mask = mask.astype(np.uint8)

        img_t  = TF.to_tensor(Image.fromarray(img))              # [3, 224, 224]
        mask_t = to_long_tensor(Image.fromarray(mask))           # [224, 224]  int64

        return {'image': img_t, 'label': mask_t}, fname


# ─── Losses (from utils_train.py) ─────────────────────────────────────────────

class WeightedDiceBCE(nn.Module):
    """WeightedDiceBCE exactly as in the original utils_train.py."""
    def __init__(self, dice_weight=0.5, BCE_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.BCE_weight  = BCE_weight

    def _dice_loss(self, logit, truth, smooth=1e-5):
        B = len(logit)
        logit = logit.view(B, -1)
        truth = truth.view(B, -1).float()
        w     = truth.detach()
        w     = w * (0.5 - 0.5) + 0.5
        p, t  = w * logit, w * truth
        inter = (p * t).sum(-1)
        union = (p * p).sum(-1) + (t * t).sum(-1)
        dice  = 1 - (2 * inter + smooth) / (union + smooth)
        return dice.mean()

    def _bce_loss(self, logit, truth):
        logit = logit.view(-1)
        truth = truth.view(-1).float()
        loss  = F.binary_cross_entropy(logit, truth, reduction='none')
        pos   = (truth > 0.5).float()
        neg   = (truth < 0.5).float()
        pw    = pos.sum().item() + 1e-12
        nw    = neg.sum().item() + 1e-12
        loss  = (0.4 * pos * loss / pw + 0.6 * neg * loss / nw).sum()
        return loss

    def forward(self, logit, truth):
        # logit: [B,1,224,224] sigmoid output, truth: [B,224,224] int64
        truth = truth.float().unsqueeze(1)      # [B,1,224,224]
        bce   = self._bce_loss(logit, truth)
        dice  = self._dice_loss(logit, truth)
        return self.BCE_weight * bce + self.dice_weight * dice

    def _show_dice(self, logit, truth):
        pred  = (logit > 0.5).float()
        truth = truth.float()
        inter = (pred * truth).sum()
        union = pred.sum() + truth.sum() + 1e-6
        return (2 * inter / union).item()


def kl_divergence(alpha, num_classes, device):
    # lgamma(C) must be a scalar — NOT lgamma(ones * C) which gives shape [1,C]
    # that would make `first` shape [N,C] instead of [N,1], crashing the mean
    S      = torch.sum(alpha, dim=1, keepdim=True)                          # [N,1]
    lgC    = torch.lgamma(torch.tensor(float(num_classes), device=device))  # scalar
    first  = (torch.lgamma(S)                                               # [N,1]
              - lgC                                                          # scalar
              - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True))        # [N,1]
    second = torch.sum((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S)),
                       dim=1, keepdim=True)                                 # [N,1]
    return first + second                                                    # [N,1]


def edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    S        = torch.sum(alpha, dim=1, keepdim=True)
    log_like = torch.sum(target * (torch.digamma(alpha) - torch.digamma(S)), dim=1)
    annealing_coef = min(1.0, epoch_num / annealing_step)
    alpha_tilde    = target + (1 - target) * alpha
    kl_loss        = annealing_coef * kl_divergence(alpha_tilde, num_classes, device)
    return torch.mean(-log_like + kl_loss.squeeze(1))


def get_loss(evidences, evidence_a, target, epoch_num, num_classes, annealing_step, device):
    """Exactly as in utils_train.py."""
    alpha_a  = evidence_a + 1
    loss_acc = edl_digamma_loss(alpha_a, target, epoch_num, num_classes, annealing_step, device)
    for v in range(len(evidences)):
        alpha     = evidences[v] + 1
        loss_acc += edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device)
    return loss_acc / (len(evidences) + 1)


def iou_on_batch(masks, pred):
    pred   = (pred > 0.5).float()
    masks  = masks.float()
    inter  = (pred * masks).sum(dim=(1, 2, 3))
    union  = pred.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - inter + 1e-6
    return (inter / union).mean().item()


# ─── Train / Val loops ────────────────────────────────────────────────────────

def train_one_epoch(loader, model, all_text, criterion, optimizer, epoch, logger):
    model.train()
    loss_sum = dice_sum = iou_sum = 0.0
    n_samples = 0

    for batch, names in tqdm(loader, desc=f"Train Epoch {epoch+1}", leave=False):
        images = batch['image'].to(DEVICE)   # [B, 3, 224, 224]
        masks  = batch['label'].to(DEVICE)   # [B, 224, 224]  int64

        # Lookup text for each image in this batch
        text_str = [all_text[n] for n in names]

        # Forward — full EviVLM forward pass (EAMG + EDSL inside)
        prob_V, prob_L, prob_VL, evi_V, evi_L, evi_VL, loss_sim = model(images, text_str)

        # Evidential loss
        target   = masks.reshape(-1)
        target   = F.one_hot(target, 2).float()
        evi_dict = {0: evi_V, 1: evi_L}
        loss_evi = get_loss(evi_dict, evi_VL, target, epoch + 1,
                            num_classes=2, annealing_step=50, device=DEVICE)

        # Segmentation loss
        loss_seg = criterion(prob_VL, masks)

        # Total loss (matches original paper weighting)
        loss = ((loss_seg + loss_evi) / 2.0) + loss_sim * 0.2

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()

        B         = images.size(0)
        dice      = criterion._show_dice(prob_VL.detach(), masks.unsqueeze(1).float())
        iou       = iou_on_batch(masks.unsqueeze(1).float(), prob_VL.detach())
        loss_sum  += loss.item() * B
        dice_sum  += dice * B
        iou_sum   += iou  * B
        n_samples += B

    return loss_sum/n_samples, dice_sum/n_samples, iou_sum/n_samples


@torch.no_grad()
def val_one_epoch(loader, model, all_text, criterion, epoch, logger):
    model.eval()
    loss_sum = dice_sum = iou_sum = 0.0
    n_samples = 0

    for batch, names in tqdm(loader, desc=f"Val   Epoch {epoch+1}", leave=False):
        images = batch['image'].to(DEVICE)
        masks  = batch['label'].to(DEVICE)
        text_str = [all_text[n] for n in names]

        prob_V, prob_L, prob_VL, evi_V, evi_L, evi_VL, _ = model(images, text_str)

        loss = criterion(prob_VL, masks)
        B    = images.size(0)
        dice = criterion._show_dice(prob_VL, masks.unsqueeze(1).float())
        iou  = iou_on_batch(masks.unsqueeze(1).float(), prob_VL)

        loss_sum  += loss.item() * B
        dice_sum  += dice * B
        iou_sum   += iou  * B
        n_samples += B

    return loss_sum/n_samples, dice_sum/n_samples, iou_sum/n_samples


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("  EviVLM — CPU Training (Frozen Encoders)")
    logger.info("=" * 60)
    logger.info(f"  Device:     {DEVICE}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  LR:         {LR}")
    logger.info(f"  Run dir:    {RUN_DIR}")

    # ── Load text annotations ──
    train_text = read_text(str(DATASET_DIR / "Train_Folder" / "Train_text.xlsx"))
    val_text   = read_text(str(DATASET_DIR / "Val_Folder"   / "Val_text.xlsx"))
    all_text   = {**train_text, **val_text}   # merged dict, keyed by image stem
    logger.info(f"  Text entries: {len(all_text)}")

    # ── Datasets ──
    train_ds = MosMedDataset("Train_Folder", augment=True)
    val_ds   = MosMedDataset("Val_Folder",   augment=False)
    logger.info(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, drop_last=False)

    # ── Model ──
    logger.info("\nLoading model ...")
    model = patch_and_load_model()

    # ── Optimizer — only trainable params ──
    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-5)

    best_dice      = 0.0
    best_epoch     = 0
    no_improve_cnt = 0

    logger.info("\n" + "-" * 60)
    logger.info("Training starts. Frozen: BERT + UNet encoder.")
    logger.info("Trainable: EAMG + decoder + prob heads + text proj heads")
    logger.info("-" * 60)

    for epoch in tqdm(range(EPOCHS), desc="Training", unit="epoch"):
        t0 = time.time()

        train_loss, train_dice, train_iou = train_one_epoch(
            train_loader, model, all_text, criterion, optimizer, epoch, logger)
        val_loss, val_dice, val_iou = val_one_epoch(
            val_loader, model, all_text, criterion, epoch, logger)

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoch [{epoch+1:3d}/{EPOCHS}]  "
            f"Train → Loss:{train_loss:.4f} Dice:{train_dice:.4f} IoU:{train_iou:.4f} | "
            f"Val → Loss:{val_loss:.4f} Dice:{val_dice:.4f} IoU:{val_iou:.4f} | "
            f"LR:{lr_now:.2e}  Time:{elapsed:.0f}s"
        )

        if val_dice > best_dice:
            best_dice  = val_dice
            best_epoch = epoch + 1
            no_improve_cnt = 0
            torch.save({
                'epoch':      epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'val_dice':   val_dice,
                'val_loss':   val_loss,
            }, str(CKPT_DIR / "best_model.pth"))
            logger.info(f"  ✓ Best model saved! Dice: {best_dice:.4f}")
        else:
            no_improve_cnt += 1
            logger.info(f"  No improve ({no_improve_cnt}/{EARLY_STOP}). "
                        f"Best: {best_dice:.4f} @ epoch {best_epoch}")

        if epoch % 10 == 9:
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict()},
                       str(CKPT_DIR / f"checkpoint_ep{epoch+1:03d}.pth"))

        if no_improve_cnt >= EARLY_STOP:
            logger.info(f"\n⏹  Early stop at epoch {epoch+1}")
            break

    logger.info(f"\n✅ Training complete. Best Val Dice: {best_dice:.4f} @ epoch {best_epoch}")
    logger.info(f"   Checkpoint: {CKPT_DIR / 'best_model.pth'}")


if __name__ == "__main__":
    main()