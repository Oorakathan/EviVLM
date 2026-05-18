"""
STEP 2b — Train EAMG + EDSL + Decoder (Fast, from Cache)
=========================================================
Loads pre-cached encoder features from step2a_cache.py.
Encoders (BERT + UNet encoder) never run during training — 0s overhead.
Images are 128x128 — decoder is 3.5x faster than 224x224.

Expected speed: ~8 min/epoch  (~11 hrs total worst case)
vs original:    ~22 min/epoch  (~74 hrs worst case)

Trainable modules (same as before):
  - EAMG  : vision_nonlocal, text_nonlocal, cross_att
  - EDSL  : BVD loss (computed in forward, no extra params)
  - Decoder: up4, up3, up2, up1
  - Prob heads: prob_alpha, prob_beta, prob_alpha_2
  - Text proj : global_embed, local_embed  (projects cached BERT feats)

Run from EviVLM-main/ folder:
    python step2b_train.py
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
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from scipy.ndimage import zoom, rotate
from PIL import Image
from tqdm import tqdm
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
CODE_DIR    = BASE_DIR / "code"
DATASET_DIR = BASE_DIR / "datasets" / "MosMedData"
CACHE_DIR   = DATASET_DIR / "cache"
BERT_PATH   = str(CODE_DIR / "backbones" / "bert_model" / "Bio_ClinicalBERT")
BERT_CFG    = str(CODE_DIR / "backbones" / "bert_model" / "bert_config.json")

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "backbones" / "bert_model"))
sys.path.insert(0, str(CODE_DIR / "nets"))

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# GPU: batch 16 fits comfortably on 2GB+ VRAM.  CPU: keep at 2.
BATCH_SIZE   = 16 if torch.cuda.is_available() else 2
LR           = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS       = 200
EARLY_STOP   = 50
SEED         = 42
IMG_SIZE     = 128

RUN_DIR  = BASE_DIR / "runs" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CKPT_DIR = RUN_DIR / "checkpoints"


# ─── Logger ───────────────────────────────────────────────────────────────────
def setup_logger():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(str(RUN_DIR / "train.log"), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ]
    )
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    return logging.getLogger()


# ─── Load model — full EviVLM with frozen encoders + trainable proj heads ─────
def patch_and_load_model():
    import backbones.bert_model.TextEncoder as te

    def patched_init(self, tokenizer=None, emb_dim=768, output_dim=512,
                     hidden_dim=2048, freeze_bert=False):
        nn.Module.__init__(self)
        self.Bio_ClinicalBERT_path = BERT_PATH
        self.last_n_layers    = 1
        self.aggregate_method = "sum"
        self.embedding_dim    = emb_dim
        self.output_dim       = output_dim
        self.freeze_bert      = freeze_bert
        self.agg_tokens       = True

        from transformers import BertTokenizer, BertConfig
        from backbones.bert_model.med import BertModel

        self.config    = BertConfig.from_json_file(BERT_CFG)
        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        self.model     = BertModel.from_pretrained(BERT_PATH, config=self.config,
                                                    add_pooling_layer=False)
        if freeze_bert:
            for param in self.model.parameters():
                param.requires_grad = False

        self.idxtoword   = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.global_embed = te.GlobalEmbedding(emb_dim, hidden_dim, output_dim)
        self.local_embed  = te.LocalEmbedding(emb_dim, hidden_dim, output_dim)

    te.TextEncoder_Bert.__init__ = patched_init

    from nets.EviVLM import EviVLM

    original_evivlm_init = EviVLM.__init__
    def patched_evivlm_init(self, n_channels=3, n_classes=1):
        original_evivlm_init(self, n_channels, n_classes)
        self.device = DEVICE   # override hardcoded cuda:0 with auto-detected device
    EviVLM.__init__ = patched_evivlm_init

    model = EviVLM(n_channels=3, n_classes=1).to(DEVICE)

    # Freeze BERT body
    for param in model.text_encoder.model.parameters():
        param.requires_grad = False

    # Freeze UNet encoder
    for module in [model.inc, model.down1, model.down2, model.down3, model.down4]:
        for param in module.parameters():
            param.requires_grad = False

    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Frozen params:    {frozen:,}")
    print(f"  Trainable params: {trainable:,}  ({trainable/1e6:.1f}M)")
    return model


# ─── Dataset — loads from cache, runs encoder forward skip ────────────────────
class CachedDataset(Dataset):
    """
    Loads pre-cached encoder features.
    Only the mask is loaded from disk each call (tiny PNG read).
    Applies augmentation to BOTH the spatial cache tensors AND the mask.
    """
    def __init__(self, split, augment=False):
        self.augment  = augment
        self.split    = split
        folder_map    = {"train": "Train_Folder", "val": "Val_Folder"}
        self.mask_dir = DATASET_DIR / folder_map[split] / "labelcol"

        cache_path = CACHE_DIR / f"{split}_cache.pt"
        assert cache_path.exists(), \
            f"Cache not found: {cache_path}\nRun step2a_cache.py first!"

        print(f"  Loading {split} cache ...")
        self.cache = torch.load(str(cache_path), map_location="cpu",
                                weights_only=False)
        self.fnames = sorted(self.cache.keys())
        print(f"  {split}: {len(self.fnames)} samples loaded from cache")

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        entry = self.cache[fname]

        # Cached features — convert half->float for computation
        x1 = entry["x1"].float()   # [64,  128, 128]
        x2 = entry["x2"].float()   # [128,  64,  64]
        x3 = entry["x3"].float()   # [256,  32,  32]
        x4 = entry["x4"].float()   # [512,  16,  16]
        x5 = entry["x5"].float()   # [512,   8,   8]

        # Raw BERT features (before projection heads — those are trainable)
        report_feat = entry["report_feat"].float()  # [768]
        word_feat   = entry["word_feat"].float()    # [19, 768]

        # Load mask from disk
        mask = np.array(Image.open(self.mask_dir / f"{fname}.png").convert("L"))
        mask = (mask > 127).astype(np.uint8)
        if mask.shape[0] != IMG_SIZE or mask.shape[1] != IMG_SIZE:
            mask = zoom(mask, (IMG_SIZE/mask.shape[0], IMG_SIZE/mask.shape[1]),
                        order=0).astype(np.uint8)

        # Augmentation — applies same transform to spatial tensors + mask
        if self.augment:
            if random.random() > 0.5:
                k = random.randint(0, 3)
                # Rotate spatial dims of cached tensors
                x1 = torch.rot90(x1, k, [1, 2])
                x2 = torch.rot90(x2, k, [1, 2])
                x3 = torch.rot90(x3, k, [1, 2])
                x4 = torch.rot90(x4, k, [1, 2])
                x5 = torch.rot90(x5, k, [1, 2])
                mask = np.rot90(mask, k).copy()

            if random.random() > 0.5:
                # Horizontal flip
                x1 = torch.flip(x1, [2])
                x2 = torch.flip(x2, [2])
                x3 = torch.flip(x3, [2])
                x4 = torch.flip(x4, [2])
                x5 = torch.flip(x5, [2])
                mask = np.fliplr(mask).copy()

        mask_t = torch.from_numpy(mask).long()  # [128, 128]  int64

        return {
            "x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5,
            "report_feat": report_feat,
            "word_feat":   word_feat,
            "mask":        mask_t,
            "fname":       fname,
        }


# ─── Cached forward pass ──────────────────────────────────────────────────────
def forward_from_cache(model, batch):
    """
    Replaces the full model.forward() when using cached features.
    Skips BERT and UNet encoder entirely.
    Runs: global_embed, local_embed (trainable projection heads)
          -> text evidence embedding via cross-attention
          -> EAMG (vision_nonlocal, text_nonlocal, cross_att)
          -> Decoder (up4->up1) x2
          -> Prob heads + evidential outputs + EDSL loss
    """
    x1 = batch["x1"].to(DEVICE)   # [B, 64,  128, 128]
    x2 = batch["x2"].to(DEVICE)   # [B, 128,  64,  64]
    x3 = batch["x3"].to(DEVICE)   # [B, 256,  32,  32]
    x4 = batch["x4"].to(DEVICE)   # [B, 512,  16,  16]
    x5 = batch["x5"].to(DEVICE)   # [B, 512,   8,   8]
    report_feat = batch["report_feat"].to(DEVICE)   # [B, 768]
    word_feat   = batch["word_feat"].to(DEVICE)     # [B, 19, 768]
    b = x5.shape[0]

    # ── Trainable projection heads (global_embed, local_embed) ──
    report_emb = model.text_encoder.global_embed(report_feat)
    report_emb = F.normalize(report_emb, dim=-1)      # [B, 512]
    word_emb   = model.text_encoder.local_embed(word_feat)
    word_emb   = F.normalize(word_emb, dim=-1)        # [B, 19, 512]

    # ── Text evidence embedding via cross-attention (trainable CA weights) ──
    # Mirrors EviVLM.forward() lines 180-191
    patch_emb = x5.view(b, 512, -1).permute(0, 2, 1)          # [B, 64, 512]
    patch_emb = F.normalize(patch_emb, dim=-1)

    Q = patch_emb                                               # [B, 64, 512]
    K = word_emb                                                # [B, 19, 512]
    V = word_emb

    attn = torch.bmm(Q, K.permute(0, 2, 1)) / (512 ** 0.5)    # [B, 64, 19]
    attn = F.softmax(attn, dim=-1)
    patch_emb_atten = torch.bmm(attn, V)                       # [B, 64, 512]

    HW = x5.shape[2]   # 8 at 128x128
    x5_2 = patch_emb_atten.permute(0, 2, 1).view(b, 512, HW, HW)  # [B,512,8,8]

    # ── EAMG: vision_nonlocal + text_nonlocal + cross_att ──
    x5_nonlocal   = model.vision_nonlocal(x5)
    x5_2_nonlocal = model.text_nonlocal(x5_2)

    if len(x5_nonlocal.size()) != 4:
        x5_nonlocal   = x5_nonlocal.unsqueeze(1)
        x5_2_nonlocal = x5_2_nonlocal.unsqueeze(1)

    cross_aff = torch.cat((x5_nonlocal, x5_2_nonlocal), dim=1)   # [B, 2, HW, HW]
    cross_w   = model.cross_att(cross_aff)
    cross_aff = cross_aff[:, 0] * cross_w[:, 0] + cross_aff[:, 1] * cross_w[:, 1]

    refined_x5   = torch.matmul(cross_aff, patch_emb)         # [B, 64, 512]
    refined_x5_2 = torch.matmul(cross_aff, patch_emb_atten)   # [B, 64, 512]

    refined_x5_affinity   = refined_x5.permute(0,2,1).view(b, 512, HW, HW)
    refined_x5_2_affinity = refined_x5_2.permute(0,2,1).view(b, 512, HW, HW)

    x5_v = x5   * refined_x5_2_affinity
    x5_t = x5_2 * refined_x5_2_affinity

    # ── Image embeddings for EDSL ──
    img_emb_V = F.normalize(patch_emb.mean(dim=1), dim=-1)          # [B, 512]
    img_emb_L = F.normalize(patch_emb_atten.mean(dim=1), dim=-1)    # [B, 512]

    # ── Decoder — vision stream ──
    xv = model.up4(x5_v, x4)
    xv = model.up3(xv,   x3)
    xv = model.up2(xv,   x2)
    x_V = model.up1(xv,  x1)    # [B, 64, 128, 128]

    # ── Decoder — text stream ──
    xt = model.up4(x5_t, x4)
    xt = model.up3(xt,   x3)
    xt = model.up2(xt,   x2)
    x_L = model.up1(xt,  x1)    # [B, 64, 128, 128]

    # ── Probability heads ──
    alpha_V  = model.prob_alpha(x_V)   # [B, 1, 128, 128]
    beta_L   = model.prob_beta(x_L)    # [B, 1, 128, 128]
    prob_VL  = model.last_activation((alpha_V + beta_L) / 2.0)
    prob_V   = model.last_activation(alpha_V)
    prob_L   = model.last_activation(beta_L)

    # ── Evidential outputs ──
    N = b * IMG_SIZE * IMG_SIZE
    alpha_V_2 = model.prob_alpha_2(x_V).permute(0,2,3,1).reshape(N, 2)
    alpha_V_2 = nn.Softplus()(alpha_V_2)
    beta_L_2  = model.prob_alpha_2(x_L).permute(0,2,3,1).reshape(N, 2)
    beta_L_2  = nn.Softplus()(beta_L_2)
    evi_VL    = (alpha_V_2 + beta_L_2) / 2.0

    # ── Uncertainty scalars ──
    un_V = torch.sigmoid((2 / (alpha_V_2.sum(1) + 1e-6)).view(b, -1).mean(1))
    un_L = torch.sigmoid((2 / (beta_L_2.sum(1)  + 1e-6)).view(b, -1).mean(1))

    # ── EDSL loss (BVD + InfoNCE) ──
    affinity_V = un_V.unsqueeze(1) * img_emb_V
    affinity_L = un_L.unsqueeze(1) * img_emb_L
    scores     = affinity_V.mm(affinity_L.t()) / model.temperature
    labels     = torch.arange(b, device=DEVICE)   # must be on same device as scores
    loss_nce   = (F.cross_entropy(scores, labels) +
                  F.cross_entropy(scores.t(), labels)) / 2.0

    E_s = affinity_V.mean(); E_t = affinity_L.mean()
    Var_s = affinity_V.var(unbiased=False)
    Var_t = affinity_L.var(unbiased=False)
    Cov   = ((affinity_V.flatten() - E_s) * (affinity_L.flatten() - E_t)).mean()
    diff_loss = b**2 * (E_s-E_t)**2 + b**2 * (Var_s + Var_t - 2*Cov)
    loss_sim  = loss_nce + 0.1 * diff_loss

    return prob_V, prob_L, prob_VL, alpha_V_2, beta_L_2, evi_VL, loss_sim


# ─── Losses ───────────────────────────────────────────────────────────────────
class WeightedDiceBCE(nn.Module):
    def __init__(self, dice_weight=0.5, BCE_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.BCE_weight  = BCE_weight

    def _dice_loss(self, logit, truth, smooth=1e-5):
        B = len(logit)
        logit = logit.view(B, -1)
        truth = truth.view(B, -1).float()
        w = truth.detach() * (0.5-0.5) + 0.5
        p, t  = w*logit, w*truth
        inter = (p*t).sum(-1)
        union = (p*p).sum(-1) + (t*t).sum(-1)
        return (1 - (2*inter+smooth)/(union+smooth)).mean()

    def _bce_loss(self, logit, truth):
        logit = logit.view(-1)
        truth = truth.view(-1).float()
        loss  = F.binary_cross_entropy(logit, truth, reduction='none')
        pos   = (truth > 0.5).float()
        neg   = (truth < 0.5).float()
        pw    = pos.sum().item() + 1e-12
        nw    = neg.sum().item() + 1e-12
        return (0.4*pos*loss/pw + 0.6*neg*loss/nw).sum()

    def forward(self, logit, truth):
        truth = truth.float().unsqueeze(1)
        return self.BCE_weight*self._bce_loss(logit, truth) + \
               self.dice_weight*self._dice_loss(logit, truth)

    def dice_score(self, logit, truth):
        pred  = (logit > 0.5).float()
        truth = truth.float()
        inter = (pred * truth).sum()
        union = pred.sum() + truth.sum() + 1e-6
        return (2*inter/union).item()


def kl_divergence(alpha, num_classes, device):
    S      = torch.sum(alpha, dim=1, keepdim=True)
    lgC    = torch.lgamma(torch.tensor(float(num_classes), device=device))
    first  = (torch.lgamma(S) - lgC
              - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True))
    second = torch.sum((alpha-1)*(torch.digamma(alpha)-torch.digamma(S)),
                       dim=1, keepdim=True)
    return first + second   # [N, 1]


def edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    S        = torch.sum(alpha, dim=1, keepdim=True)
    log_like = torch.sum(target*(torch.digamma(alpha)-torch.digamma(S)), dim=1)
    annealing_coef = min(1.0, epoch_num / annealing_step)
    alpha_tilde    = target + (1-target)*alpha
    kl_loss        = annealing_coef * kl_divergence(alpha_tilde, num_classes, device)
    return torch.mean(-log_like + kl_loss.squeeze(1))


def get_loss(evi_V, evi_L, evi_VL, target, epoch_num, device):
    alpha_VL = evi_VL + 1
    alpha_V  = evi_V  + 1
    alpha_L  = evi_L  + 1
    L_VL = edl_digamma_loss(alpha_VL, target, epoch_num, 2, 50, device)
    L_V  = edl_digamma_loss(alpha_V,  target, epoch_num, 2, 50, device)
    L_L  = edl_digamma_loss(alpha_L,  target, epoch_num, 2, 50, device)
    return (L_VL + L_V + L_L) / 3.0


def iou_on_batch(masks, pred):
    pred  = (pred > 0.5).float()
    masks = masks.float()
    inter = (pred*masks).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3)) - inter + 1e-6
    return (inter/union).mean().item()


# ─── Collate (handle variable-length fname strings) ───────────────────────────
def collate_fn(batch):
    keys   = [k for k in batch[0].keys() if k != "fname"]
    out    = {k: torch.stack([b[k] for b in batch]) for k in keys}
    out["fname"] = [b["fname"] for b in batch]
    return out


# ─── Train / Val ──────────────────────────────────────────────────────────────
def train_one_epoch(loader, model, criterion, optimizer, epoch):
    model.train()
    loss_sum = dice_sum = iou_sum = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"  Train E{epoch+1:03d}", ncols=100, leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}] {postfix}")

    for batch in pbar:
        masks = batch["mask"].to(DEVICE)   # [B, 128, 128]

        prob_V, prob_L, prob_VL, evi_V, evi_L, evi_VL, loss_sim = \
            forward_from_cache(model, batch)

        target = F.one_hot(masks.reshape(-1), 2).float()
        loss_evi = get_loss(evi_V, evi_L, evi_VL, target, epoch+1, DEVICE)
        loss_seg = criterion(prob_VL, masks)
        loss     = (loss_seg + loss_evi*0.5) / 2.0 + loss_sim*0.2

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()

        B         = masks.size(0)
        dice      = criterion.dice_score(prob_VL.detach(), masks.unsqueeze(1).float())
        iou       = iou_on_batch(masks.unsqueeze(1).float(), prob_VL.detach())
        loss_sum += loss.item()*B; dice_sum += dice*B; iou_sum += iou*B; n += B

        pbar.set_postfix(loss=f"{loss_sum/n:.4f}",
                         dice=f"{dice_sum/n:.4f}",
                         iou=f"{iou_sum/n:.4f}")
    pbar.close()
    return loss_sum/n, dice_sum/n, iou_sum/n


@torch.no_grad()
def val_one_epoch(loader, model, criterion, epoch):
    model.eval()
    loss_sum = dice_sum = iou_sum = 0.0
    n = 0

    pbar = tqdm(loader, desc=f"  Val   E{epoch+1:03d}", ncols=100, leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}] {postfix}")

    for batch in pbar:
        masks = batch["mask"].to(DEVICE)
        prob_V, prob_L, prob_VL, evi_V, evi_L, evi_VL, _ = \
            forward_from_cache(model, batch)

        loss = criterion(prob_VL, masks)
        B    = masks.size(0)
        dice = criterion.dice_score(prob_VL, masks.unsqueeze(1).float())
        iou  = iou_on_batch(masks.unsqueeze(1).float(), prob_VL)
        loss_sum += loss.item()*B; dice_sum += dice*B; iou_sum += iou*B; n += B

        pbar.set_postfix(loss=f"{loss_sum/n:.4f}",
                         dice=f"{dice_sum/n:.4f}",
                         iou=f"{iou_sum/n:.4f}")
    pbar.close()
    return loss_sum/n, dice_sum/n, iou_sum/n


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("  EviVLM Step 2b — Fast Training from Cache")
    logger.info(f"  Device:     {DEVICE}" + (
        f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else " (CPU)"))
    logger.info(f"  Image size: {IMG_SIZE}x{IMG_SIZE}  |  Batch: {BATCH_SIZE}")
    logger.info(f"  LR: {LR}  |  Run dir: {RUN_DIR}")
    logger.info("=" * 60)

    # ── Datasets ──
    logger.info("\nLoading datasets from cache ...")
    train_ds = CachedDataset("train", augment=True)
    val_ds   = CachedDataset("val",   augment=False)
    logger.info(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    nw = 4 if torch.cuda.is_available() else 0   # parallel workers on GPU, 0 on CPU
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=nw, drop_last=True, collate_fn=collate_fn,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=nw, drop_last=False, collate_fn=collate_fn,
                              pin_memory=torch.cuda.is_available())

    # ── Model ──
    logger.info("\nLoading model (trainable: EAMG + decoder + heads) ...")
    model = patch_and_load_model()

    criterion = WeightedDiceBCE()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-5)

    best_dice = 0.0; best_epoch = 0; no_improve = 0

    logger.info("\n" + "-"*60)
    logger.info("Training started.")
    logger.info("Frozen:    BERT body + UNet encoder (never run — cached)")
    logger.info("Trainable: EAMG + decoder + prob heads + text proj heads")
    logger.info("-"*60)

    # Outer epoch progress bar
    epoch_pbar = tqdm(range(EPOCHS), desc="Epochs", ncols=60,
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")

    for epoch in epoch_pbar:
        t0 = time.time()

        tr_loss, tr_dice, tr_iou = train_one_epoch(train_loader, model, criterion, optimizer, epoch)
        vl_loss, vl_dice, vl_iou = val_one_epoch(val_loader,   model, criterion, epoch)

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        msg = (f"Epoch [{epoch+1:3d}/{EPOCHS}] "
               f"Train Loss:{tr_loss:.4f} Dice:{tr_dice:.4f} IoU:{tr_iou:.4f} | "
               f"Val Loss:{vl_loss:.4f} Dice:{vl_dice:.4f} IoU:{vl_iou:.4f} | "
               f"LR:{lr_now:.1e} Time:{elapsed:.0f}s")
        logger.info(msg)
        epoch_pbar.set_postfix(vDice=f"{vl_dice:.4f}", best=f"{best_dice:.4f}")

        if vl_dice > best_dice:
            best_dice  = vl_dice
            best_epoch = epoch + 1
            no_improve = 0
            torch.save({
                "epoch":      epoch+1,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_dice":   vl_dice,
                "val_loss":   vl_loss,
                "img_size":   IMG_SIZE,
            }, str(CKPT_DIR / "best_model.pth"))
            logger.info(f"  [BEST] Dice: {best_dice:.4f} saved -> best_model.pth")
        else:
            no_improve += 1
            logger.info(f"  No improve ({no_improve}/{EARLY_STOP}). "
                        f"Best: {best_dice:.4f} @ epoch {best_epoch}")

        if epoch % 10 == 9:
            torch.save({"epoch": epoch+1, "state_dict": model.state_dict()},
                       str(CKPT_DIR / f"checkpoint_ep{epoch+1:03d}.pth"))

        if no_improve >= EARLY_STOP:
            logger.info(f"\nEarly stop at epoch {epoch+1}.")
            break

    epoch_pbar.close()
    logger.info(f"\nTraining complete. Best Val Dice: {best_dice:.4f} @ epoch {best_epoch}")
    logger.info(f"Checkpoint: {CKPT_DIR / 'best_model.pth'}")


if __name__ == "__main__":
    main()
