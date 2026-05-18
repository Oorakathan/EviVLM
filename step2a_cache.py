"""
STEP 2a — Cache Encoder Features (Run ONCE, never again)
=========================================================
Runs the frozen BioClinicalBERT + frozen U-Net encoder over every
train and val image ONCE and saves the outputs to disk.

During training (step2b), these cached tensors are loaded directly —
the heavy encoders never run again, saving ~2.5s BERT + 1.2s UNet per batch.

Images are resized to 128x128 here (Fix 2).
The feature map x5 becomes [512, 8, 8] instead of [512, 14, 14].
NLBlock attention drops from 196x196 to 64x64 — 9x fewer operations.

Run from EviVLM-main/ folder:
    python step2a_cache.py

Output:
    datasets/MosMedData/cache/
        train_cache.pt   (~150 MB)
        val_cache.pt     (~22 MB)

Time: ~25-35 minutes total (runs once, never again).
"""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import zoom
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

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128     # 128x128 — faster than 224x224, good for both CPU and GPU

CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ─── Minimal U-Net encoder (mirrors original EviVLM inc+down1-4) ─────────────
class ConvBatchNorm(nn.Module):
    def __init__(self, in_ch, out_ch, nb_Conv=2):
        super().__init__()
        layers = []
        for i in range(nb_Conv):
            layers += [nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
                       nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool  = nn.MaxPool2d(2)
        self.convs = ConvBatchNorm(in_ch, out_ch, nb_Conv=2)
    def forward(self, x): return self.convs(self.pool(x))

class UNetEncoder(nn.Module):
    """Matches EviVLM's inc + down1-4 exactly."""
    def __init__(self, in_channels=64):
        super().__init__()
        C = in_channels
        self.inc   = ConvBatchNorm(3,   C,   nb_Conv=2)
        self.down1 = DownBlock(C,   C*2)
        self.down2 = DownBlock(C*2, C*4)
        self.down3 = DownBlock(C*4, C*8)
        self.down4 = DownBlock(C*8, C*8)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


# ─── Text encoder loader (same patch as step2b) ───────────────────────────────
def build_text_encoder():
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
        for param in self.model.parameters():
            param.requires_grad = False

        self.idxtoword   = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.global_embed = te.GlobalEmbedding(emb_dim, hidden_dim, output_dim)
        self.local_embed  = te.LocalEmbedding(emb_dim, hidden_dim, output_dim)

    te.TextEncoder_Bert.__init__ = patched_init
    enc = te.TextEncoder_Bert()
    enc.eval()
    for p in enc.model.parameters():
        p.requires_grad = False
    return enc


# ─── Dataset reader ───────────────────────────────────────────────────────────
def read_split(split_folder, xlsx_name):
    img_dir  = DATASET_DIR / split_folder / "img"
    mask_dir = DATASET_DIR / split_folder / "labelcol"
    df       = pd.read_excel(str(DATASET_DIR / split_folder / xlsx_name))

    # Build text map, pad short sentences exactly as read_text() does
    text_map = {}
    for i in df.index.values:
        desc  = str(df.Description[i])
        count = len(desc.split())
        if count < 9:
            desc = desc + ' EOF XXX' * (9 - count)
        text_map[df.Image[i]] = desc

    samples = []
    for img_file in sorted(img_dir.glob("*.png")):
        fname = img_file.stem
        text  = text_map.get(fname, "Pulmonary infection, infected area, lung. EOF XXX EOF XXX")
        samples.append((fname, img_file, mask_dir / f"{fname}.png", text))
    return samples


# ─── Cache one split ──────────────────────────────────────────────────────────
def cache_split(split_name, split_folder, xlsx_name, vision_enc, text_enc):
    out_path = CACHE_DIR / f"{split_name}_cache.pt"
    if out_path.exists():
        print(f"  [SKIP] {split_name}_cache.pt already exists.")
        return

    samples = read_split(split_folder, xlsx_name)
    print(f"\n  Caching {split_name} ({len(samples)} samples) ...")
    cache = {}

    for fname, img_path, mask_path, text in tqdm(samples, desc=f"  {split_name}",
                                                   ncols=80, leave=True):
        # ── Load and resize image to 128x128 ──
        img = np.array(Image.open(str(img_path)).convert("RGB"))
        h, w = img.shape[:2]
        if h != IMG_SIZE or w != IMG_SIZE:
            img = zoom(img, (IMG_SIZE/h, IMG_SIZE/w, 1), order=3).astype(np.uint8)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # Normalize (ImageNet mean/std used by original)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img_t = (img_t - mean) / std
        img_t = img_t.unsqueeze(0).to(DEVICE)   # [1, 3, 128, 128]

        # ── Vision encoder (frozen, no grad) ──
        with torch.no_grad():
            x1, x2, x3, x4, x5 = vision_enc(img_t)
            # x5: [1, 512, 8, 8] at 128x128

        # ── Text encoder (frozen, no grad) ──
        with torch.no_grad():
            report_feat, word_feat, _, sents = text_enc([text], DEVICE)
            # global_embed and local_embed have trainable weights —
            # we cache BERT output BEFORE projection heads so those can still train
            # report_feat: [1, 768], word_feat: [1, 19, 768]

        cache[fname] = {
            # Always save to CPU — loaded to DEVICE during training via .to(DEVICE)
            "x1": x1.squeeze(0).cpu().half(),   # [64,  128, 128]
            "x2": x2.squeeze(0).cpu().half(),   # [128,  64,  64]
            "x3": x3.squeeze(0).cpu().half(),   # [256,  32,  32]
            "x4": x4.squeeze(0).cpu().half(),   # [512,  16,  16]
            "x5": x5.squeeze(0).cpu().half(),   # [512,   8,   8]
            # Raw BERT outputs (before global_embed/local_embed projection)
            "report_feat": report_feat.squeeze(0).cpu().half(),  # [768]
            "word_feat":   word_feat.squeeze(0).cpu().half(),    # [19, 768]
            "text": text,
        }

    torch.save(cache, str(out_path))
    mb = out_path.stat().st_size / 1e6
    print(f"  Saved {split_name}_cache.pt  ({mb:.1f} MB) -> {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  EviVLM Step 2a — Cache Encoder Features")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print("=" * 60)

    # Load U-Net encoder
    print("\nBuilding U-Net encoder ...")
    vision_enc = UNetEncoder(in_channels=64).to(DEVICE).eval()
    for p in vision_enc.parameters():
        p.requires_grad = False
    print(f"  U-Net encoder ready  [{DEVICE}]")

    # Load BioClinicalBERT
    print("\nLoading BioClinicalBERT ...")
    text_enc = build_text_encoder()
    text_enc  = text_enc.to(DEVICE)
    print(f"  BioClinicalBERT ready  [{DEVICE}]")

    # Cache both splits
    t0 = time.time()
    cache_split("train", "Train_Folder", "Train_text.xlsx", vision_enc, text_enc)
    cache_split("val",   "Val_Folder",   "Val_text.xlsx",   vision_enc, text_enc)

    total = time.time() - t0
    print(f"\nDone in {total/60:.1f} min")
    print(f"Cache saved to: {CACHE_DIR}")
    for f in sorted(CACHE_DIR.glob("*.pt")):
        print(f"  {f.name}: {f.stat().st_size/1e6:.1f} MB")
    print("\nNow run: python step2b_train.py")


if __name__ == "__main__":
    main()
