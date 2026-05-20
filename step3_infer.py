"""
STEP 3 — Full Inference
========================
Loads the trained checkpoint and runs the complete EviVLM pipeline
on any new CT image: image + text → segmentation mask + uncertainty map.

Full inference pipeline:
  Image (PNG/NIfTI slice)
    → U-Net encoder (inc, down1-4)    [frozen during training]
    → EAMG (vision_nonlocal + cross_att)  [trained]
    → U-Net decoder (up1-4)           [trained]
    → prob_alpha / prob_beta heads    [trained]
    → sigmoid → prob_VL               [final segmentation mask]

  Text sentence
    → BioClinicalBERT tokenizer+model [frozen during training]
    → global_embed / local_embed      [trained proj heads]
    → cross-attention with image      [trained, inside EAMG]

Usage:
    # Segment a single PNG image with a text description
    python step3_infer.py \
        --image  datasets/MosMedData/Test_Folder/img/study_0255_slice045.png \
        --text   "Bilateral pulmonary infection, two infected areas, lower left lung and upper right lung." \
        --ckpt   runs/<timestamp>/checkpoints/best_model.pth \
        --out    output_mask.png

    # Segment all slices in the Test_Folder and save results
    python step3_infer.py \
        --test_dir datasets/MosMedData/Test_Folder \
        --ckpt     runs/<timestamp>/checkpoints/best_model.pth \
        --out_dir  inference_results/
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from scipy.ndimage import zoom
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CODE_DIR = BASE_DIR / "code"
BERT_PATH= str(CODE_DIR / "backbones" / "bert_model" / "Bio_ClinicalBERT")
BERT_CFG = str(CODE_DIR / "backbones" / "bert_model" / "bert_config.json")

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "backbones" / "bert_model"))
sys.path.insert(0, str(CODE_DIR / "nets"))

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128   # default; overridden by checkpoint value if saved


# ─── Patch + load model (same as training) ────────────────────────────────────
def load_model(ckpt_path):
    import backbones.bert_model.TextEncoder as te

    original_init = te.TextEncoder_Bert.__init__

    def patched_init(self, tokenizer=None, emb_dim=768, output_dim=512,
                     hidden_dim=2048, freeze_bert=False):
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

        self.idxtoword   = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.global_embed = te.GlobalEmbedding(emb_dim, hidden_dim, output_dim)
        self.local_embed  = te.LocalEmbedding(emb_dim, hidden_dim, output_dim)

    te.TextEncoder_Bert.__init__ = patched_init

    from nets.EviVLM import EviVLM

    original_evivlm_init = EviVLM.__init__

    def patched_evivlm_init(self, n_channels=3, n_classes=1):
        original_evivlm_init(self, n_channels, n_classes)
        self.device = DEVICE

    EviVLM.__init__ = patched_evivlm_init

    model = EviVLM(n_channels=3, n_classes=1).to(DEVICE)

    # Load checkpoint
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()

    # Read IMG_SIZE that was used during training
    global IMG_SIZE
    IMG_SIZE = ckpt.get('img_size', 128)

    ep   = ckpt.get('epoch', '?')
    dice = ckpt.get('val_dice', '?')
    dev_name = f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
    print(f"  Device:     {dev_name}")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}  (from checkpoint)")
    print(f"  Checkpoint: epoch {ep}, val Dice {dice}")
    return model


# ─── Image preprocessing ──────────────────────────────────────────────────────
def preprocess_image(img_path):
    """Load PNG -> [1, 3, IMG_SIZE, IMG_SIZE] tensor on DEVICE."""
    img = np.array(Image.open(str(img_path)).convert("RGB"))
    h, w = img.shape[:2]
    if h != IMG_SIZE or w != IMG_SIZE:
        img = zoom(img, (IMG_SIZE/h, IMG_SIZE/w, 1), order=3)
    img = img.astype(np.uint8)
    t = TF.to_tensor(Image.fromarray(img)).unsqueeze(0)   # [1, 3, H, W]
    return t.to(DEVICE)


def postprocess_mask(prob_VL, threshold=0.5):
    """prob_VL: [1,1,H,W] -> binary numpy [H,W]."""
    return (prob_VL.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255


def uncertainty_map(evi_V, evi_L, B=1):
    """Pixel-level uncertainty. Returns [IMG_SIZE, IMG_SIZE] numpy float."""
    evi_V = evi_V.cpu(); evi_L = evi_L.cpu()
    alpha_V = evi_V + 1.0;  alpha_L = evi_L + 1.0
    S_V = alpha_V.sum(dim=1); S_L = alpha_L.sum(dim=1)
    un_V = (2.0 / S_V).view(B, IMG_SIZE, IMG_SIZE).squeeze(0)
    un_L = (2.0 / S_L).view(B, IMG_SIZE, IMG_SIZE).squeeze(0)
    return ((un_V + un_L) / 2.0).numpy()


def save_results(mask, uncertainty, out_path):
    """Save segmentation mask and uncertainty map side by side."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save binary mask
    Image.fromarray(mask).save(str(out_path))

    # Save uncertainty map (normalized to 0-255 for visualization)
    unc_norm = (uncertainty * 255).clip(0, 255).astype(np.uint8)
    unc_path = out_path.parent / (out_path.stem + "_uncertainty.png")
    Image.fromarray(unc_norm).save(str(unc_path))
    print(f"  Mask saved:        {out_path}")
    print(f"  Uncertainty saved: {unc_path}")


# ─── Single image inference ───────────────────────────────────────────────────
def infer_single(model, img_path, text):
    """Run inference on one image+text pair."""
    image = preprocess_image(img_path).to(DEVICE)   # [1, 3, 224, 224]

    with torch.no_grad():
        prob_V, prob_L, prob_VL, evi_V, evi_L, evi_VL, _ = model(image, [text])

    mask = postprocess_mask(prob_VL.cpu())
    unc  = uncertainty_map(evi_V.cpu(), evi_L.cpu(), B=1)
    dice_raw = prob_VL.squeeze().cpu().numpy()   # raw probability map

    return mask, unc, dice_raw


def read_text_excel(xlsx_path):
    """Read text annotations from Excel → dict {image_stem: description}."""
    df = pd.read_excel(str(xlsx_path))
    return {row['Image']: row['Description'] for _, row in df.iterrows()}


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EviVLM Full Inference")
    parser.add_argument("--ckpt",     required=True, help="Path to best_model.pth")
    parser.add_argument("--image",    default=None,  help="Single PNG image path")
    parser.add_argument("--text",     default=None,  help="Text description for single image")
    parser.add_argument("--test_dir", default=None,  help="Test folder path (batch mode)")
    parser.add_argument("--out",      default="output_mask.png", help="Output mask path (single)")
    parser.add_argument("--out_dir",  default="inference_results", help="Output dir (batch)")
    parser.add_argument("--threshold",type=float, default=0.5, help="Segmentation threshold")
    args = parser.parse_args()

    print("Loading model ...")
    model = load_model(args.ckpt)

    if args.image and args.text:
        # ── Single image mode ──
        print(f"Inferring: {args.image}")
        print(f"Text:      {args.text}")
        mask, unc, prob = infer_single(model, args.image, args.text)
        save_results(mask, unc, args.out)
        print(f"  Mean uncertainty: {unc.mean():.4f}")
        print(f"  Lesion coverage:  {(mask>0).mean()*100:.1f}%")

    elif args.test_dir:
        # ── Batch mode: all test images ──
        test_dir = Path(args.test_dir)
        xlsx     = test_dir / "Test_text.xlsx"
        img_dir  = test_dir / "img"
        out_dir  = Path(args.out_dir)

        all_text = read_text_excel(xlsx)
        img_files = sorted(img_dir.glob("*.png"))
        print(f"Batch inference on {len(img_files)} images ...")

        dice_scores = []
        mask_dir_gt = test_dir / "labelcol"  # ground truth masks

        for img_path in img_files:
            fname = img_path.stem
            text  = all_text.get(fname, "Pulmonary infection, infected area, lung.")
            mask, unc, prob = infer_single(model, img_path, text)
            save_results(mask, unc, out_dir / f"{fname}_pred.png")

            # Compute Dice if GT mask exists
            gt_path = mask_dir_gt / f"{fname}.png"
            if gt_path.exists():
                gt   = np.array(Image.open(str(gt_path)).convert("L"))
                gt_b = (gt > 127).astype(np.float32)
                pr_b = (mask > 127).astype(np.float32)
                inter = (gt_b * pr_b).sum()
                union = gt_b.sum() + pr_b.sum() + 1e-6
                dice  = 2 * inter / union
                dice_scores.append(dice)

        if dice_scores:
            print(f"\nTest Dice: {np.mean(dice_scores)*100:.2f}%  "
                  f"(over {len(dice_scores)} samples)")
        print(f"\n✅ Inference complete. Results saved to: {out_dir}")

    else:
        parser.print_help()
        print("\nERROR: Provide either --image + --text OR --test_dir")


if __name__ == "__main__":
    main()
