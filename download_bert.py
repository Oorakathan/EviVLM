"""
Download Bio_ClinicalBERT PyTorch files from HuggingFace.

The repo has exactly these files needed for PyTorch:
  - pytorch_model.bin   (436 MB)  ← actual model weights
  - config.json         (385 B)   ← architecture config
  - vocab.txt           (213 KB)  ← tokenizer vocabulary

NO tokenizer_config.json exists in this repo — ignore that file.

Usage:
    python download_bert.py
"""

import os
from pathlib import Path

save_dir = Path(__file__).parent / "code" / "backbones" / "bert_model" / "Bio_ClinicalBERT"
save_dir.mkdir(parents=True, exist_ok=True)

# Only these 3 files exist in the HuggingFace repo
files_needed = [
    "pytorch_model.bin",   # 436 MB — the weights
    "config.json",         # 385 B  — architecture
    "vocab.txt",           # 213 KB — tokenizer vocab
]

from huggingface_hub import hf_hub_download

for fname in files_needed:
    out = save_dir / fname
    if out.exists():
        print(f"  Already exists, skipping: {fname}")
        continue
    print(f"Downloading {fname} ...")
    hf_hub_download(
        repo_id="emilyalsentzer/Bio_ClinicalBERT",
        filename=fname,
        local_dir=str(save_dir),
    )
    print(f"  ✓ Done — {out.stat().st_size / 1e6:.1f} MB")

print(f"\n✅ All files saved to:\n   {save_dir}")
print("\nYour Bio_ClinicalBERT folder should now contain:")
for f in sorted(save_dir.glob("*")):
    print(f"   {f.name}  ({f.stat().st_size/1e6:.2f} MB)")
