import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

def run_and_visualize():
    os.makedirs("output", exist_ok=True)
    print("-> Simulating Evidence and Loading Ground Truth Mask...")
    
    # Simulate decoded evidence [B, C, H, W]
    e_V = torch.rand(1, 2, 64, 64)
    e_T = torch.rand(1, 2, 64, 64)
    
    # Load Real Ground Truth Mask
    mask_path = "sample_mask.png"
    if not os.path.exists(mask_path):
        print(f"WARNING: '{mask_path}' not found! Falling back to dummy mask.")
        gt_mask = torch.zeros(1, 64, 64, dtype=torch.long)
        gt_mask[0, 20:40, 20:40] = 1 
    else:
        print(f"-> Reading local ground truth mask from '{mask_path}'...")
        mask_img = Image.open(mask_path).convert('L') # Convert to grayscale
        # Resize to match the spatial dimensions of our evidence maps
        mask_img = mask_img.resize((64, 64), Image.Resampling.NEAREST)
        mask_array = np.array(mask_img)
        # Threshold to 0 (background) and 1 (foreground)
        binary_mask = (mask_array > 127).astype(np.int64)
        gt_mask = torch.tensor(binary_mask).unsqueeze(0) # [1, 64, 64]

    # Equation 41: Fusion and Cross Entropy
    # fusion = sigmoid(e_V + e_T)
    fused_logits = e_V + e_T 
    fused_probs = torch.sigmoid(fused_logits)

    # Compute Cross Entropy Loss (pixel-wise for visualization)
    loss_map = F.cross_entropy(fused_logits, gt_mask, reduction='none') # [1, 64, 64]
    total_loss = loss_map.mean()

    print(f"-> Total Segmentation Loss: {total_loss.item():.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Module 06: Segmentation Loss (L_seg = {total_loss.item():.4f})", fontsize=16)

    axes[0].imshow(gt_mask[0].numpy(), cmap='gray')
    axes[0].set_title("Ground Truth Mask")

    # Show probability of foreground (class 1)
    im1 = axes[1].imshow(fused_probs[0, 1].numpy(), cmap='magma')
    axes[1].set_title("Fused Probability\n" + r"$\sigma(e^V + e^T)$")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(loss_map[0].numpy(), cmap='hot')
    axes[2].set_title("Pixel-wise CE Loss")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig("output/06_segmentation_loss.png")
    print("-> Visualization saved to output/06_segmentation_loss.png")
    plt.show()

if __name__ == "__main__":
    run_and_visualize()