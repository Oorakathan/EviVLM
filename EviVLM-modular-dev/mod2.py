import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class NonLocalBlock(nn.Module):
    """Generates the Modality-Specific Affinity Maps (Eq. 8)"""
    def __init__(self, channels=64):
        super().__init__()
        self.q_conv = nn.Conv2d(channels, channels//8, 1)
        self.k_conv = nn.Conv2d(channels, channels//8, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        # Extract query and key, flatten spatial dimensions
        Q = self.q_conv(x).view(B, -1, H*W).permute(0, 2, 1) # [B, HW, C/8]
        K = self.k_conv(x).view(B, -1, H*W)                  # [B, C/8, HW]
        
        # Calculate affinity matrix (spatial position similarities)
        affinity = torch.bmm(Q, K) # [B, HW, HW]
        affinity = F.softmax(affinity, dim=-1)
        return affinity

def run_and_visualize():
    os.makedirs("output", exist_ok=True)
    print("-> Generating Dummy Embeddings (from Module 1)...")
    
    # Simulating outputs from Module 1: [B, C, H, W]
    x_e_V = torch.rand(1, 64, 16, 16) # Vision Evidence
    x_e_T = torch.rand(1, 64, 16, 16) # Aligned Text Evidence
    
    # Create distinct patterns for visualization
    x_e_V[0, :, 4:12, 4:12] += 2.0  # Center focus for vision
    x_e_T[0, :, 8:16, 8:16] += 2.0  # Bottom-right focus for text

    print("-> Running Non-Local Blocks...")
    nl_vision = NonLocalBlock(channels=64)
    nl_text = NonLocalBlock(channels=64)

    with torch.no_grad():
        A_V = nl_vision(x_e_V) # [1, 256, 256]
        A_T = nl_text(x_e_T)   # [1, 256, 256]

    print("-> Generating Visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Module 02: EAMG Modality-Specific Affinity Maps", fontsize=16, fontweight='bold')

    sns.heatmap(A_V[0].numpy(), cmap='mako', ax=axes[0], cbar=True)
    axes[0].set_title("Vision Affinity Matrix $A_{evi}^V$\n(256 x 256 Pixels)")
    axes[0].set_xlabel("Key Spatial Position (Flattened)")
    axes[0].set_ylabel("Query Spatial Position (Flattened)")

    sns.heatmap(A_T[0].numpy(), cmap='flare', ax=axes[1], cbar=True)
    axes[1].set_title("Text Affinity Matrix $A_{evi}^T$\n(256 x 256 Pixels)")
    axes[1].set_xlabel("Key Spatial Position (Flattened)")
    axes[1].set_ylabel("Query Spatial Position (Flattened)")

    plt.tight_layout()
    plt.savefig("output/02_local_affinity.png", dpi=300)
    print("-> Visualization saved to output/02_local_affinity.png")
    plt.show()

if __name__ == "__main__":
    run_and_visualize()