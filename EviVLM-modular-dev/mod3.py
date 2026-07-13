import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os

class GlobalAffinityAndRefinement(nn.Module):
    """Fuses local affinities (Eq 9) and refines embeddings (Eq 10, 11)"""
    def __init__(self):
        super().__init__()
        # Self attention for weighting affinity maps
        self.conv = nn.Conv2d(2, 2, kernel_size=1) 

    def forward(self, A_V, A_T, x_e_V, x_e_T):
        B, N, N = A_V.shape # N = H*W
        # Stack affinities: [B, 2, HW, HW]
        A_concat = torch.stack([A_V, A_T], dim=1) 
        
        # Calculate weights w^V and w^T
        weights = F.softmax(self.conv(A_concat), dim=1)
        w_V, w_T = weights[:, 0, :, :], weights[:, 1, :, :]
        
        # Eq 9: Global Cross-Modal Affinity Map
        A_global = w_V * A_V + w_T * A_T 

        # Eq 10 & 11: Affine Refinement -> affine(A, x) * x
        B, C, H, W = x_e_V.shape
        x_v_flat = x_e_V.view(B, C, N).permute(0, 2, 1) # [B, HW, C]
        x_t_flat = x_e_T.view(B, C, N).permute(0, 2, 1)
        
        # Matrix multiply spatial affinity with features
        affine_V = torch.bmm(A_global, x_v_flat).permute(0, 2, 1).view(B, C, H, W)
        affine_T = torch.bmm(A_global, x_t_flat).permute(0, 2, 1).view(B, C, H, W)
        
        # Hadamard product
        x_e_a_V = affine_V * x_e_V
        x_e_a_T = affine_T * x_e_T
        
        return A_global, x_e_a_V, x_e_a_T

def run_and_visualize():
    os.makedirs("output", exist_ok=True)
    # Mock inputs
    x_e_V = torch.rand(1, 64, 16, 16)
    x_e_T = torch.rand(1, 64, 16, 16)
    A_V = F.softmax(torch.randn(1, 256, 256), dim=-1)
    A_T = F.softmax(torch.randn(1, 256, 256), dim=-1)

    refiner = GlobalAffinityAndRefinement()
    with torch.no_grad():
        A_global, x_e_a_V, x_e_a_T = refiner(A_V, A_T, x_e_V, x_e_T)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Module 03: Global Affinity & Feature Refinement", fontsize=16, fontweight='bold')

    sns.heatmap(A_global[0].numpy(), cmap='plasma', ax=axes[0])
    axes[0].set_title("Global Cross-Modal Affinity ($A_{evi}$)")

    sns.heatmap(x_e_V[0].mean(0).numpy(), cmap='viridis', ax=axes[1])
    axes[1].set_title("Original Vision Embedding (Mean)")

    sns.heatmap(x_e_a_V[0].mean(0).numpy(), cmap='viridis', ax=axes[2])
    axes[2].set_title("Refined Vision Embedding (Mean)\n(After Affine Operator)")

    plt.tight_layout()
    plt.savefig("output/03_global_refinement.png")
    print("-> Visualization saved to output/03_global_refinement.png")
    plt.show()

if __name__ == "__main__":
    run_and_visualize()