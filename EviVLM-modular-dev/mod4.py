import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_edsl(x_v, x_t):
    """Calculates S_V2T, S_T2V, and the BVD differential matrix (Eq 17-20)"""
    B = x_v.shape[0]
    
    # Global average pooling to get 1D vector per batch item
    v_pool = F.adaptive_avg_pool2d(x_v, 1).view(B, -1)
    t_pool = F.adaptive_avg_pool2d(x_t, 1).view(B, -1)
    
    # Normalize for cosine similarity
    v_norm = F.normalize(v_pool, dim=-1)
    t_norm = F.normalize(t_pool, dim=-1)
    
    # Bidirectional Similarity Matrices (B x B)
    S_V2T = torch.mm(v_norm, t_norm.t()) # [B, B]
    S_T2V = torch.mm(t_norm, v_norm.t()) # [B, B]
    
    # Differential Matrix
    S_diff = S_V2T - S_T2V
    
    # Bias-Variance Decomposition (BVD) parts for Loss
    mean_s = S_V2T.mean()
    mean_t = S_T2V.mean()
    bias = (mean_s - mean_t) ** 2
    variance = S_diff.var()
    
    loss_diff = (B**2) * bias + (B**2) * variance
    
    return S_V2T, S_T2V, S_diff, bias, variance, loss_diff

def run_and_visualize():
    os.makedirs("output", exist_ok=True)
    print("-> Generating Batch of Size 8 for Contrastive Similarities...")
    # Simulate a batch of 8 samples
    x_v = torch.randn(8, 64, 16, 16)
    x_t = x_v + torch.randn(8, 64, 16, 16) * 0.5 # Add noise to create correlation

    S_V2T, S_T2V, S_diff, bias, var, loss = calculate_edsl(x_v, x_t)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Module 04: EDSL | Bias: {bias:.4f} | Var: {var:.4f} | Diff Loss: {loss:.4f}", fontsize=16)

    sns.heatmap(S_V2T.numpy(), annot=True, fmt=".2f", cmap='Blues', ax=axes[0])
    axes[0].set_title("Vision-to-Text Similarity ($S_{V2T}$)")

    sns.heatmap(S_T2V.numpy(), annot=True, fmt=".2f", cmap='Greens', ax=axes[1])
    axes[1].set_title("Text-to-Vision Similarity ($S_{T2V}$)")

    sns.heatmap(S_diff.numpy(), annot=True, fmt=".2f", cmap='coolwarm', ax=axes[2], center=0)
    axes[2].set_title("Differential Matrix ($S_{diff}$)")

    plt.tight_layout()
    plt.savefig("output/04_edsl_matrices.png")
    print("-> Visualization saved to output/04_edsl_matrices.png")
    plt.show()

if __name__ == "__main__":
    run_and_visualize()