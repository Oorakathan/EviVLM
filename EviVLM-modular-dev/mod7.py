import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

def subjective_logic(e, num_classes=2):
    """Transforms Evidence into Dirichlet params, Belief, and Uncertainty (Eq 1-5)"""
    alpha = e + 1.0                           # Eq 2
    S = torch.sum(alpha, dim=1, keepdim=True) # Eq 3 (Total Evidence + classes)
    
    belief = e / S                            # Belief mass
    uncertainty = num_classes / S             # Eq 5: Uncertainty mass
    return alpha, belief, uncertainty

def dempster_shafer_aggregation(b_V, u_V, b_T, u_T, num_classes=2):
    """Aggregates opinions from Vision and Text using DS rules (Eq 24)"""
    # Calculate conflict (kappa) -> sum(b_i^V * b_j^T) for i != j
    conflict = (b_V[:, 0:1] * b_T[:, 1:2]) + (b_V[:, 1:2] * b_T[:, 0:1])
    
    M = 1.0 - conflict # Normalization factor
    
    # Eq 24 Aggregation
    b_agg_0 = (b_V[:, 0:1]*b_T[:, 0:1] + b_V[:, 0:1]*u_T + b_T[:, 0:1]*u_V) / M
    b_agg_1 = (b_V[:, 1:2]*b_T[:, 1:2] + b_V[:, 1:2]*u_T + b_T[:, 1:2]*u_V) / M
    b_agg = torch.cat([b_agg_0, b_agg_1], dim=1)
    
    u_agg = (u_V * u_T) / M
    
    return b_agg, u_agg

def run_and_visualize():
    os.makedirs("output", exist_ok=True)
    
    # 1. Simulating conflicting evidence maps
    e_V = torch.zeros(1, 2, 64, 64)
    e_T = torch.zeros(1, 2, 64, 64)
    
    # Image is very sure about center left
    e_V[0, 1, 20:40, 10:30] = 5.0 
    # Text is very sure about center right
    e_T[0, 1, 20:40, 30:50] = 5.0
    
    # 2. Subjective Logic mappings
    _, b_V, u_V = subjective_logic(e_V)
    _, b_T, u_T = subjective_logic(e_T)
    
    # 3. Opinion Aggregation
    b_agg, u_agg = dempster_shafer_aggregation(b_V, u_V, b_T, u_T)

    # 4. Visualization
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle("Module 07: Subjective Logic & Evidential Fusion", fontsize=16, fontweight='bold')

    # Vision
    sns.heatmap(b_V[0, 1].numpy(), vmin=0, vmax=1, cmap='Purples', ax=axes[0, 0])
    axes[0, 0].set_title("Vision Belief (Foreground)")
    sns.heatmap(u_V[0, 0].numpy(), vmin=0, vmax=1, cmap='YlOrRd', ax=axes[0, 1])
    axes[0, 1].set_title("Vision Uncertainty")

    # Text
    sns.heatmap(b_T[0, 1].numpy(), vmin=0, vmax=1, cmap='Greens', ax=axes[1, 0])
    axes[1, 0].set_title("Text Belief (Foreground)")
    sns.heatmap(u_T[0, 0].numpy(), vmin=0, vmax=1, cmap='YlOrRd', ax=axes[1, 1])
    axes[1, 1].set_title("Text Uncertainty")

    # Aggregated
    sns.heatmap(b_agg[0, 1].numpy(), vmin=0, vmax=1, cmap='Blues', ax=axes[2, 0])
    axes[2, 0].set_title("Aggregated Belief (Foreground)")
    sns.heatmap(u_agg[0, 0].numpy(), vmin=0, vmax=1, cmap='YlOrRd', ax=axes[2, 1])
    axes[2, 1].set_title("Aggregated Uncertainty\n(Notice reduction where they agree/combine!)")

    plt.tight_layout()
    plt.savefig("output/07_evidential_fusion.png")
    print("-> Visualization saved to output/07_evidential_fusion.png")
    plt.show()

if __name__ == "__main__":
    run_and_visualize()