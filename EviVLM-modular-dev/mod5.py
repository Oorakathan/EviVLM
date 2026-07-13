import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import os

class EvidenceDecoder(nn.Module):
    """Simulates U-Net Decoder path to yield raw evidence e >= 0"""
    def __init__(self, in_channels=64, num_classes=2):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, 32, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)
        
        # Crucial for Evidential Learning: Evidence MUST be non-negative
        self.activation = nn.Softplus() 

    def forward(self, x):
        x = torch.relu(self.upconv(x))
        x = torch.relu(self.conv(x))
        x = torch.relu(self.upconv2(x))
        x = self.final_conv(x)
        return self.activation(x) # Output shape: [B, C, H, W], completely positive

def run_and_visualize():
    os.makedirs("output", exist_ok=True)
    # Refined embeddings from Module 3 [1, 64, 16, 16]
    x_e_a_V = torch.randn(1, 64, 16, 16)
    x_e_a_T = torch.randn(1, 64, 16, 16)

    # Note: In EviVLM paper, parameters of Vision & Text decoders are SHARED
    shared_decoder = EvidenceDecoder(num_classes=2) 
    
    with torch.no_grad():
        e_V = shared_decoder(x_e_a_V) # [1, 2, 64, 64]
        e_T = shared_decoder(x_e_a_T) # [1, 2, 64, 64]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    # Changed \ge to \geq to fix the Matplotlib ParseFatalException
    fig.suptitle(r"Module 05: Decoding to Raw Evidence ($e \geq 0$)", fontsize=16, fontweight='bold')

    sns.heatmap(e_V[0, 0].numpy(), cmap='Reds', ax=axes[0, 0])
    axes[0, 0].set_title("Vision Evidence: Background Class ($e^V_0$)")
    
    sns.heatmap(e_V[0, 1].numpy(), cmap='Reds', ax=axes[0, 1])
    axes[0, 1].set_title("Vision Evidence: Foreground Class ($e^V_1$)")

    sns.heatmap(e_T[0, 0].numpy(), cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title("Text Evidence: Background Class ($e^T_0$)")
    
    sns.heatmap(e_T[0, 1].numpy(), cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_title("Text Evidence: Foreground Class ($e^T_1$)")

    plt.tight_layout()
    plt.savefig("output/05_decoded_evidence.png")
    print("-> Visualization saved to output/05_decoded_evidence.png")
    plt.show()

if __name__ == "__main__":
    run_and_visualize()