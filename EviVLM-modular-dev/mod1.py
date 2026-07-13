import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import requests
from io import BytesIO
from PIL import Image

# Import standard libraries that mimic the user's custom backbones
import timm
from transformers import AutoTokenizer, AutoModel

# ==========================================
# 1. ARCHITECTURE (Mimicking User's Files)
# ==========================================

class RealCrossAttention(nn.Module):
    """
    Equation 6 & 7 from the EviVLM paper.
    Aligns the ViT visual tokens (x_e_V) with BERT text tokens (x_t).
    """
    def __init__(self, embed_dim=768):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x_v, x_t):
        # x_v shape: [B, C, H, W] -> flatten to [B, H*W, C]
        B, C, H, W = x_v.shape
        x_v_flat = x_v.view(B, C, -1).permute(0, 2, 1) 
        
        Q = self.q_proj(x_v_flat) # [B, 196, 768]
        K = self.k_proj(x_t)      # [B, L, 768]
        V = self.v_proj(x_t)      # [B, L, 768]

        # Eq 7: Attention Weights (alpha)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.embed_dim ** 0.5)
        attention_map = F.softmax(scores, dim=-1) # [B, 196, L]

        # Eq 6: Attended Values
        x_t_attended = torch.bmm(attention_map, V) # [B, 196, 768]
        
        # Reshape back to image spatial dimensions [B, 768, 14, 14]
        x_e_T = x_t_attended.permute(0, 2, 1).view(B, C, H, W)
        
        return x_e_T, attention_map

# ==========================================
# 2. REAL DATA LOADING
# ==========================================

def get_real_data():
    img_path = "sample_image.png"
    txt_path = "sample_text.txt"
    
    print(f"-> Reading local image from '{img_path}' and text from '{txt_path}'...")
    
    if not os.path.exists(img_path):
        print(f"WARNING: '{img_path}' not found! Falling back to noise image. Please add a '{img_path}' file.")
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    else:
        img = Image.open(img_path).convert('RGB')
        
    img = img.resize((224, 224))
    
    # Normalize for ViT
    img_array = np.array(img) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_array - mean) / std
    
    image_tensor = torch.tensor(img_normalized).float().permute(2, 0, 1).unsqueeze(0) # [1, 3, 224, 224]

    # Clinical Text Prompt
    if not os.path.exists(txt_path):
        print(f"WARNING: '{txt_path}' not found! Using fallback text. Please add a '{txt_path}' file.")
        text_prompt = "Bilateral pulmonary infection, two infected areas, left lung and lower right lung."
    else:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text_prompt = f.read().strip()
            
    print(f"-> Text Prompt: '{text_prompt}'")
    
    return image_tensor, img_array, text_prompt

# ==========================================
# 3. PIPELINE EXECUTION & STATE SAVING
# ==========================================

def run_and_visualize():
    os.makedirs("../pipeline_state", exist_ok=True)
    
    image_tensor, original_img, text_prompt = get_real_data()
    
    print("-> Loading Pre-trained ViT and BERT Models...")
    
    # =========================================================
    # --- HOW TO LOAD YOUR FRIEND'S WEIGHTS ---
    # =========================================================
    
    # 1. Vision Encoder (ViT)
    # Turn off default pretraining, we will load custom weights
    vision_encoder = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0) 
    
    # TODO: When you get the weights, uncomment this block and update the path!
    # vit_weights_path = "path/to/friends_vit_weights.pth"
    # print(f"-> Loading custom ViT weights from {vit_weights_path}...")
    # vit_state = torch.load(vit_weights_path, map_location='cpu')
    # vision_encoder.load_state_dict(vit_state, strict=False)
    
    # 2. Text Encoder (BERT / BioClinicalBERT)
    # TODO: If your friend gives you a folder with config.json and pytorch_model.bin:
    # bert_folder = "path/to/friends_bert_folder/"
    # print(f"-> Loading custom BERT weights from {bert_folder}...")
    # tokenizer = AutoTokenizer.from_pretrained(bert_folder)
    # text_encoder = AutoModel.from_pretrained(bert_folder)
    
    # DEFAULT (Fallback while waiting for friend's weights):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    text_encoder = AutoModel.from_pretrained('bert-base-uncased')
    
    # 3. Cross Attention
    cross_attn = RealCrossAttention(embed_dim=768)
    
    # TODO: If your friend provides weights for the fusion/attention layers:
    # attn_weights_path = "path/to/friends_attention_weights.pth"
    # cross_attn.load_state_dict(torch.load(attn_weights_path, map_location='cpu'))

    print("-> Encoding Image and Text...")
    with torch.no_grad():
        # 1. Vision Forward (Output: [1, 196, 768] without cls token if num_classes=0 depending on timm version)
        # We use forward_features to get tokens, strip CLS token, reshape to 14x14 grid
        v_feat = vision_encoder.forward_features(image_tensor) 
        v_patches = v_feat[:, 1:, :] # Drop CLS token -> [1, 196, 768]
        x_e_V = v_patches.permute(0, 2, 1).view(1, 768, 14, 14) # Spatial Feature Map
        
        # 2. Text Forward
        tokens = tokenizer(text_prompt, return_tensors="pt")
        t_feat = text_encoder(**tokens).last_hidden_state # [1, L, 768]
        word_labels = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
        
        # 3. Cross Attention Alignment
        x_e_T, attn_map = cross_attn(x_e_V, t_feat)
        
    print(f"   Vision Evidence Shape (x_e_V): {x_e_V.shape}")
    print(f"   Text Evidence Shape   (x_e_T): {x_e_T.shape}")
    
    # Save State for Module 2
    torch.save({
        'x_e_V': x_e_V,
        'x_e_T': x_e_T,
        'tokens': word_labels,
        'original_img': original_img
    }, "../pipeline_state/module1_output.pt")
    print("-> Saved tensors to ../pipeline_state/module1_output.pt")

    # ==========================================
    # 4. VISUALIZATION
    # ==========================================
    print("-> Generating Visualizations...")
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("EviVLM Module 01: Real Vision-Text Evidence Extraction", fontsize=18, fontweight='bold')

    # Original Image
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(original_img)
    ax1.set_title("Input Chest X-Ray (224x224)")
    ax1.axis('off')

    # Text Input
    ax2 = plt.subplot(2, 3, 2)
    display_text = "\n".join([text_prompt[i:i+40] for i in range(0, len(text_prompt), 40)])
    ax2.text(0.5, 0.5, "Input Text:\n\n" + display_text, 
             fontsize=14, ha='center', va='center', bbox=dict(facecolor='lightcyan', alpha=0.5))
    ax2.set_title("Clinical Report")
    ax2.axis('off')

    # Vision Feature Map
    ax3 = plt.subplot(2, 3, 4)
    vision_features_mean = x_e_V[0].mean(dim=0).numpy()
    sns.heatmap(vision_features_mean, cmap='viridis', ax=ax3, cbar=False)
    ax3.set_title("ViT Evidence ($x_e^V$)\n(Mean Activation 14x14 Grid)")
    ax3.axis('off')

    # Find the index of the word "infection" to visualize attention
    target_word = "infection"
    try:
        word_idx = word_labels.index(target_word)
    except ValueError:
        word_idx = 1 # Fallback to first actual word

    # Cross Attention Overlay
    ax5 = plt.subplot(2, 3, 5)
    # attn_map is [1, 196, L]. Extract attention for target word, reshape to 14x14
    attn_to_word = attn_map[0, :, word_idx].view(14, 14).numpy()
    
    # Resize 14x14 to 224x224 for overlay
    import cv2
    attn_resized = cv2.resize(attn_to_word, (224, 224), interpolation=cv2.INTER_CUBIC)
    
    ax5.imshow(original_img)
    ax5.imshow(attn_resized, cmap='jet', alpha=0.5) # Overlay!
    ax5.set_title(f"Cross-Attention Overlay\n(Which image patches attend to '{word_labels[word_idx]}')")
    ax5.axis('off')

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/01_real_encoding_visualization.png", dpi=300)
    print("-> Visualization saved to output/01_real_encoding_visualization.png")
    plt.show()

if __name__ == "__main__":
    run_and_visualize()