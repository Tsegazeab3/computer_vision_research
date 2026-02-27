#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from scipy.interpolate import Rbf

# ==========================================
# 1. PHYSICS ENGINE
# ==========================================
class GainCalculator:
    def __init__(self):
        self.data = np.array([
            [0.5, 90], [1.5, 60], [2.5, 50], [3.25, 30], [6.5, 15], [9.75, 7.5],
            [13, 4.5], [20, 3.2], [26, 2.8], [36, 2.4], [45, 2.2], [54, 1.8],
            [67, 1.5], [80, 1.3], [90, 1.1], [110, 1.05], [130, 0.9], [145, 0.65],
            [155, 0.56], [160, 0.51], [200, 0.4881704]
        ])
        self.rbf = Rbf(self.data[:, 0], self.data[:, 1], function='linear')

    def get_gain(self, avg_PPP):
        return self.rbf(avg_PPP)

def torch_forward_model(avg_PPP, photon_flux, gain_calc):
    device = photon_flux.device
    QE, theta_dark, sigma_read = 0.6, 1.6, 0.2
    fwc, max_val = 200, 7
    
    raw_gain = gain_calc.get_gain(avg_PPP)
    theta = photon_flux * (avg_PPP / (torch.mean(photon_flux) + 1e-6))
    lam = (QE * theta) + theta_dark
    
    noisy = torch.poisson(lam)
    noisy = torch.clamp(noisy, 0, fwc)
    noisy = noisy + torch.normal(0, sigma_read, size=noisy.shape, device=device)
    
    noisy = torch.round(noisy * raw_gain * max_val / fwc)
    noisy = torch.clamp(noisy, 0, max_val)
    
    noisy_out = noisy / max_val
    norm_gain = torch.tensor([[raw_gain / 90.0]], device=device).float()
    
    return noisy_out.float(), norm_gain

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class QIS_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def cb(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, 1, 1),
                nn.BatchNorm2d(o),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(o, o, 3, 1, 1),
                nn.BatchNorm2d(o),
                nn.LeakyReLU(0.2, True)
            )
        self.enc1 = cb(1, 64)
        self.enc2 = cb(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = cb(128, 256)
        self.gain_fc = nn.Linear(1, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = cb(256 + 128, 128)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = cb(128 + 64, 64)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x, g):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        b = self.bottleneck(self.pool(s2))
        g_emb = torch.sigmoid(self.gain_fc(g)).view(-1, 256, 1, 1)
        b = b * g_emb
        d2 = self.dec2(torch.cat([self.up2(b), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
        return torch.clamp(x + self.final(d1), 0, 1)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def save_heatmap_with_scale(clean_np, test_np, filename, ppp_val):
    """
    Generates an error map with a Matplotlib colorbar (scale).
    """
    # 1. Calculate Raw Absolute Difference (0-255)
    diff = cv2.absdiff(clean_np, test_np)
    
    # 2. Plot using Matplotlib
    plt.figure(figsize=(6, 5))
    
    # We set vmax=50 to make small errors visible (scientific standard for denoising)
    # If you want the full range (0-255), change vmax=255.
    plt.imshow(diff, cmap='jet', vmin=0, vmax=50)
    
    cbar = plt.colorbar()
    cbar.set_label('Pixel Error Magnitude')
    
    plt.title(f"Error Map (PPP={ppp_val})")
    plt.axis('off')
    
    # 3. Save and Close
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()

def tensor_to_np(tensor):
    return (tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

# ==========================================
# 4. MAIN INFERENCE LOOP
# ==========================================
def run_multilevel_inference(image_path, model_path):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PPP_LEVELS = [0.5, 1, 3, 7]
    
    # 1. Load Model
    print(f"Loading model from {model_path}...")
    model = QIS_UNet().to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print("WARNING: Model path not found. Initializing random weights.")
    model.eval()

    # 2. Load Image
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    img_pil = Image.open(image_path).convert('L')
    w, h = img_pil.size
    img_pil = img_pil.resize((w - w%4, h - h%4))
    
    transform = transforms.ToTensor()
    clean_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    gain_calc = GainCalculator()

    print(f"Processing image: {image_path}")
    print("-" * 40)

    # 3. Loop through PPP Levels
    for ppp in PPP_LEVELS:
        print(f"Generating images for PPP = {ppp}...")
        
        # A. Simulate & Run
        noisy_tensor, norm_gain = torch_forward_model(ppp, clean_tensor, gain_calc)
        with torch.no_grad():
            output_tensor = model(noisy_tensor, norm_gain)
        
        # B. Convert to Numpy
        clean_np = tensor_to_np(clean_tensor)
        noisy_np = tensor_to_np(noisy_tensor)
        out_np = tensor_to_np(output_tensor)
        
        # C. Save 3 Images per Level
        base_name = f"ppp_{ppp}"
        
        # 1. Noisy Input
        cv2.imwrite(f"{base_name}_noisy.png", noisy_np)
        
        # 2. AI Output
        cv2.imwrite(f"{base_name}_output.png", out_np)
        
        # 3. Error Map (With Scale)
        save_heatmap_with_scale(clean_np, out_np, f"{base_name}_error.png", ppp)

    print("-" * 40)
    print("Done! Saved 12 images (3 per PPP level).")

if __name__ == "__main__":
    # --- CONFIGURE PATHS HERE ---
    input_image = "image_tested.jpg"          
    model_weights = "qis_master.pth"   
    
    run_multilevel_inference(input_image, model_weights)
