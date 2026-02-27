#!/usr/bin/env python3
import os
import torch
import ResUnet_model ResUnet_model
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from scipy.interpolate import Rbf

# ==========================================
# 1. CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PPP_LEVEL = 0.5 
MODEL_PATH = "best_model_resunet.pth" 

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

def apply_qis_noise(img_tensor, ppp):
    # Physics simulator
    gain_calc = GainCalculator()
    raw_gain = gain_calc.get_gain(ppp)
    
    # QIS Physics Constants
    QE, theta_dark, sigma_read = 0.6, 1.6, 0.2
    fwc, max_val = 200, 7
    
    photon_flux = img_tensor
    theta = photon_flux * (ppp / (torch.mean(photon_flux) + 1e-6))
    lam = (QE * theta) + theta_dark
    
    # Poisson + Read Noise
    noisy = torch.poisson(lam)
    noisy = torch.clamp(noisy, 0, fwc)
    noisy = noisy + torch.normal(0, sigma_read, size=noisy.shape, device=DEVICE)
    
    # Quantization
    noisy = torch.round(noisy * raw_gain * max_val / fwc)
    noisy = torch.clamp(noisy, 0, max_val)
    
    # Normalize
    noisy_out = noisy / max_val
    norm_gain = torch.tensor([[raw_gain / 90.0]], device=DEVICE).float()
    
    return noisy_out.float(), norm_gain

# ==========================================
# 3. RUN INFERENCE (Local Version)
# ==========================================
def run_local_demo():
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: '{MODEL_PATH}' not found in the current directory.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = ResUNet_Gain().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model Loaded Successfully.")

    # 2. Get Image Path
    image_path = input("Enter the path to your image (e.g., test.jpg): ").strip()
    image_path = image_path.replace('"', '').replace("'", "")
    
    if not os.path.exists(image_path):
        print("Error: Image file not found.")
        return
    
    # 3. Process Image
    try:
        img_pil = Image.open(image_path).convert('L')
    except Exception as e:
        print(f"Error opening image: {e}")
        return
    
    # Resize to multiple of 16
    w, h = img_pil.size
    new_w, new_h = w - (w % 16), h - (h % 16)
    if new_w != w or new_h != h:
        img_pil = img_pil.resize((new_w, new_h))
    
    # Convert to Tensor
    transform = transforms.ToTensor()
    clean_tensor = transform(img_pil).to(DEVICE)
    
    # 4. Create Noisy Input (Simulation)
    print("Simulating low-light noise...")
    noisy_tensor, gain_tensor = apply_qis_noise(clean_tensor, PPP_LEVEL)
    
    noisy_input = noisy_tensor.unsqueeze(0).float()
    
    # 5. Model Prediction
    print("Running Denoising AI...")
    with torch.no_grad():
        output = model(noisy_input, gain_tensor)
        output = torch.clamp(output, 0, 1)
        
    # 6. Save Results
    noisy_np = (noisy_tensor.cpu().squeeze().numpy() * 255).astype(np.uint8)
    out_np = (output.cpu().squeeze().numpy() * 255).astype(np.uint8)
    clean_np = (clean_tensor.cpu().squeeze().numpy() * 255).astype(np.uint8)
    
    cv2.imwrite("local_noisy_input.png", noisy_np)
    cv2.imwrite("local_AI_output.png", out_np)
    cv2.imwrite("local_original.png", clean_np)
    
    print("\nDone! Saved in current folder:")
    print("1. local_noisy_input.png")
    print("2. local_AI_output.png")
    print("3. local_original.png")

if __name__ == "__main__":
    run_local_demo()
