# ==========================================
# 0. SETUP & INSTALLATION
# ==========================================
import os

# Install LPIPS for perceptual metrics
if os.system("pip install lpips") != 0:
    print("Installing LPIPS...")
    os.system("pip install lpips > /dev/null")

import glob
import random
import urllib.request
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import lpips
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from scipy.interpolate import Rbf
from skimage.metrics import structural_similarity as ssim

# ==========================================
# 1. DOWNLOAD DATASET (DIV2K)
# ==========================================
DATASET_PATH = "DIV2K"
TRAIN_SUBDIR = "DIV2K_train_HR"
TEST_SUBDIR = "DIV2K_valid_HR"

def setup_div2k():
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
        print("Downloading DIV2K Dataset (this may take a minute)...")
        
        urls = {
            "train": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
            "valid": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
        }
        
        for split, url in urls.items():
            filename = os.path.join(DATASET_PATH, f"{split}.zip")
            if not os.path.exists(filename):
                print(f"Downloading {split} set...")
                urllib.request.urlretrieve(url, filename)
                
            folder_name = f"DIV2K_{split}_HR"
            target_path = os.path.join(DATASET_PATH, folder_name)
            if not os.path.exists(target_path):
                print(f"Extracting {split} set...")
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall(DATASET_PATH)

setup_div2k()
print("Dataset Ready.")

# ==========================================
# 2. CONFIGURATION
# ==========================================
BATCH_SIZE = 16
EPOCHS = 100
PPP_LEVEL = 1.5
N_FRAMES = 1        # Single Shot
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Initializing TRUE ResUNet Training on {DEVICE} | N={N_FRAMES}")

# ==========================================
# 3. PHYSICS ENGINE
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

@torch.no_grad()
def torch_forward_model(avg_PPP, photon_flux, gain_calc, N=1):
    QE, theta_dark, sigma_read = 0.6, 1.6, 0.2
    fwc, Nbits = 200, 3
    max_val = 2 ** Nbits - 1
    
    raw_gain = gain_calc.get_gain(avg_PPP) * N
    gain_tensor = torch.tensor([[raw_gain]], device=photon_flux.device).float()
    
    theta = photon_flux * (avg_PPP / (torch.mean(photon_flux) + 1e-4))
    lam = ((QE * theta) + theta_dark) / N 
    
    img_sum = torch.zeros_like(theta)
    
    for _ in range(N):
        tmp = torch.poisson(lam)
        tmp = torch.clamp(tmp, 0, fwc)
        tmp = tmp + torch.normal(0, sigma_read, size=tmp.shape, device=photon_flux.device)
        tmp = torch.round(tmp * raw_gain * max_val / fwc)
        tmp = torch.clamp(tmp, 0, max_val)
        img_sum += tmp
        
    img_out = img_sum / (N * max_val)
    norm_gain = gain_tensor / 90.0
    return img_out, norm_gain


# ==========================================
# 4. DATASET
# ==========================================
class QISDataset(Dataset):
    def __init__(self, root_dir, subdir, train=True):
        self.path = os.path.join(root_dir, subdir)
        self.files = glob.glob(os.path.join(self.path, "*.png")) + glob.glob(os.path.join(self.path, "*.jpg"))
        self.transform = transforms.ToTensor()
        self.gain_calc = GainCalculator()
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('L')
        if self.train:
            w, h = img.size
            if w < 128 or h < 128: img = img.resize((128, 128))
            else:
                i = random.randint(0, h - 128); j = random.randint(0, w - 128)
                img = img.crop((j, i, j+128, i+128))
        else:
            w, h = img.size
            img = img.resize((w - w%16, h - h%16))

        clean = self.transform(img)
        noisy, gain = torch_forward_model(PPP_LEVEL, clean, self.gain_calc, N=N_FRAMES)
        return noisy, clean, gain

# ==========================================
# 5. TRUE RESIDUAL U-NET ARCHITECTURE
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.match_dims = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.match_dims(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual # The skip connection
        out = self.relu(out)
        return out

class ResUNet_Gain(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        
        self.e1 = ResidualBlock(1, 64)
        self.e2 = ResidualBlock(64, 128)
        self.e3 = ResidualBlock(128, 256)
        self.e4 = ResidualBlock(256, 512)
        
        self.b = ResidualBlock(512, 1024)
        self.gain_fc = nn.Linear(1, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.d4 = ResidualBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d3 = ResidualBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d2 = ResidualBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d1 = ResidualBlock(128, 64)
        
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x, g):
        c1 = self.e1(x); p1 = self.pool(c1)
        c2 = self.e2(p1); p2 = self.pool(c2)
        c3 = self.e3(p2); p3 = self.pool(c3)
        c4 = self.e4(p3); p4 = self.pool(c4)
        
        bn = self.b(p4)
        g_emb = torch.sigmoid(self.gain_fc(g)).view(-1, 1024, 1, 1)
        bn = bn * g_emb
        
        u4 = self.up4(bn); d4 = self.d4(torch.cat([u4, c4], dim=1))
        u3 = self.up3(d4); d3 = self.d3(torch.cat([u3, c3], dim=1))
        u2 = self.up2(d3); d2 = self.d2(torch.cat([u2, c2], dim=1))
        u1 = self.up1(d2); d1 = self.d1(torch.cat([u1, c1], dim=1))
        
        return self.out(d1)

# ==========================================
# 6. TRAINING LOOP
# ==========================================
train_ds = QISDataset(DATASET_PATH, TRAIN_SUBDIR, train=True)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = QISDataset(DATASET_PATH, TEST_SUBDIR, train=False)
test_noisy, test_clean, test_gain = test_ds[0]
test_noisy = test_noisy.unsqueeze(0).to(DEVICE)
test_gain = test_gain.to(DEVICE)

model = ResUNet_Gain().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
criterion = nn.L1Loss()


print("Starting Training... (Press Stop Button to finish early and save)")

best_loss = float('inf')

try:
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for noisy, clean, gain in train_loader:
            noisy, clean, gain = noisy.to(DEVICE), clean.to(DEVICE), gain.to(DEVICE)
            preds = model(noisy, gain)
            loss = criterion(preds, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.5f} | LR: {current_lr:.6f}")
        
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model_resunet.pth")

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")

# ==========================================
# 7. EVALUATION & METRICS
# ==========================================
print("\nRunning Final Evaluation...")

if os.path.exists("best_model_resunet.pth"):
    model.load_state_dict(torch.load("best_model_resunet.pth"))
    print("Loaded Best Model.")

model.eval()
with torch.no_grad():
    output = model(test_noisy, test_gain)
    output = torch.clamp(output, 0, 1)

# Prepare Images
noisy_np = (test_noisy.cpu().squeeze().numpy() * 255).astype(np.uint8)
out_np = (output.cpu().squeeze().numpy() * 255).astype(np.uint8)
clean_np = (test_clean.squeeze().numpy() * 255).astype(np.uint8)

# Save Images
cv2.imwrite(f"final_resunet_input.png", noisy_np)
cv2.imwrite(f"final_resunet_output.png", out_np)
cv2.imwrite("final_ground_truth.png", clean_np)

# Error Map
diff = cv2.absdiff(clean_np, out_np)
diff = cv2.multiply(diff, 3)
error_map = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
cv2.imwrite("final_resunet_error.png", error_map)

# METRICS
# 1. PSNR
mse = np.mean((clean_np.astype(float) - out_np.astype(float)) ** 2)
psnr_val = 20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else 100

# 2. SSIM
ssim_val, _ = ssim(clean_np, out_np, full=True)

# 3. LPIPS
loss_fn_alex = lpips.LPIPS(net='alex')
if torch.cuda.is_available(): loss_fn_alex.cuda()
def to_tensor(img_np):
    t = torch.from_numpy(img_np).float() / 255.0
    t = t * 2 - 1 # Normalize [-1, 1]
    return t.unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1).to(DEVICE) # Make RGB

lpips_val = loss_fn_alex(to_tensor(clean_np), to_tensor(out_np)).item()

print("-" * 30)
print(f"RESULTS (N={N_FRAMES})")
print("-" * 30)
print(f"PSNR:  {psnr_val:.2f} dB")
print(f"SSIM:  {ssim_val:.4f}")
print(f"LPIPS: {lpips_val:.4f} (Lower is better)")
print("-" * 30)
print("Files Saved: final_resunet_output.png, final_resunet_error.png, final_ground_truth.png")

