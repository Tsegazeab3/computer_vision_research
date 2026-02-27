import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import relu
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.interpolate import Rbf
import os
import random
import requests
import tarfile
def download_bsds():
    url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
    target_path = "BSDS300-images.tgz"
    
    if os.path.exists('BSDS300'):
        print("Dataset already exists.")
        return

    print("Downloading BSDS300 Dataset (22MB)...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(target_path, 'wb') as f:
            f.write(response.raw.read())
            
        print("Extracting...")
        with tarfile.open(target_path, 'r:gz') as tar:
            tar.extractall()
        
        os.remove(target_path)
        print("Done.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")
# --- 1. PHYSICS SIMULATION ---
def get_gain(avg_PPP, N=1):
    data = np.array([
        [0.5, 90], [1.5, 60], [2.5, 50], [3.25, 30], [6.5, 15], [9.75, 7.5],
        [13, 4.5], [20, 3.2], [26, 2.8], [36, 2.4], [45, 2.2], [54, 1.8],
        [67, 1.5], [80, 1.3], [90, 1.1], [110, 1.05], [130, 0.9], [145, 0.65],
        [155, 0.56], [160, 0.51], [200, 0.4881704]
    ])
    rbf = Rbf(data[:, 0], data[:, 1], function='linear')
    return rbf(avg_PPP) * N

@torch.no_grad()
def torch_forward_model(avg_PPP, photon_flux, QE=0.6, theta_dark=1.6, sigma_read=0.2, N=1, Nbits=3, fwc=200, normalize=True):
    min_val, max_val = 0, 2 ** Nbits - 1
    gain = get_gain(avg_PPP)

    if not torch.is_tensor(avg_PPP): avg_PPP = torch.tensor(avg_PPP, dtype=torch.float32)
    if not torch.is_tensor(photon_flux): photon_flux = torch.tensor(photon_flux, dtype=torch.float32)
    
    QE = torch.tensor(QE, dtype=torch.float32)
    theta_dark = torch.tensor(theta_dark, dtype=torch.float32)
    sigma_read = torch.tensor(sigma_read, dtype=torch.float32)
    gain = torch.tensor(gain, dtype=torch.float32)
    fwc = torch.tensor(fwc, dtype=torch.float32)

    theta = photon_flux * (avg_PPP / (torch.mean(photon_flux) + 0.0001))
    lam = ((QE * theta) + theta_dark) / N
    img_out = torch.zeros_like(theta)

    for i in range(N):
        tmp = torch.poisson(lam)
        tmp = torch.clamp(tmp, 0, fwc.item())
        tmp = tmp + torch.normal(mean=0, std=sigma_read, size=img_out.shape, device=theta.device)
        tmp = torch.round(tmp * gain * max_val / fwc)
        tmp = torch.clamp(tmp, min_val, max_val)
        img_out = img_out + tmp

    img_out = img_out / N
    if normalize: img_out = img_out / max_val
    return img_out, gain

# --- 2. DATASET ---
class QIS_BSDS_Dataset(Dataset):
    # CHANGED: patch_size=128
    def __init__(self, root_dir='./BSDS300/images/train', patch_size=128, patches_per_img=10):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.patch_size = patch_size
        self.patches_per_img = patches_per_img
        self.data = self._preload_patches()

    def _preload_patches(self):
        patches = []
        for fp in self.files:
            img = Image.open(fp).convert('L')
            w, h = img.size
            # Only crop if image is large enough
            if w >= self.patch_size and h >= self.patch_size:
                for _ in range(self.patches_per_img):
                    x = random.randint(0, w - self.patch_size)
                    y = random.randint(0, h - self.patch_size)
                    patch = img.crop((x, y, x+self.patch_size, y+self.patch_size))
                    patches.append(transforms.ToTensor()(patch))
        return patches

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        img_clean = self.data[idx]
        img_noisy, _ = torch_forward_model(avg_PPP=7, photon_flux=img_clean)
        return img_noisy, img_clean

# --- 3. MODEL (128 Filters -> 8x8 Bottleneck) ---
class UNet_Noise_128(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # --- ENCODER ---
        # 128x128 -> 128 filters
        self.e11 = nn.Conv2d(1, 128, 3, padding=1)
        self.e12 = nn.Conv2d(128, 128, 3, padding=1)

        # 64x64 -> 256 filters
        self.e21 = nn.Conv2d(128, 256, 3, padding=1)
        self.e22 = nn.Conv2d(256, 256, 3, padding=1)

        # 32x32 -> 512 filters
        self.e31 = nn.Conv2d(256, 512, 3, padding=1)
        self.e32 = nn.Conv2d(512, 512, 3, padding=1)

        # 16x16 -> 512 filters
        self.e41 = nn.Conv2d(512, 512, 3, padding=1)
        self.e42 = nn.Conv2d(512, 512, 3, padding=1)

        # --- BOTTLENECK (8x8) ---
        self.b1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.b2 = nn.Conv2d(1024, 1024, 3, padding=1)

        # --- DECODER ---
        # Up 4: 8x8 -> 16x16
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.d41 = nn.Conv2d(1024, 512, 3, padding=1) # 512+512 input
        self.d42 = nn.Conv2d(512, 512, 3, padding=1)

        # Up 3: 16x16 -> 32x32
        self.up3 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.d31 = nn.Conv2d(1024, 512, 3, padding=1)
        self.d32 = nn.Conv2d(512, 256, 3, padding=1)

        # Up 2: 32x32 -> 64x64
        self.up2 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.d21 = nn.Conv2d(512, 256, 3, padding=1)
        self.d22 = nn.Conv2d(256, 128, 3, padding=1)

        # Up 1: 64x64 -> 128x128
        self.up1 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.d11 = nn.Conv2d(256, 128, 3, padding=1)
        self.d12 = nn.Conv2d(128, 128, 3, padding=1)

        self.out = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        # E1
        e1 = relu(self.e12(relu(self.e11(x))))
        p1 = self.pool(e1)
        # E2
        e2 = relu(self.e22(relu(self.e21(p1))))
        p2 = self.pool(e2)
        # E3
        e3 = relu(self.e32(relu(self.e31(p2))))
        p3 = self.pool(e3)
        # E4
        e4 = relu(self.e42(relu(self.e41(p3))))
        p4 = self.pool(e4)

        # Bottleneck (8x8)
        b = relu(self.b2(relu(self.b1(p4))))

        # D4
        u4 = self.up4(b)
        d4 = relu(self.d42(relu(self.d41(torch.cat([u4, e4], dim=1)))))
        # D3
        u3 = self.up3(d4)
        d3 = relu(self.d32(relu(self.d31(torch.cat([u3, e3], dim=1)))))
        # D2
        u2 = self.up2(d3)
        d2 = relu(self.d22(relu(self.d21(torch.cat([u2, e2], dim=1)))))
        # D1
        u1 = self.up1(d2)
        d1 = relu(self.d12(relu(self.d11(torch.cat([u1, e1], dim=1)))))

        return self.out(d1)

# --- 4. EXECUTOR ---
def train_bsds():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CHANGED: Ensure patch size is passed here
    dataset = QIS_BSDS_Dataset(root_dir='./BSDS300/images/train', patch_size=128) 
    
    # Reduced batch size to 16 because of heavy model/memory usage
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # CHANGED: Using the new class
    model = UNet_Noise_128().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss() 

    print(f"Starting Training on {device}...")
    for epoch in range(50):
        model.train()
        total_loss = 0
        for n, c in loader:
            n, c = n.to(device), c.to(device)
            optimizer.zero_grad()
            loss = criterion(model(n), c)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.5f}")
    
    torch.save(model.state_dict(), "qis_unet_128x128.pth")
    return model

if __name__ == "__main__":
    if os.path.exists('./BSDS300/images/train'):
        train_bsds()
    else:
        download_bsds()
        train_bsds()


