import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import relu
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from scipy.interpolate import Rbf
import os

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

# --- 2. DATASET (Log-Uniform, 28x28) ---
class QIS_MNIST_Base(Dataset):
    def __init__(self, root='./data', train=True, download=True):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.mnist_data = datasets.MNIST(root=root, train=train, download=download, transform=self.transform)

    def __len__(self): return len(self.mnist_data)

    def __getitem__(self, idx):
        img_clean, _ = self.mnist_data[idx]
        
        # Log-Uniform 1.5 to 10.0
        log_min, log_max = np.log10(1.5), np.log10(10.0)
        current_ppp = 10 ** np.random.uniform(log_min, log_max)
        
        img_noisy, _ = torch_forward_model(avg_PPP=current_ppp, photon_flux=img_clean)
        return img_noisy, img_clean

# --- 3. MODEL (UNetMNIST - 2 Pools) ---
class UNetMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        # 28 -> 14
        self.e11 = nn.Conv2d(1, 64, 3, 1, 1)
        self.e12 = nn.Conv2d(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        # 14 -> 7
        self.e21 = nn.Conv2d(64, 128, 3, 1, 1)
        self.e22 = nn.Conv2d(128, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        # Bottleneck
        self.b1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.b2 = nn.Conv2d(256, 256, 3, 1, 1)
        # Up -> 14
        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d11 = nn.Conv2d(256, 128, 3, 1, 1)
        self.d12 = nn.Conv2d(128, 128, 3, 1, 1)
        # Up -> 28
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d21 = nn.Conv2d(128, 64, 3, 1, 1)
        self.d22 = nn.Conv2d(64, 64, 3, 1, 1)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = relu(self.e12(relu(self.e11(x))))
        p1 = self.pool1(e1)
        e2 = relu(self.e22(relu(self.e21(p1))))
        p2 = self.pool2(e2)
        b = relu(self.b2(relu(self.b1(p2))))
        
        u1 = self.up1(b)
        d1 = relu(self.d12(relu(self.d11(torch.cat([u1, e2], dim=1)))))
        u2 = self.up2(d1)
        d2 = relu(self.d22(relu(self.d21(torch.cat([u2, e1], dim=1)))))
        return self.out(d2)

# --- 4. EXECUTOR ---
def train_mnist():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = QIS_MNIST_Base(train=True)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    model = UNetMNIST().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    print("Starting MNIST Training...")
    for epoch in range(5):
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
    
    torch.save(model.state_dict(), "qis_mnist_28x28.pth")
    return model

if __name__ == "__main__":
    train_mnist()
