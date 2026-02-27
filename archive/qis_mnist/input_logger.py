from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
import torch

class QIS_MNIST_Dataset(Dataset):
    def __init__(self, root_dir='./data', train=True, download=True):
        # NO RESIZE: We keep the native 28x28 resolution
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.mnist_data = datasets.MNIST(
            root=root_dir,
            train=train,
            download=download,
            transform=self.transform
        )

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        # 1. Get Clean Image [1, 28, 28]
        img_clean, label = self.mnist_data[idx] 
        # 2. Log-Uniform Sampling
        log_min = np.log10(1.5)   
        log_max = np.log10(10.0)  
        random_exponent = np.random.uniform(log_min, log_max)
        current_ppp = 10 ** random_exponent
        # 3. Simulate Noise
        img_noisy, _ = torch_forward_model(
            avg_PPP=current_ppp,
            photon_flux=img_clean,
            QE=0.6,
            theta_dark=0.001, 
            sigma_read=0.2,
            N=1,
            Nbits=3,
            fwc=20
        )
        return img_noisy, img_clean
