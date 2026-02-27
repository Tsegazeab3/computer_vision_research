from UNET import UNet
from scipy.interpolate import Rbf
import numpy as np
import torch
from torchvision.transforms.transforms import ToTensor
from torchvision import datasets
from torchvision import datasets, transforms
import matplotlib.pyplot  as plt

""" downloading and handling stange of the training process, 

        inheriting from the datasets.MNIST
        ret 
        

"""
class QIS_MNIST_Base(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ppp_range = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    def __getitem__(self, index):
            img_raw = self.data[index] 
            img_clean_pil = transforms.ToPILImage()(img_raw)
            photon_flux = np.array(img_clean_pil).astype(np.float32) / 255.0
            low, high = np.log(0.5), np.log(20.0)
            selected_ppp = np.exp(np.random.uniform(low, high))
            img_out_tensor, _ = torch_forward_model(
                avg_PPP=selected_ppp, photon_flux=photon_flux
            )
            target_tensor = transforms.ToTensor()(img_clean_pil)
            return img_out_tensor, target_tensor


'''
Gain function is mapping avg_PPP to gain value.
avg_PPP can be converted to lux assuming exposure time = 1/2000 sec. Ex: 3.25PPP -> 1 lux and 9.75 PPP -> 3 lux.
This mapping function works for QIS sensors with pixel size 1.1um and full well capacity 200e-.
Sensor Parameters assumed:
QE = 0.6
theta_dark = 1.6 e-/pix/frame
sigma_read = 0.2 e- RMS
Nbits = 3 bits
N = 1 frame
fwc = 200 e-
'''
def get_gain(avg_PPP, N=1):
    # Evaluate the polynomial at avg_PPP
    data = np.array([
        [0.5, 90], [1.5, 60], [2.5, 50], [3.25, 30], [6.5, 15], [9.75, 7.5],
        [13, 4.5], [20, 3.2], [26, 2.8], [36, 2.4], [45, 2.2], [54, 1.8],
        [67, 1.5], [80, 1.3], [90, 1.1], [110, 1.05], [130, 0.9], [145, 0.65],
        [155, 0.56], [160, 0.51], [200, 0.4881704]
        ])
    x = data[:, 0]
    y = data[:, 1]
    rbf = Rbf(x, y, function='linear')
    return rbf(avg_PPP) * N


@torch.no_grad()
def torch_forward_model(avg_PPP, photon_flux, QE=0.6, 
                theta_dark=1.6, sigma_read=0.2, N=1, 
                Nbits=3, fwc=200, normalize = True):
    min_val = 0
    max_val = 2 ** Nbits - 1
    gain = get_gain(avg_PPP)

    # Convert all inputs to torch tensors if they are not already
    avg_PPP = torch.tensor(avg_PPP, dtype=torch.float32)
    photon_flux = torch.tensor(photon_flux, dtype=torch.float32)
    QE = torch.tensor(QE, dtype=torch.float32)
    theta_dark = torch.tensor(theta_dark, dtype=torch.float32)
    sigma_read = torch.tensor(sigma_read, dtype=torch.float32)
    gain = torch.tensor(gain, dtype=torch.float32)
    fwc = torch.tensor(fwc, dtype=torch.float32)

    # Calculate theta
    theta = photon_flux * (avg_PPP / (torch.mean(photon_flux) + 0.0001))

    # Calculate lam
    lam = ((QE * theta) + theta_dark) / N

    #c, m, n = theta.shape
    img_out = torch.zeros_like(theta)

    for i in range(N):
        # Poisson sampling
        # print(lam.min(), lam.max(), lam.shape, lam.type())
        tmp = torch.poisson(lam)

        # Clipping to full well capacity (fwc)
        tmp = torch.clamp(tmp, 0, fwc.item())

        # Adding read noise
        tmp = tmp + torch.normal(mean=0, std=sigma_read, size=img_out.shape, device=theta.device)

        # Amplifying, quantizing, and clipping
        tmp = torch.round(tmp * gain * max_val / fwc)
        tmp = torch.clamp(tmp, min_val, max_val)

        # Summing up the images
        img_out = img_out + tmp

    # Averaging over N frames
    img_out = img_out / N
    if normalize:
        img_out = img_out/max_val

    #normalized gain 
    gain = gain / 90.0
    return img_out, gain

def load_data(batch_size=128):
    """
    Initializes the custom QIS datasets and returns DataLoaders for training and testing.
    """
    # 1. Instantiate the custom datasets
    train_dataset = QIS_MNIST_Base(root='./data', train=True, download=True)
    test_dataset = QIS_MNIST_Base(root='./data', train=False, download=True)

    # 2. Create DataLoaders with NCHW batching
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader
