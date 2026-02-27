#!/usr/bin/env python3
from scipy.interpolate import Rbf
import numpy as np
import torch
import cv2


class GainCalculator:
    def __init__(self):
        self.data = np.array([
            [0.5, 90], [1.5, 60], [2.5, 50], [3.25, 30], [6.5, 15], [9.75, 7.5],
            [13, 4.5], [20, 3.2], [26, 2.8], [36, 2.4], [45, 2.2], [54, 1.8],
            [67, 1.5], [80, 1.3], [90, 1.1], [110, 1.05], [130, 0.9],
            [145, 0.65], [155, 0.56], [160, 0.51], [200, 0.4881704]
        ])
        x = self.data[:, 0]
        y = self.data[:, 1]
        self.rbf = Rbf(x, y, function='linear')

    def get_gain(self, avg_PPP, N=1):
        return self.rbf(avg_PPP)


@torch.no_grad()
def torch_forward_model(avg_PPP, photon_flux, QE, theta_dark, sigma_read, N, Nbits, fwc, normalize=True):
    min_val = 0
    max_val = 2 ** Nbits - 1

    gain = GainCalculator().get_gain(avg_PPP)

    avg_PPP = torch.tensor(avg_PPP, dtype=torch.float32)
    photon_flux = torch.tensor(photon_flux, dtype=torch.float32)
    QE = torch.tensor(QE, dtype=torch.float32)
    theta_dark = torch.tensor(theta_dark, dtype=torch.float32)
    sigma_read = torch.tensor(sigma_read, dtype=torch.float32)
    gain = torch.tensor(gain, dtype=torch.float32)
    fwc = torch.tensor(fwc, dtype=torch.float32)

    theta = photon_flux * (avg_PPP / (torch.mean(photon_flux) + 1e-4))
    lam = ((QE * theta) + theta_dark) / N

    img_out = torch.zeros_like(theta)

    for _ in range(N):
        tmp = torch.poisson(lam)
        tmp = torch.clamp(tmp, 0, fwc)
        tmp = tmp + torch.normal(
            mean=0.0,
            std=sigma_read,
            size=tmp.shape,
            device=tmp.device
        )
        tmp = torch.round(tmp * gain * max_val / fwc)
        tmp = torch.clamp(tmp, min_val, max_val)
        img_out += tmp / N

    if normalize:
        img_out /= max_val

    gain /= 90.0
    return img_out, gain


if __name__ == "__main__":
    avg_PPP = 1.5 

    photon_flux = cv2.imread('./tennis.jpg', cv2.IMREAD_GRAYSCALE)
    if photon_flux is None:
        raise FileNotFoundError("Image not found")

    photon_flux = photon_flux.astype(np.float32) / 255.0

    QE = 0.6
    theta_dark = 1.6
    sigma_read = 0.2
    N = 1 
    Nbits = 3
    fwc = 20 


    img_out, gain = torch_forward_model(
        avg_PPP, photon_flux, QE, theta_dark, sigma_read, N, Nbits, fwc
    )

    import matplotlib.pyplot as plt

    # Convert to integer [0, 255] for standard image format
    save_image = (img_out.detach().cpu().squeeze().numpy() * 255).astype(np.uint8)
    cv2.imwrite("image_tested_noise.jpg", save_image)
