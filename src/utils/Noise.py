#!/usr/bin/python3
import torch
from scipy.interpolate import Rbf
import numpy as np
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
def qis_noise(avg_PPP, photon_flux, QE=0.6, 
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

