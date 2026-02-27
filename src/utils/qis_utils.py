#!/usr/bin/env python3
from scipy.interpolate import Rbf
import numpy as np
import torch
import cv2


class GainCalculator:
    '''
    Gain class is mapping avg_PPP to gain value.
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
    def __init__(self):
        # Define the (x, y) pairs
        self.data = np.array([
            [0.5, 90], [1.5, 60], [2.5, 50], [3.25, 30], [6.5, 15], [9.75, 7.5],
            [13, 4.5], [20, 3.2], [26, 2.8], [36, 2.4], [45, 2.2], [54, 1.8],
            [67, 1.5], [80, 1.3], [90, 1.1], [110, 1.05], [130, 0.9], [145, 0.65],
            [155, 0.56], [160, 0.51], [200, 0.4881704]
            ])
        x = self.data[:, 0]
        y = self.data[:, 1]

        # Fit an RBF interpolator
        self.rbf = Rbf(x, y, function='linear')

    def get_gain(self, avg_PPP, N=1):
        # Evaluate the polynomial at avg_PPP
        return self.rbf(avg_PPP) * N


@torch.no_grad()
def torch_forward_model(avg_PPP, photon_flux, QE, theta_dark, sigma_read, N, Nbits, fwc, normalize = True):
    min_val = 0
    max_val = 2 ** Nbits - 1
    gain_func = GainCalculator()
    gain = gain_func.get_gain(avg_PPP)
    # print(f"Using QIS gain value: {gain}")
    
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


if __name__ == "__main__":
    # Example usage
    avg_PPP = 0.5  # Example average photons per pixel
    photon_flux = cv2.imread('./tennis.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    QE = 0.6
    theta_dark = 1.6
    sigma_read = 0.2
    N = 1
    Nbits = 3
    fwc = 200

    img_out, gain = torch_forward_model(avg_PPP, photon_flux, QE, theta_dark, sigma_read, N, Nbits, fwc)

    #visualize photon_flux and img_out
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.title('Input Photon Flux')
    plt.imshow(photon_flux, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Simulated QIS Output')
    plt.imshow(img_out.cpu().numpy(), cmap='gray')
    plt.show()


