from torch.utils.data import Dataset
from torchvision import datasets, transforms
import random
from Noise import qis_noise 

class QIS_MNIST_Dataset(Dataset):
    def __init__(self, root_dir='./data', train=True, download=True):
        self.mnist_data = datasets.MNIST(
            root=root_dir,
            train=train,
            download=download,
            transform = transforms.ToTensor()
        )
    def __getitem__(self, idx):
        img_clean, label = self.mnist_data[idx] 
        img_noisy, _ = qis_noise(
            avg_PPP= 1,
            photon_flux=img_clean,
            QE=0.6,
            theta_dark=0.001, 
            sigma_read=0.2,
            N=1,
            Nbits=3,
            fwc=20
        )
        img_clean_binary = (img_clean > 0.5).float()
        return img_noisy, img_clean_binary
class QIS_bsd_Dataset()
