import torch.nn as nn
from torch.nn.functional import relu

class UNetMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1) 
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2) # [64, 14, 14]

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2) # [128, 7, 7]

        # Bottleneck
        self.b1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.b2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) 
        self.d11 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) 
        self.d21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.outconv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe12 = relu(self.e12(relu(self.e11(x))))
        xp1 = self.pool1(xe12)

        xe22 = relu(self.e22(relu(self.e21(xp1))))
        xp2 = self.pool2(xe22)

        # Bottleneck
        xb = relu(self.b2(relu(self.b1(xp2))))

        # Decoder
        xu1 = self.upconv1(xb)
        xu11 = torch.cat([xu1, xe22], dim=1)
        xd1 = relu(self.d12(relu(self.d11(xu11))))

        xu2 = self.upconv2(xd1)
        xu22 = torch.cat([xu2, xe12], dim=1)
        xd2 = relu(self.d22(relu(self.d21(xu22))))

        return self.outconv(xd2)
