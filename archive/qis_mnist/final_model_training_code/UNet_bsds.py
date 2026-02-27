#!/usr/bin/python3
import torch
import torch.nn as nn

class Unet_Noise_8x8(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        # --- ENCODER ---
        
        # Layer 1: 128x128 -> 128 filters
        self.e11 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Layer 2: 64x64 -> 256 filters
        self.e21 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Layer 3: 32x32 -> 512 filters
        self.e31 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Layer 4: 16x16 -> 512 filters (Output here is 8x8 after pool)
        self.e41 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # --- BOTTLENECK (8x8) ---
        self.b1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.b2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)


        #DECODER 

        # Up 4: 8x8 -> 16x16
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(1024, 512, kernel_size=3, padding=1) # 512 from up + 512 from skip
        self.d42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Up 3: 16x16 -> 32x32
        self.upconv3 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(512, 256, kernel_size=3, padding=1) # Reducing filters

        # Up 2: 32x32 -> 64x64
        self.upconv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 128, kernel_size=3, padding=1) # Reducing filters

        # Up 1: 64x64 -> 128x128
        self.upconv1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.out_conv = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        # L1
        x1 = self.relu(self.e12(self.relu(self.e11(x))))
        p1 = self.pool(x1)
        # L2
        x2 = self.relu(self.e22(self.relu(self.e21(p1))))
        p2 = self.pool(x2)
        # L3
        x3 = self.relu(self.e32(self.relu(self.e31(p2))))
        p3 = self.pool(x3)
        # L4 (Output is 8x8 at p4)
        x4 = self.relu(self.e42(self.relu(self.e41(p3))))
        p4 = self.pool(x4)

        # Bottleneck
        b = self.relu(self.b2(self.relu(self.b1(p4))))

        # Decoder
        # Up 4
        u4 = self.upconv4(b)
        cat4 = torch.cat([x4, u4], dim=1)
        d4 = self.relu(self.d42(self.relu(self.d41(cat4))))

        # Up 3
        u3 = self.upconv3(d4)
        cat3 = torch.cat([x3, u3], dim=1)
        d3 = self.relu(self.d32(self.relu(self.d31(cat3))))

        # Up 2
        u2 = self.upconv2(d3)
        cat2 = torch.cat([x2, u2], dim=1)
        d2 = self.relu(self.d22(self.relu(self.d21(cat2))))

        # Up 1
        u1 = self.upconv1(d2)
        cat1 = torch.cat([x1, u1], dim=1)
        d1 = self.relu(self.d12(self.relu(self.d11(cat1))))

        return self.out_conv(d1)
