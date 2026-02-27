#!/usr/bin/python3
import torch
import torch.nn as nn



class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.match_dims = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.match_dims(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class ResUNet_Gain(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1 = ResidualBlock(1, 64)
        self.e2 = ResidualBlock(64, 128)
        self.e3 = ResidualBlock(128, 256)
        self.e4 = ResidualBlock(256, 512)
        self.b = ResidualBlock(512, 1024)
        self.gain_fc = nn.Linear(1, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.d4 = ResidualBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d3 = ResidualBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d2 = ResidualBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d1 = ResidualBlock(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x, g):
        c1 = self.e1(x); p1 = self.pool(c1)
        c2 = self.e2(p1); p2 = self.pool(c2)
        c3 = self.e3(p2); p3 = self.pool(c3)
        c4 = self.e4(p3); p4 = self.pool(c4)
        bn = self.b(p4)
        g_emb = torch.sigmoid(self.gain_fc(g)).view(-1, 1024, 1, 1)
        bn = bn * g_emb
        u4 = self.up4(bn); d4 = self.d4(torch.cat([u4, c4], dim=1))
        u3 = self.up3(d4); d3 = self.d3(torch.cat([u3, c3], dim=1))
        u2 = self.up2(d3); d2 = self.d2(torch.cat([u2, c2], dim=1))
        u1 = self.up1(d2); d1 = self.d1(torch.cat([u1, c1], dim=1))
        return self.out(d1)


