#!/usr/bin/python3
import torch
import torch.nn as nn
""" need relu and conv2d classes from pytorch"""
class Unet_mnist(nn.Module):
    def __init__(self):
        super().__init__()
        # start creating the appropriately sampled matrices
        #initial matrix image is x0
        #we need this padding to maintain the size during convolution
        #28x28
        self.relu = nn.ReLU()
        self.e11 = nn.Conv2d(1, 64, kernel_size=3,padding=1)
        #relu at each step
        self.e12 = nn.Conv2d(64, 64, kernel_size=3,padding=1)
        #this are linearly connected so total number of ernels created till now is 128
        #creating a maxpooling matrix it's more of a function than a matrix
        self.pl1 = nn.MaxPool2d(2, 2)
        #this is 
        #14x14
        self.e21 = nn.Conv2d(64, 128, kernel_size=3,padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3,padding=1)
        self.pl2 = nn.MaxPool2d(2, 2)
        #7x7
        #bottle neck
        self.b31 = nn.Conv2d(128, 256, kernel_size=3,padding=1)
        self.b32 = nn.Conv2d(256, 256, kernel_size=3,padding=1)

        #starting start upscaling 
        #14x14
        self.upconv1= nn.ConvTranspose2d(256, 128, kernel_size=2,stride = 2)
        self.d11 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        #28x28
        self.upconv2= nn.ConvTranspose2d(128, 64, kernel_size=2, stride = 2) 
        self.d21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
    def forward(self, input_image) :
        #first_encoder_layer
        e1 = self.relu(self.e12(self.relu(self.e11(input_image))))
        pl1 = self.pl1(e1)

        #second_encoder_layer
        e2 = self.relu(self.e22(self.relu(self.e21(pl1))))
        pl2 = self.pl2(e2)

        #bottle_neck_layer
        b = self.relu(self.b32(self.relu(self.b31(pl2))))

        #first_decoder_layer, level 2
        up1 = self.upconv1(b)
        u3 = torch.cat([e2,up1], dim = 1)
        d1 = self.relu(self.d12(self.relu(self.d11(u3))))

        #second_decoder_layer, level1
        up2 = self.upconv2(d1)
        u2 = torch.cat([e1,up2], dim = 1)
        d2 = self.relu(self.d22(self.relu(self.d21(u2))))

        return self.out_conv(d2)
