import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, num_channels):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(num_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.up4 = self.upconv(1024, 512)
        self.dec4 = self.conv_block(1024, 512)
        self.up3 = self.upconv(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = self.upconv(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = self.upconv(128, 64)
        self.dec1 = self.conv_block(128, 64)
        
        # Final output
        self.out = nn.Conv2d(64, num_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def downsample(self, x):
        return nn.MaxPool2d(kernel_size=2, stride=2)(x)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.downsample(e1))
        e3 = self.enc3(self.downsample(e2))
        e4 = self.enc4(self.downsample(e3))
        
        # Bottleneck
        b = self.bottleneck(self.downsample(e4))
        
        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        # Final output
        return self.out(d1)
    





