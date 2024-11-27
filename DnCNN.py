import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, num_channels, num_layers):
        super(DnCNN, self).__init__()
        layers = []
        # First layer (conv + ReLU)
        layers.append(nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers (conv + BN + ReLU)
        for i in range(num_layers-2):
            layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features=64))
            layers.append(nn.ReLU(inplace=True))

        # Last layer (conv only)
        layers.append(nn.Conv2d(in_channels=64, out_channels=num_channels, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn(x) # Residual learning