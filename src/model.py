import torch.nn.functional as F
from torch import nn


class Conv3dBlock(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size=3, padding="same", dropout=0.3):
        self.conv1 = nn.Conv3d(filters_in, filters_out, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(filters_out)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(filters_out, filters_out, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(filters_out)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        return x


class UNet3dEncoder(nn.Module):
    def __init__(self, filters=16, num_layers=4, kernel_size=3):
        self.layers = [
            Conv3dBlock(filters*(2**i), filters*(2**(i+1)), kernel_size=kernel_size)
            for i in range(num_layers)
        ]
        self.downsample = [
            nn.MaxPool3d(kernel_size)
            for _ in range(num_layers)
        ]
        self.head = Conv3dBlock(filters*(2**num_layers), filters*(2**(num_layers+1)))

    def forward(self, x):
        for layer, down in zip(self.layers, self.downsample):
            x = layer(x)
            x = down(x)
        x = self.head(x)
        return x


# TODO: Implement
class UNet3dDecoder(nn.Module):
    def __init__(self):
        pass

    def forward(self, x, skip):
        pass

