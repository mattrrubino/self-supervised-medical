import torch
import torch.nn.functional as F
from torch import nn


class Conv3dBlock(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size=3, padding="same", dropout=0.3):
        super().__init__()
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
    def __init__(self, filters_in, filters=16, num_layers=4, kernel_size=3):
        super().__init__()
        self.layers = nn.ModuleList([
            Conv3dBlock(filters_in, filters, kernel_size=kernel_size),
            *[Conv3dBlock(filters*(2**i), filters*(2**(i+1)), kernel_size=kernel_size)
            for i in range(num_layers-1)]
        ])
        self.downsample = nn.ModuleList([
            nn.MaxPool3d(2)
            for _ in range(num_layers)
        ])
        self.head = Conv3dBlock(filters*(2**(num_layers-1)), filters*(2**num_layers))

    def forward(self, x):
        skip_outputs = []
        for layer, down in zip(self.layers, self.downsample):
            x = layer(x)
            skip_outputs.append(x)
            x = down(x)
        x = self.head(x)
        return x, skip_outputs


class UNet3dDecoder(nn.Module):
    def __init__(self, filters_out, filters=16, num_layers=4, kernel_size=3):
        super().__init__()
        self.layers = nn.ModuleList([
            Conv3dBlock(filters*(2**i), filters*(2**(i-1)), kernel_size=kernel_size)
            for i in range(num_layers, 0, -1)
        ])
        self.upsample = nn.ModuleList([
            nn.ConvTranspose3d(filters*(2**i), filters*(2**(i-1)), 2, 2)
            for i in range(num_layers, 0, -1)
        ])
        self.head = nn.Conv3d(filters, filters_out, 1)

    def forward(self, x, skip_outputs):
        for layer, up, skip_output in zip(self.layers, self.upsample, reversed(skip_outputs)):
            x = up(x)
            x = torch.cat((x, skip_output), dim=1)
            x = layer(x)
        x = self.head(x)
        return x


class UNet3d(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x, skip_outputs = self.encoder(x)
        x = self.decoder(x, skip_outputs)
        return x

