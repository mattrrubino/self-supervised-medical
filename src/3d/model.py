import torch
import torch.nn.functional as F
from torch import nn


def he_normal_init(layer: nn.Module):
    if layer.weight is not None:
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu") # pyright: ignore
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0) # pyright: ignore


def glorot_uniform_init(layer: nn.Module):
    if layer.weight is not None:
        nn.init.xavier_uniform_(layer.weight) # pyright: ignore
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0) # pyright: ignore


class MulticlassClassifier(nn.Module):
    def __init__(self, in_features, out_features, p=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = p

        self.l0 = nn.Linear(in_features, 2048)
        self.bn0 = nn.BatchNorm1d(2048)
        self.d0 = nn.Dropout(p)
        self.l1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.d1 = nn.Dropout(p)
        self.l2 = nn.Linear(1024, out_features)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.l0(x))
        x = self.bn0(x)
        x = self.d0(x)
        x = F.relu(self.l1(x))
        x = self.bn1(x)
        x = self.d1(x)
        x = self.l2(x)
        return x


class Conv3dBlock(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size=3, padding="same", dropout=0.5):
        super().__init__()
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.dropout = dropout

        self.conv1 = nn.Conv3d(filters_in, filters_out, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(filters_out, 1e-3, 1e-2)
        self.d = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(filters_out, filters_out, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(filters_out, 1e-3, 1e-2)
        he_normal_init(self.conv1)
        he_normal_init(self.conv2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.d(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        return x


class UNet3dEncoder(nn.Module):
    def __init__(self, filters_in, filters=16, num_layers=4, kernel_size=3):
        super().__init__()
        self.filters_in = filters_in
        self.filters = filters
        self.num_layers = num_layers
        self.kernel_size = kernel_size

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
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        skip_outputs = []
        for layer, down in zip(self.layers, self.downsample):
            x = layer(x)
            skip_outputs.append(x)
            x = down(x)
        x = self.head(x)
        x = self.pool(x)
        return x, skip_outputs


class UNet3dDecoder(nn.Module):
    def __init__(self, filters_out, filters=16, num_layers=4, kernel_size=3):
        super().__init__()
        self.filters_out = filters_out
        self.filters = filters
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        self.unpool = nn.Upsample(scale_factor=(2, 2, 2), mode="nearest")
        self.layers = nn.ModuleList([
            Conv3dBlock(filters*(2**i), filters*(2**(i-1)), kernel_size=kernel_size)
            for i in range(num_layers, 0, -1)
        ])
        self.upsample = nn.ModuleList([
            nn.ConvTranspose3d(filters*(2**i), filters*(2**(i-1)), 2, 2)
            for i in range(num_layers, 0, -1)
        ])
        self.head = nn.Conv3d(filters, filters_out, 1)

        for conv in self.upsample:
            glorot_uniform_init(conv)
        glorot_uniform_init(self.head)

    def forward(self, x, skip_outputs):
        x = self.unpool(x)
        for layer, up, skip_output in zip(self.layers, self.upsample, reversed(skip_outputs)):
            x = up(x)
            x = torch.cat((x, skip_output), dim=1)
            x = layer(x)
        x = self.head(x)
        x = F.softmax(x, dim=1)
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


class SkipStripper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[0]


class MultipatchClassifier(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        b = len(x)
        x = torch.flatten(x, end_dim=1)
        x, _ = self.encoder(x)
        x = x.reshape(b, -1)
        x = self.head(x)
        return x


class MultipatchEmbedder(nn.Module):
    def __init__(self, encoder, hidden_dim, code_size, channels):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleList(
            nn.Linear(hidden_dim, code_size) for _ in range(channels)
        )

    def forward(self, x):
        out = []
        for i,head in enumerate(self.heads):
            patch = x[:,i].unsqueeze(1)
            t, _ = self.encoder(patch)
            t = t.flatten(start_dim=1)
            t = head(t)
            t = F.sigmoid(t)
            out.append(t)
        return out


def create_unet3d(filters_in=1, filters_out=3, filters=16, num_layers=4):
    encoder = UNet3dEncoder(filters_in, filters, num_layers)
    decoder = UNet3dDecoder(filters_out, filters, num_layers)
    model = UNet3d(encoder, decoder)
    return encoder, decoder, model


def create_multiclass_classifier(encoder, classes, _, data_dim=128):
    filters = encoder.filters
    num_layers = encoder.num_layers
    hidden_dim = int((filters*2**num_layers)*((data_dim//(2**(num_layers+1)))**3))
    head = MulticlassClassifier(hidden_dim, classes)
    classifier = torch.nn.Sequential(encoder, SkipStripper(), head)
    return classifier


def create_multipatch_classifier(encoder, classes, patches, data_dim=39):
    filters = encoder.filters
    num_layers = encoder.num_layers
    hidden_dim = int((filters*2**num_layers)*((data_dim//(2**(num_layers+1)))**3))
    head = MulticlassClassifier(patches*hidden_dim, classes)
    classifier = MultipatchClassifier(encoder, head)
    return classifier


def create_multipatch_embedder(encoder, classes, patches, data_dim=128):
    filters = encoder.filters
    num_layers = encoder.num_layers
    hidden_dim = int((filters*2**num_layers)*((data_dim//(2**(num_layers+1)))**3))
    embedder = MultipatchEmbedder(encoder, hidden_dim, classes, patches)
    return embedder

