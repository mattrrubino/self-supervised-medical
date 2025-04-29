import torch
import torch.nn as nn

class Encoder2D(nn.Module):
    """
    2D Encoder for the Exemplar Network as described in the paper
    "3D Self-Supervised Methods for Medical Imaging"
    """
    def __init__(self, in_channels=1, embedding_size=1024):
        super(Encoder2D, self).__init__()
        
        # downsampling path
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )
        
        # taking global average pooling & embedding
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, embedding_size)
        
        # Non-linear transformation for final embedding
        self.fc_relu = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x = self.pool1(x1)
        
        x2 = self.enc2(x)
        x = self.pool2(x2)
        
        x3 = self.enc3(x)
        x = self.pool3(x3)
        
        x4 = self.enc4(x)
        x = self.pool4(x4)
        
        x = self.bottleneck(x)
        
        # Global pooling and embedding
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        
        # Non-linear transformation
        x = self.fc_relu(x)
        
        return x, {"enc1": x1, "enc2": x2, "enc3": x3, "enc4": x4}

# This code should run if the file is executed directly
if __name__ == "__main__":
    # Creating a dummy input
    dummy_input = torch.randn(1, 1, 128, 128)  # Batch size 1, 1 channel, 128x128 image
    
    # Initializing the encoder
    encoder = Encoder2D(in_channels=1, embedding_size=1024)
    
    print("2D Exemplar Encoder Architecture:")
    print(f"Input channels: 1")
    print(f"Embedding size: 1024")
    print(f"Parameter count: {sum(p.numel() for p in encoder.parameters())}")
    
    # Forward pass
    embeddings, features = encoder(dummy_input)
    
    print("\nForward Pass Results:")
    print(f"Embedding shape: {embeddings.shape}")
    
    print("\nFeature Map Shapes:")
    for name, feature_map in features.items():
        print(f"  {name}: {feature_map.shape}")
    
    print("\n2D Exemplar Encoder successfully implemented")
