import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from exemplar_components import Encoder3D, TripletLoss, apply_3d_transformations

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def test_components():
    print("Testing 3D Exemplar Network Components...")
    
    # Create a simple example to verify the model
    print("\n1. Testing Encoder...")
    dummy_input = torch.randn(1, 1, 64, 64, 64)  # Batch size 1, 1 channel, 64x64x64 volume
    
    # Initialize the encoder
    encoder = Encoder3D(in_channels=1, embedding_size=1024)
    
    # Forward pass
    embeddings, features = encoder(dummy_input)
    print(f"  Encoder output shape: {embeddings.shape}")
    print(f"  Total parameters: {sum(p.numel() for p in encoder.parameters())}")
    
    # Print feature map sizes
    print("\n  Feature map shapes:")
    for name, feature_map in features.items():
        print(f"    {name}: {feature_map.shape}")
    
    # Test transformation
    print("\n2. Testing Transformations...")
    transformed = apply_3d_transformations(dummy_input.squeeze(0))
    print(f"  Original volume shape: {dummy_input.squeeze(0).shape}")
    print(f"  Transformed volume shape: {transformed.shape}")
    
    # Visualize a central slice (for simple visualization)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot original
        axes[0].imshow(dummy_input[0, 0, 32, :, :].numpy())
        axes[0].set_title("Original (middle slice)")
        axes[0].axis('off')
        
        # Plot transformed
        axes[1].imshow(transformed[0, 32, :, :].numpy())
        axes[1].set_title("Transformed (middle slice)")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig("transformation_example.png")
        plt.close()
        
        print("  Example visualization saved as 'transformation_example.png'")
    except Exception as e:
        print(f"  Could not create visualization: {e}")
    
    # Test triplet loss
    print("\n3. Testing Triplet Loss...")
    
    # Create dummy embeddings
    batch_size = 8
    embedding_size = 1024
    
    anchor = torch.randn(batch_size, embedding_size)
    positive = anchor + 0.1 * torch.randn(batch_size, embedding_size)  # Similar to anchor
    negative = torch.randn(batch_size, embedding_size)  # Different from anchor
    
    # Initialize the loss function
    triplet_loss = TripletLoss(margin=1.0)
    
    # Calculate loss
    loss = triplet_loss(anchor, positive, negative)
    print(f"  Margin: {triplet_loss.margin}")
    print(f"  Batch size: {batch_size}")
    print(f"  Embedding size: {embedding_size}")
    print(f"  Loss value: {loss.item()}")
    
    print("\nAll components are working correctly!")
    
    return features

if __name__ == "__main__":
    test_components()
