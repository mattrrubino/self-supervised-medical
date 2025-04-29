import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from exemplar_components_2d import Encoder2D, TripletLoss, apply_2d_transformations
from exemplar_components_2d.slice_extractor import extract_2d_slices_from_3d

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def test_components():
    print("Testing 2D Exemplar Network Components...")
    
    # dummy 3D volume
    print("\n1. Testing Slice Extraction...")
    dummy_volume = torch.rand(1, 64, 64, 64)
    slices = extract_2d_slices_from_3d(dummy_volume, num_slices=5, axis=0)
    print(f"  Original volume shape: {dummy_volume.shape}")
    print(f"  Number of extracted slices: {len(slices)}")
    print(f"  Shape of each slice: {slices[0].shape}")
    
    # Testing encoder
    print("\n2. Testing Encoder...")
    dummy_input = slices[0]  # Take one slice
    
    # Initializing  encoder
    encoder = Encoder2D(in_channels=1, embedding_size=1024)
    
    # Forward pass
    embeddings, features = encoder(dummy_input.unsqueeze(0))  # Add batch dimension
    print(f"  Encoder output shape: {embeddings.shape}")
    print(f"  Total parameters: {sum(p.numel() for p in encoder.parameters())}")
    
    print("\n  Feature map shapes:")
    for name, feature_map in features.items():
        print(f"    {name}: {feature_map.shape}")
    
    # Testing transformation
    print("\n3. Testing Transformations...")
    transformed = apply_2d_transformations(dummy_input)
    print(f"  Original image shape: {dummy_input.shape}")
    print(f"  Transformed image shape: {transformed.shape}")
    
    # Visualization of a central slice (simple)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot original
        axes[0].imshow(dummy_input[0].numpy(), cmap='gray')
        axes[0].set_title("Original slice")
        axes[0].axis('off')
        
        # Plot transformed
        axes[1].imshow(transformed[0].numpy(), cmap='gray')
        axes[1].set_title("Transformed slice")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig("2d_transformation_example.png")
        plt.close()
        
        print("  Example visualization saved as '2d_transformation_example.png'")
    except Exception as e:
        print(f"  Could not create visualization: {e}")
    
    # Testing triplet loss
    print("\n4. Testing Triplet Loss...")
    
    # dummy embeddings
    batch_size = 8
    embedding_size = 1024
    
    anchor = torch.randn(batch_size, embedding_size)
    positive = anchor + 0.1 * torch.randn(batch_size, embedding_size)  # Similar to anchor
    negative = torch.randn(batch_size, embedding_size)  # Different from anchor
    
    # Initializing the loss function
    triplet_loss = TripletLoss(margin=1.0)
    
    # loss
    loss = triplet_loss(anchor, positive, negative)
    print(f"  Margin: {triplet_loss.margin}")
    print(f"  Batch size: {batch_size}")
    print(f"  Embedding size: {embedding_size}")
    print(f"  Loss value: {loss.item()}")
    
    print("\nAll components are working correctly")
    
    return features

if __name__ == "__main__":
    test_components()
