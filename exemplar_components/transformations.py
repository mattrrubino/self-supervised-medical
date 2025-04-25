import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

def apply_3d_transformations(volume):
    """
    transformations to create a positive sample for the 3D Exemplar Network
    
    paper's specifications:
    - Random flipping along arbitrary axis (50% chance)
    - Random rotation along arbitrary axis (50% chance)
    - Random brightness and contrast (50% chance)
    - Random zooming (20% chance)
    
    Args:
        volume: 3D volume tensor of shape [C, D, H, W]
        
    Returns:
        Transformed volume tensor of same shape
    """
    # Clone the tensor to avoid in-place modifications
    transformed = volume.clone()
    
    # Random flipping (50% chance)
    if random.random() < 0.5:
        axis = random.randint(1, 3)  # Dimensions 1, 2, 3 (D, H, W)
        transformed = torch.flip(transformed, dims=[axis])
    
    # Random rotation (50% chance)
    if random.random() < 0.5:
        k = random.randint(1, 3)  # Rotate by k*90 degrees
        axes = [(1, 2), (1, 3), (2, 3)]  # Possible rotation planes
        dims = random.choice(axes)
        transformed = torch.rot90(transformed, k, dims=dims)
    
    # Random brightness and contrast (50% chance)
    if random.random() < 0.5:
        # Brightness adjustment
        brightness_factor = random.uniform(0.8, 1.2)
        transformed = transformed * brightness_factor
        
        # Contrast adjustment
        contrast_factor = random.uniform(0.8, 1.2)
        mean = torch.mean(transformed)
        transformed = (transformed - mean) * contrast_factor + mean
    
    # Random zoom (20% chance)
    if random.random() < 0.2:
        zoom_factor = random.uniform(0.85, 1.15)
        
        if zoom_factor != 1:
            # Get current dimensions
            c, d, h, w = transformed.shape
            
            # Calculate new dimensions
            new_d = int(d * zoom_factor)
            new_h = int(h * zoom_factor)
            new_w = int(w * zoom_factor)
            
            # Resize
            transformed = F.interpolate(
                transformed.unsqueeze(0),  # Add batch dimension
                size=(new_d, new_h, new_w),
                mode='trilinear',
                align_corners=False
            ).squeeze(0)  # Remove batch dimension
            
            # Crop or pad to match original size
            if zoom_factor > 1:  # Zoomed in, need to crop
                d_diff = new_d - d
                h_diff = new_h - h
                w_diff = new_w - w
                
                d_start = d_diff // 2
                h_start = h_diff // 2
                w_start = w_diff // 2
                
                transformed = transformed[:, d_start:d_start+d, h_start:h_start+h, w_start:w_start+w]
            else:  # Zoomed out, need to pad
                d_diff = d - new_d
                h_diff = h - new_h
                w_diff = w - new_w
                
                d_pad_before = d_diff // 2
                h_pad_before = h_diff // 2
                w_pad_before = w_diff // 2
                
                d_pad_after = d_diff - d_pad_before
                h_pad_after = h_diff - h_pad_before
                w_pad_after = w_diff - w_pad_before
                
                transformed = F.pad(
                    transformed,
                    (w_pad_before, w_pad_after, h_pad_before, h_pad_after, d_pad_before, d_pad_after)
                )
    
    # Ensure values are clipped to the valid range
    transformed = torch.clamp(transformed, 0, 1)
    
    return transformed

# This code will run if this file is executed directly
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Create a dummy 3D volume
    # Using random noise for visualization
    volume = torch.randn(1, 16, 16, 16)
    
    # Apply transformations
    transformed = apply_3d_transformations(volume)
    
    # Print information
    print("3D Transformations Implementation:")
    print(f"Original volume shape: {volume.shape}")
    print(f"Transformed volume shape: {transformed.shape}")
    
    # Visualize a central slice (for simple visualization)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot original
    axes[0].imshow(volume[0, 8, :, :].numpy())
    axes[0].set_title("Original (middle slice)")
    axes[0].axis('off')
    
    # Plot transformed
    axes[1].imshow(transformed[0, 8, :, :].numpy())
    axes[1].set_title("Transformed (middle slice)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("transformation_example.png")
    plt.close()
    
    print(f"\nExample visualization saved as 'transformation_example.png'")
    print("3D Transformations successfully implemented!")
