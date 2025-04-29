import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

def apply_2d_transformations(image):
    """
    Applying transformations to create a positive sample for the 2D Exemplar Network
    
    - Random flipping along arbitrary axis (50% chance)
    - Random rotation (50% chance)
    - Random brightness and contrast (50% chance)
    - Random zooming (20% chance)
    
    Args:
        image: 2D image tensor of shape [C, H, W]
        
    Returns:
        Transformed image tensor of same shape
    """
    # Cloninh the tensor to avoid in-place modifications
    transformed = image.clone()
    
    # Random flipping (50% chance)
    if random.random() < 0.5:
        axis = random.randint(1, 2)  # Dimensions 1, 2 (H, W)
        transformed = torch.flip(transformed, dims=[axis])
    
    # Random rotation (50% chance)
    if random.random() < 0.5:
        k = random.randint(1, 3)  # Rotate by k*90 degrees
        transformed = torch.rot90(transformed, k, dims=[1, 2])
    
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
            c, h, w = transformed.shape
            
            # Calculating new dimensions
            new_h = int(h * zoom_factor)
            new_w = int(w * zoom_factor)
            
            # Resizing
            transformed = F.interpolate(
                transformed.unsqueeze(0),  # Add batch dimension
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # Remove batch dimension
            
            # Crop or pad to match original size
            if zoom_factor > 1:  # Zoomed in, need to crop
                h_diff = new_h - h
                w_diff = new_w - w
                
                h_start = h_diff // 2
                w_start = w_diff // 2
                
                transformed = transformed[:, h_start:h_start+h, w_start:w_start+w]
            else:  # Zoomed out, need to pad
                h_diff = h - new_h
                w_diff = w - new_w
                
                h_pad_before = h_diff // 2
                w_pad_before = w_diff // 2
                
                h_pad_after = h_diff - h_pad_before
                w_pad_after = w_diff - w_pad_before
                
                transformed = F.pad(
                    transformed,
                    (w_pad_before, w_pad_after, h_pad_before, h_pad_after)
                )
    
    # Ensure values are clipped to the valid range
    transformed = torch.clamp(transformed, 0, 1)
    
    return transformed

# This code will run if this file is executed directly
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # dummy 2D image
    # Using random noise for visualization
    image = torch.rand(1, 128, 128)
    
    # Applying transformations
    transformed = apply_2d_transformations(image)
    
    print("2D Transformations Implementation:")
    print(f"Original image shape: {image.shape}")
    print(f"Transformed image shape: {transformed.shape}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot original
    axes[0].imshow(image[0].numpy(), cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Plot transformed
    axes[1].imshow(transformed[0].numpy(), cmap='gray')
    axes[1].set_title("Transformed")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("2d_transformation_example.png")
    plt.close()
    
    print(f"\nExample visualization saved as '2d_transformation_example.png'")
    print("2D Transformations successfully implemented")
