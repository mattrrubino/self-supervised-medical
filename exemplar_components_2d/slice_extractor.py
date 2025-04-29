import torch
import numpy as np
import random

def extract_2d_slices_from_3d(volume, num_slices=10, random_selection=True, axis=0):
    """
    Extract 2D slices from a 3D volume
    
    Args:
        volume: 3D volume tensor of shape [C, D, H, W]
        num_slices: Number of slices to extract
        random_selection: If True, select slices randomly, otherwise equally spaced
        axis: Axis along which to extract slices (0: depth, 1: height, 2: width)
        
    Returns:
        List of 2D slices
    """
    c, d, h, w = volume.shape
    
    if axis == 0:
        # Extracting slices along depth axis
        if random_selection:
            indices = random.sample(range(d), min(num_slices, d))
        else:
            indices = np.linspace(0, d-1, num_slices, dtype=int)
        
        slices = [volume[:, i, :, :] for i in indices]
    
    elif axis == 1:
        # Extract slices along height axis
        if random_selection:
            indices = random.sample(range(h), min(num_slices, h))
        else:
            indices = np.linspace(0, h-1, num_slices, dtype=int)
        
        slices = [volume[:, :, i, :] for i in indices]
    
    else:  # axis == 2
        # Extract slices along width axis
        if random_selection:
            indices = random.sample(range(w), min(num_slices, w))
        else:
            indices = np.linspace(0, w-1, num_slices, dtype=int)
        
        slices = [volume[:, :, :, i] for i in indices]
    
    return slices

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # dummy 3D volume
    volume = torch.rand(1, 64, 64, 64)
    
    # Extracting slices along different axes
    slices_depth = extract_2d_slices_from_3d(volume, num_slices=5, axis=0)
    slices_height = extract_2d_slices_from_3d(volume, num_slices=5, axis=1)
    slices_width = extract_2d_slices_from_3d(volume, num_slices=5, axis=2)
    
    print("2D Slice Extractor:")
    print(f"Original volume shape: {volume.shape}")
    print(f"Number of slices extracted along depth axis: {len(slices_depth)}")
    print(f"Shape of a slice along depth axis: {slices_depth[0].shape}")
    print(f"Number of slices extracted along height axis: {len(slices_height)}")
    print(f"Shape of a slice along height axis: {slices_height[0].shape}")
    print(f"Number of slices extracted along width axis: {len(slices_width)}")
    print(f"Shape of a slice along width axis: {slices_width[0].shape}")
    
    print("\n2D Slice Extractor successfully implemented")
