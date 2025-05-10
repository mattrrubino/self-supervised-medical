import numpy as np
import random
from PIL import Image
import torch
import torchvision.transforms as transforms
import math 
import torch.nn.functional as F

# @def rotates a single 2d image and and then returns the classifcation
# the rotation classification for prediction
# @param is a PIL image to be rotated
def rotate_2dimages(image):
    rotation_label = np.random.randint(low =1, high = 5)
    rotation = rotation_label * 90
    image = image.rotate(angle=rotation)

    one_hot = np.zeros((4, 1))
    one_hot[rotation_label -1] = 1

    return image, one_hot
    
def jigsawify(image, is_training, patches_per_side, patch_jitter, permutations):
    # transform = transforms.Compose([
    #     transforms.ToTensor()
    # ])
    # image = transform(image)  # Now it's a Tensor with shape (C, H, W)
    c, h, w = image.shape
    overlap_mode = False

    patch_overlap = 0
    if patch_jitter < 0: # If we are given a negative patch jitter, we actually want overlapping patches
        patch_overlap = -patch_jitter
        overlap_mode = True        

    h_step = (h - patch_overlap)//patches_per_side
    w_step = (w - patch_overlap)//patches_per_side

    patch_height = h_step - patch_jitter
    patch_width = w_step - patch_jitter

    patch_arr = []
    for row in range(patches_per_side):
        for col in range(patches_per_side):
            patch = img_crop(image, row*h_step, col*w_step, h_step+patch_overlap, w_step+patch_overlap)

            if not overlap_mode: # do jitter
                x_patch_start = 0
                y_patch_start = 0
                if is_training:
                    x_patch_start = random.randint(0, patch_jitter)
                    y_patch_start = random.randint(0, patch_jitter)
                else:
                    x_patch_start = patch_jitter // 2
                    y_patch_start = patch_jitter // 2
                patch = img_crop(patch, x_patch_start, y_patch_start, patch_height, patch_width)
            patch_arr.append(patch)

    label = random.randint(0, len(permutations)-1) # permutation label
    y = np.zeros((len(permutations),)) # one hot permutation label
    y[label] = 1

    patch_arr = np.array([patch.numpy() for patch in patch_arr])
    output_arr = patch_arr[permutations[label]]
    return reform_image(output_arr), np.array(y)

def img_crop(image, x, y, h, w):
    return image[:, x:(x+h), y:(y+w)]

def reform_image(patches):
    C, H, W = patches[0].shape

    num_patches = int(math.sqrt(len(patches)))

    # Reshape into grid and concatenate
    rows = []
    for i in range(num_patches):
        row_patches = [patches[i * num_patches + j] for j in range(num_patches)]
        row = np.concatenate(row_patches, axis=2)  # concat along width
        rows.append(row)

    full_image = np.concatenate(rows, axis=1)  # concat rows along height

    return torch.from_numpy(full_image)

def exemplar_preprocess(data):
    positive = apply_2d_transformations(data)
    negative = positive.flip(dims=[0])

    
    return (positive, negative)


#used for exexmplar task
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


def rpl_preprocess(image, grid_size=3, patch_size=(32, 32), jitter=5):
    """
    gen a pair of 2D patches (center, query) and the relative position label between them.
    """

    if image.ndim == 3 and image.shape[0] == 1:
        image = image.squeeze(0)

    H, W = image.shape
    #print(H, W)
    step_h, step_w = H // grid_size, W // grid_size

    centers = []
    for i in range(grid_size):
        for j in range(grid_size):
            center_y = i * step_h + step_h // 2
            center_x = j * step_w + step_w // 2

            jittered_y = np.clip(center_y + np.random.randint(-jitter, jitter+1), patch_size[0]//2, H - patch_size[0]//2)
            jittered_x = np.clip(center_x + np.random.randint(-jitter, jitter+1), patch_size[1]//2, W - patch_size[1]//2)

            centers.append((jittered_y, jittered_x))

    center_index = len(centers) // 2  # center patch index (4 for 3x3 grid)
    query_index = random.choice([i for i in range(len(centers)) if i != center_index])

    def extract_patch(cy, cx):
        y1, y2 = cy - patch_size[0]//2, cy + patch_size[0]//2
        x1, x2 = cx - patch_size[1]//2, cx + patch_size[1]//2
        return image[y1:y2, x1:x2]

    center_patch = extract_patch(*centers[center_index])
    query_patch = extract_patch(*centers[query_index])

    # comp rel label based on grid position difference
    ci_row, ci_col = center_index // grid_size, center_index % grid_size
    qi_row, qi_col = query_index // grid_size, query_index % grid_size

    dy, dx = qi_row - ci_row, qi_col - ci_col
    offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    relative_label = offsets.index((dy, dx))

    # norm and stack as (2, H, W)
    x_pair = np.stack([center_patch, query_patch], axis=0).astype(np.float32)
    x_pair = (x_pair - x_pair.min()) / (x_pair.max() - x_pair.min())

    return x_pair, relative_label

def rplify(image, grid_size=3, patch_size=(32, 32), jitter=5):
    """
    Applies 2D-RPL pretext transformation to a single image.
    Returns:
        torch.Tensor: stacked (2, 224, 224) tensor of patches.
        int: relative position label (0-7)
    """
    from torchvision.transforms import Resize

    # If image is RGB (3 channels), manually convert it to grayscale
    if image.ndim == 3 and image.shape[0] == 3:
        # standard grayscale conversion weights
        image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]

    # print(f"Image shape after squeeze: {image.shape}")

    image = image.numpy()

    x_pair, label = rpl_preprocess(image, grid_size, patch_size, jitter)  # shape: (2, 32, 32)
    
    x_pair = torch.from_numpy(x_pair)  # shape: (2, 32, 32)
    
    # resize to (2, 224, 224)
    resize_fn = Resize((224, 224))
    x_pair = resize_fn(x_pair)

    return x_pair, label

# def test_rplify_on_random_image():
#     # Create a random fake RGB image (3, 224, 224)
#     random_image = torch.rand(3, 224, 224)

#     print(f"Input random image shape: {random_image.shape}")

#     # Try to run rplify
#     x_pair, label = rplify(random_image)

#     print(f"Output x_pair shape: {x_pair.shape}")
#     print(f"Output label: {label}")

# if __name__ == "__main__":
#     test_rplify_on_random_image()

