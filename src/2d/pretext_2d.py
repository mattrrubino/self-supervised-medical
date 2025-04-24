import numpy as np
import random
from PIL import Image
import torch
import torchvision.transforms as transforms
import math 
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




def rpl_preprocess(image, grid_size=3, patch_size=(32, 32), jitter=5):
    """
    gen a pair of 2D patches (center, query) and the relative position label between them.
    """

    if image.ndim == 3 and image.shape[0] == 1:
        image = image.squeeze(0)

    H, W = image.shape
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

    image = image.convert("L")  # grayscale
    image = np.array(image)  # H x W

    x_pair, label = rpl_preprocess(image, grid_size, patch_size, jitter)  # shape: (2, 32, 32)
    
    x_pair = torch.from_numpy(x_pair)  # shape: (2, 32, 32)
    
    # resize to (2, 224, 224)
    resize_fn = Resize((224, 224))
    x_pair = resize_fn(x_pair)

    return x_pair, label

