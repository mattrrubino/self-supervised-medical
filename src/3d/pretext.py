import os
import random

import numpy as np
import skimage.transform as skTrans
import torch
import torch.nn.functional as F


#PERMUTATION_FILE = os.path.join(os.environ.get("VIRTUAL_ENV", "."), "..", "src", "permutations", "permutations_100_27.npy")
PERMUTATION_FILE = "/home/caleb/school/deep_learning/self-supervised-medical/src/permutations/permutations_100_27.npy"

PERMUTATIONS = np.load(PERMUTATION_FILE)


# Operates on a single input
def norm(data: np.ndarray) -> np.ndarray:
    data = data.astype(np.float32)
    data = (data - data.min()) / (data.max() - data.min())
    return data


# Operates on a single input
def crop(data: np.ndarray, normalize: bool = True, threshold: float = 0.05) -> tuple[np.ndarray, tuple[int,int,int,int,int,int]]:
    if normalize:
        data = norm(data)

    # Compute the 3D bounding box
    sx, ex, sy, ey, sz, ez = 0, 0, 0, 0, 0, 0
    for x in range(data.shape[0]):
        if np.any(data[x, :, :] > threshold):
            sx = x
            break
    for x in range(data.shape[0] - 1, -1, -1):
        if np.any(data[x, :, :] > threshold):
            ex = x
            break
    for y in range(data.shape[1]):
        if np.any(data[:, y, :] > threshold):
            sy = y
            break
    for y in range(data.shape[1] - 1, -1, -1):
        if np.any(data[:, y, :] > threshold):
            ey = y
            break
    for z in range(data.shape[2]):
        if np.any(data[:, :, z] > threshold):
            sz = z
            break
    for z in range(data.shape[2] - 1, -1, -1):
        if np.any(data[:, :, z] > threshold):
            ez = z
            break

    return data[sx:ex,sy:ey,sz:ez], (sx,ex,sy,ey,sz,ez)


# Operates on a single input-output pair
def preprocess(x: np.ndarray, y: np.ndarray, resolution=(128,128,128)) -> tuple[np.ndarray, np.ndarray]:
    # Crop to bounding box
    cropped_x, (sx,ex,sy,ey,sz,ez) = crop(x)
    cropped_y = y[sx:ex,sy:ey,sz:ez]

    # Resize with interpolation
    out_x = skTrans.resize(cropped_x, resolution, order=1, preserve_range=True)
    out_y = skTrans.resize(cropped_y, resolution, order=1, preserve_range=True)

    # Add channel dimension
    out_x = np.expand_dims(out_x, axis=0)

    # Normalize
    out_x = norm(out_x)

    return out_x, out_y


# Operates on a single input or multiple inputs
def rotation_preprocess(data):
    def rotate(x):
        r = random.randrange(10)
        if r == 1:
            x_rot = torch.flip(x, (2,)).permute(0, 2, 1, 3)
        elif r == 2:
            x_rot = torch.flip(x, (1, 2))
        elif r == 3:
            x_rot = torch.flip(x.permute(0, 2, 1, 3), (2,))
        elif r == 4:
            x_rot = torch.flip(x, (2,)).permute(0, 1, 3, 2)
        elif r == 5:
            x_rot = torch.flip(x, (2, 3))
        elif r == 6:
            x_rot = torch.flip(x.permute(0, 1, 3, 2), (2,))
        elif r == 7:
            x_rot = torch.flip(x, (1,)).permute(0, 3, 2, 1)
        elif r == 8:
            x_rot = torch.flip(x, (1, 3))
        elif r == 9:
            x_rot = torch.flip(x.permute(0, 3, 2, 1), (1,))
        else:
            x_rot = x
        return x_rot, torch.tensor(r)
    if data.ndim == 5:
        transformed = [rotate(x) for x in data]
        x_rot = torch.stack([x for x,_ in transformed])
        y_rot = torch.stack([y for _,y in transformed])
    else:
        x_rot, y_rot = rotate(data)
    return x_rot.float(), y_rot.long()


def patchify_3d(volume, grid_size=3, patch_size=(39, 39, 39), jitter=3):
    """
    split a 3D volume into jittered patches and return all patches and their grid positions
    """
    D, H, W = volume.shape
    step_d, step_h, step_w = D // grid_size, H // grid_size, W // grid_size

    patches = []
    centers = []

    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                center = [
                    i * step_d + step_d // 2,
                    j * step_h + step_h // 2,
                    k * step_w + step_w // 2
                ]

                jittered_center = [
                    np.clip(center[0] + np.random.randint(-jitter, jitter + 1), patch_size[0]//2, D - patch_size[0]//2),
                    np.clip(center[1] + np.random.randint(-jitter, jitter + 1), patch_size[1]//2, H - patch_size[1]//2),
                    np.clip(center[2] + np.random.randint(-jitter, jitter + 1), patch_size[2]//2, W - patch_size[2]//2)
                ]

                slices = tuple(slice(c - s // 2, c + s // 2) for c, s in zip(jittered_center, patch_size))
                patch = volume[slices]

                # zero-pad any undersized patch to guarantee [39, 39, 39]
                if patch.shape != patch_size:
                    pad = [(0, max(0, s - patch.shape[i])) for i, s in enumerate(patch_size)]
                    patch = np.pad(patch, pad_width=pad, mode='constant')

                patches.append(torch.tensor(patch, dtype=torch.float32))
                centers.append((i, j, k))

    return patches, centers


def rpl_preprocess(data, grid_size=3, patch_size=(39, 39, 39), jitter=3):
    """
    create a (2, D, H, W) input where the first and second patches come from
    a grid of jittered 3D patches
    
    label is the relative spatial index (0-25)
    """
    def rpl_single(volume_tensor):
        volume = volume_tensor.squeeze().numpy()
        patches, _ = patchify_3d(volume, grid_size=grid_size, patch_size=patch_size, jitter=jitter)

        center_idx = len(patches) // 2
        query_idx = random.choice([i for i in range(len(patches)) if i != center_idx])

        xc = patches[center_idx].unsqueeze(0)  # shape: (1, 39, 39, 39)
        xq = patches[query_idx].unsqueeze(0)

        rel_label = query_idx if query_idx < center_idx else query_idx - 1
        return (torch.cat([xc, xq], dim=0), torch.tensor(rel_label))

    if data.ndim == 5:
        transformed = [rpl_single(x) for x in data]
        x_rpl = torch.stack([x for x,_ in transformed])
        y_rpl = torch.stack([y for _,y in transformed])
    else:
        x_rpl, y_rpl = rpl_single(data)

    return x_rpl.float().transpose(0, 1), y_rpl.long()


# Operates on a single input or multiple inputs
def jigsaw_preprocess(data):
    if data.ndim == 5:
        transformed = [jigsawify(x, True, 3, 3, PERMUTATIONS) for x in data]
        x_jig = torch.stack([torch.from_numpy(x) for x,_ in transformed])
        y_jig = torch.stack([torch.tensor(y) for _,y in transformed])
    else:
        x_jig, y_jig = jigsawify(data, True, 3, 3, PERMUTATIONS)
        x_jig = torch.from_numpy(x_jig)
        y_jig = torch.tensor(y_jig)
    return x_jig.float(), y_jig.long()


def jigsawify(image, is_training, patches_per_side, patch_jitter, permutations):
    overlap_mode = False
    _, h, w, d = image.shape

    patch_overlap = 0
    if patch_jitter < 0: # If we are given a negative patch jitter, we actually want overlapping patches
        patch_overlap = -patch_jitter
        overlap_mode = True        

    h_step = (h - patch_overlap)//patches_per_side
    w_step = (w - patch_overlap)//patches_per_side
    d_step = (d - patch_overlap)//patches_per_side

    patch_height = h_step - patch_jitter
    patch_width = w_step - patch_jitter
    patch_depth = d_step - patch_jitter

    patch_arr = []
    for x in range(patches_per_side):
        for y in range(patches_per_side):
            for z in range(patches_per_side):
                patch = img_crop(image, x*h_step, y*w_step, z*d_step, h_step+patch_overlap, w_step+patch_overlap, d_step+patch_overlap)

                if not overlap_mode: # do jitter
                    x_patch_start = 0
                    y_patch_start = 0
                    z_patch_start = 0
                    if is_training:
                        x_patch_start = random.randint(0, patch_jitter)
                        y_patch_start = random.randint(0, patch_jitter)
                        z_patch_start = random.randint(0, patch_jitter)
                    else:
                        x_patch_start = patch_jitter // 2
                        y_patch_start = patch_jitter // 2
                        z_patch_start = patch_jitter // 2
                    patch = img_crop(patch, x_patch_start, y_patch_start, z_patch_start, patch_height, patch_width, patch_depth)
                patch_arr.append(patch)

    y = random.randint(0, len(permutations)-1) # permutation label
    return np.array(patch_arr)[permutations[y]], y


def img_crop(image, x, y, z, h, w, d):
    return image[:, x:(x+h), y:(y+w), z:(z+d)]


# Operates on multiple inputs
def exemplar_preprocess(data):
    assert data.ndim == 5, "Negative examples undefined for unbatched data"

    positive = torch.stack([apply_3d_transformations(x) for x in data])
    negative = positive.flip(dims=[0])

    x = torch.cat((data, positive.float(), negative.float()), dim=1)
    return x, None


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
            _, d, h, w = transformed.shape
            
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

