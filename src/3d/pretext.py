import os
import random

import numpy as np
import skimage.transform as skTrans
import torch


PERMUTATION_FILE = os.path.join(os.environ.get("VIRTUAL_ENV", "."), "..", "src", "permutations", "permutations_100_27.npy")
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


# Operates on a single input or multiple inputs
def rpl_preprocess(data, grid_size=3, patch_size=(32, 32, 32)):
    
    def get_patch(volume, center, size):
        slices = tuple(slice(c - s // 2, c + s // 2) for c, s in zip(center, size))
        return volume[slices]

    def rpl_single(x_tensor):
        x_np = x_tensor.squeeze().numpy()
        vol_shape = x_np.shape
        step = [vol_shape[i] // grid_size for i in range(3)]

        # random jitter (up to ±step//4 in each direction)
        jitter_range = [s // 4 for s in step] # maybe change to jitter_range = [3, 3, 3]??? 3 voxels matches the paper exactly 

        patches = []
        centers = []
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    center = [i * step[0] + step[0] // 2,
                              j * step[1] + step[1] // 2,
                              k * step[2] + step[2] // 2]
                    jittered_center = [np.clip(c + np.random.randint(-j_range, j_range+1), size//2, vol_shape[idx]-size//2)
                                       for idx, (c, j_range, size) in enumerate(zip(center, jitter_range, patch_size))]
                    patch = get_patch(x_np, jittered_center, patch_size)
                    patches.append(torch.tensor(patch, dtype=torch.float32))
                    centers.append((i, j, k))

        center_index = len(patches) // 2
        query_index = random.choice([i for i in range(len(patches)) if i != center_index])
        xc = patches[center_index].unsqueeze(0)  # 1x32x32x32
        xq = patches[query_index].unsqueeze(0)

        relative_position = query_index if query_index < center_index else query_index - 1
        return (torch.stack([xc, xq], dim=0), torch.tensor(relative_position))

    # batch or single
    if data.ndim == 5:
        transformed = [rpl_single(x) for x in data]
        x_rpl = torch.stack([torch.cat([item[0][0], item[0][1]], dim=0) for item in transformed])
        y_rpl = torch.stack([item[1] for item in transformed])
    else:
        x_pair, y = rpl_single(data)
        x_rpl = torch.cat([x_pair[0], x_pair[1]], dim=0).unsqueeze(0)
        y_rpl = y
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

