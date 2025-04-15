import random

import numpy as np
import skimage.transform as skTrans
import torch

# Operates on a single input
def crop(data: np.ndarray, normalize: bool = True, threshold: float = 0.05) -> np.ndarray:
    if normalize:
        data = (data - data.min()) / (data.max() - data.min())

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

    return data[sx:ex,sy:ey,sz:ez]


# Operates on a single input-output pair
def xy_preprocess(x: np.ndarray, y: np.ndarray, resolution=(128,128,128)) -> tuple[np.ndarray, np.ndarray]:
    # Resize with interpolation
    out_x = skTrans.resize(x, resolution, order=1, preserve_range=True)
    out_y = skTrans.resize(y, resolution, order=1, preserve_range=True)

    # Add channel dimension (grayscale)
    out_x = np.expand_dims(out_x, axis=0)

    # Normalize
    out_x = (out_x - out_x.min()) / (out_x.max() - out_x.min())

    return out_x, out_y


# Operates on a single input
def x_preprocess(x: np.ndarray, resolution=(128,128,128)) -> np.ndarray:
    # Crop to bounding box
    cropped_x = crop(x)

    # Resize with interpolation
    out_x = skTrans.resize(cropped_x, resolution, order=1, preserve_range=True)

    # Add channel dimension (grayscale)
    out_x = np.expand_dims(out_x, axis=0)

    # Normalize
    out_x = (out_x - out_x.min()) / (out_x.max() - out_x.min())

    return out_x


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
    if len(data.shape) == 5:
        transformed = [rotate(x) for x in data]
        x_rot = torch.stack([x for x,_ in transformed])
        y_rot = torch.stack([r for _,r in transformed])
    else:
        x_rot, y_rot = rotate(data)
    return x_rot.float(), y_rot.long()

def rpl_preprocess(data, grid_size=3, patch_size=(32, 32, 32)):
    
    def get_patch(volume, center, size):
        slices = tuple(slice(c - s // 2, c + s // 2) for c, s in zip(center, size))
        return volume[slices]

    def rpl_single(x_tensor):
        x_np = x_tensor.squeeze().numpy()
        vol_shape = x_np.shape
        step = [vol_shape[i] // grid_size for i in range(3)]

        # random jitter (up to ±step//4 in each direction)
        jitter_range = [s // 4 for s in step]

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
    return x_rpl.float(), y_rpl.long()
