import random

import numpy as np
import skimage.transform as skTrans
import torch


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
    if len(data.shape) == 5:
        transformed = [rotate(x) for x in data]
        x_rot = torch.stack([x for x,_ in transformed])
        y_rot = torch.stack([r for _,r in transformed])
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
        x_rpl = torch.stack([x[0] for x in transformed])
        y_rpl = torch.stack([x[1] for x in transformed])
    else:
        x_pair, y = rpl_single(data)
        x_rpl = x_pair.unsqueeze(0)
        y_rpl = y

    return x_rpl.float(), y_rpl.long()
