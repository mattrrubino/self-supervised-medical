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

