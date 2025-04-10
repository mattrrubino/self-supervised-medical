import random

import numpy as np
import skimage.transform as skTrans


# Operates on a single input
def crop(data: np.ndarray, normalize: bool = True, threshold: float = 0.05) -> tuple[np.ndarray, tuple[int,int,int,int,int,int]]:
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

    return data[sx:ex,sy:ey,sz:ez], (sx,ex,sy,ey,sz,ez)


# Operates on a single input-output pair
def xy_preprocess(x: np.ndarray, y: np.ndarray, resolution=(128,128,128)) -> tuple[np.ndarray, np.ndarray]:
    # Crop to bounding box
    cropped_x, (sx,ex,sy,ey,sz,ez) = crop(x)
    cropped_y = y[sx:ex,sy:ey,sz:ez]

    # Resize with interpolation
    out_x = skTrans.resize(cropped_x, resolution, order=1, preserve_range=True)
    out_y = skTrans.resize(cropped_y, resolution, order=1, preserve_range=True)

    # Add channel dimension (grayscale)
    out_x = np.expand_dims(out_x, axis=0)

    return out_x, out_y


# Operates on a single input
def x_preprocess(x: np.ndarray, resolution=(128,128,128)) -> np.ndarray:
    # Crop to bounding box
    cropped_x, _ = crop(x)

    # Resize with interpolation
    out_x = skTrans.resize(cropped_x, resolution, order=1, preserve_range=True)

    # Add channel dimension (grayscale)
    out_x = np.expand_dims(out_x, axis=0)

    return out_x


# Operates on a single input
def rotation_preprocess(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = random.randrange(10)
    y_rot = np.zeros(10)
    y_rot[r] = 1

    if r == 1:
        x_rot = np.transpose(np.flip(x, 2), (0, 2, 1, 3))
    elif r == 2:
        x_rot = np.flip(x, (1, 2))
    elif r == 3:
        x_rot = np.flip(np.transpose(x, (0, 2, 1, 3)), 2)
    elif r == 4:
        x_rot = np.transpose(np.flip(x, 2), (0, 1, 3, 2))
    elif r == 5:
        x_rot = np.flip(x, (2, 3))
    elif r == 6:
        x_rot = np.flip(np.transpose(x, (0, 1, 3, 2)), 2)
    elif r == 7:
        x_rot = np.transpose(np.flip(x, 1), (0, 3, 2, 1))
    elif r == 8:
        x_rot = np.flip(x, (1, 3))
    elif r == 9:
        x_rot = np.flip(np.transpose(x, (0, 3, 2, 1)), 1)
    else:
        x_rot = x

    # TODO: Might need to make these Tensor objects
    # TODO: Make sure they are floats (not doubles)
    return x_rot, y_rot

