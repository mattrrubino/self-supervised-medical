import random

import numpy as np
import skimage.transform as skTrans


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


def preprocess(x: np.ndarray, y: np.ndarray, resolution=(128,128,128)) -> tuple[np.ndarray, np.ndarray]:
    assert len(x) == len(y),"Must have an equal number of inputs and outputs"

    # Crop to bounding box
    cropped_x = [crop(entry) for entry in x]
    cropped_y = [label[sx:ex,sy:ey,sz:ez] for label,(_,(sx,ex,sy,ey,sz,ez)) in zip(y, cropped_x)]

    # Resize with interpolation
    out_x = np.array([skTrans.resize(entry, resolution, order=1, preserve_range=True) for entry,_ in cropped_x])
    out_y = np.array([skTrans.resize(entry, resolution, order=1, preserve_range=True) for entry in cropped_y])

    # Add channel dimension (grayscale)
    out_x = np.expand_dims(out_x, axis=1)
    out_y = np.expand_dims(out_y, axis=1)

    return out_x, out_y


def rotation_preprocess(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = x.shape[0]
    y_rot = np.zeros((b, 10))

    x_rot = []
    for i, data in enumerate(x):
        r = random.randrange(10)

        if r == 0:
            x_rot.append(data)
        elif r == 1:
            x_rot.append(np.transpose(np.flip(data, 2), (0, 2, 1, 3)))
        elif r == 2:
            x_rot.append(np.flip(data, (1, 2)))
        elif r == 3:
            x_rot.append(np.flip(np.transpose(data, (0, 2, 1, 3)), 2))
        elif r == 4:
            x_rot.append(np.transpose(np.flip(data, 2), (0, 1, 3, 2)))
        elif r == 5:
            x_rot.append(np.flip(data, (2, 3)))
        elif r == 6:
            x_rot.append(np.flip(np.transpose(data, (0, 1, 3, 2)), 2))
        elif r == 7:
            x_rot.append(np.transpose(np.flip(data, 1), (0, 3, 2, 1)))
        elif r == 8:
            x_rot.append(np.flip(data, (1, 3)))
        elif r == 9:
            x_rot.append(np.flip(np.transpose(data, (0, 3, 2, 1)), 1))
        y_rot[i,r] = 1

    return np.array(x_rot), y_rot

