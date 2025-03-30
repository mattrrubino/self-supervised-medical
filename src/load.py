import nibabel as nib
import numpy as np
import skimage.transform as skTrans


def load_files(files: list[str]) -> np.ndarray:
    imgs = [nib.load(file) for file in files] # pyright: ignore
    return np.array([img.get_fdata() for img in imgs]) # pyright: ignore


def crop(data: np.ndarray, normalize: bool = True, threshold: float = 0.05) -> tuple[np.ndarray, tuple[int,int,int,int,int,int]]:
    if normalize:
        data = (data - data.min()) / (data.max() - data.min())

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


def pretrain_preprocess(x: np.ndarray, y: np.ndarray, resolution=(128,128,128)) -> tuple[list[np.ndarray], list[np.ndarray]]:
    assert len(x) == len(y),"Must have an equal number of inputs and outputs"
    cropped_x = [crop(entry) for entry in x]
    cropped_y = [label[sx:ex,sy:ey,sz:ez] for label,(_,(sx,ex,sy,ey,sz,ez)) in zip(y, cropped_x)]
    print(cropped_x[0][0].shape, cropped_y[0].shape)
    out_x = [skTrans.resize(entry, resolution, order=1, preserve_range=True) for entry,_ in cropped_x]
    out_y = [skTrans.resize(entry, resolution, order=1, preserve_range=True) for entry in cropped_y]
    print(out_x[0].shape, out_y[0].shape)
    return out_x, out_y


x = load_files(["../Task07_Pancreas/imagesTr/pancreas_001.nii.gz"])
y = load_files(["../Task07_Pancreas/labelsTr/pancreas_001.nii.gz"])
x, y = pretrain_preprocess(x, y)

