import nibabel as nib
import numpy as np

from pretext_3d import pretext_preprocess, rotation_preprocess


def load_files(files: list[str]) -> np.ndarray:
    imgs = [nib.load(file) for file in files] # pyright: ignore
    return np.array([img.get_fdata() for img in imgs]) # pyright: ignore


x = load_files(["../Task07_Pancreas/imagesTr/pancreas_001.nii.gz"])
y = load_files(["../Task07_Pancreas/labelsTr/pancreas_001.nii.gz"])

x, y = pretext_preprocess(x, y)
x_rot, y_rot = rotation_preprocess(x)
print(x_rot.shape, y_rot)

