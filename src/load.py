import nibabel as nib
import numpy as np


def load_files(files: list[str]) -> np.ndarray:
    imgs = [nib.load(file) for file in files]
    return np.array([img.get_fdata() for img in imgs])


data_x = load_files(["../Task07_Pancreas/imagesTr/pancreas_001.nii.gz"])
data_y = load_files(["../Task07_Pancreas/labelsTr/pancreas_001.nii.gz"])
print(data_x.shape, data_y.shape)

