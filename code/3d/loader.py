import math
import os
import sys

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from pretext import preprocess


# Pancreas dataset path constants
PANCREAS_PATH = os.path.join(os.environ.get("VIRTUAL_ENV", "."), "..", "data", "Task07_Pancreas")
PANCREAS_IMAGES_TR = os.path.join(PANCREAS_PATH, "imagesTr")
PANCREAS_LABELS_TR = os.path.join(PANCREAS_PATH, "labelsTr")
PANCREAS_IMAGES_PATH = os.path.join(PANCREAS_PATH, "preprocessed", "train")
PANCREAS_LABELS_PATH = os.path.join(PANCREAS_PATH, "preprocessed", "train_labels")


# Create directories if they do not exist
if not os.path.exists(PANCREAS_PATH):
    print("Please download the pancreas dataset and place 'Task07_Pancreas' in the project root directory.")
    sys.exit(-1)
if not os.path.exists(PANCREAS_IMAGES_PATH):
    os.makedirs(PANCREAS_IMAGES_PATH, exist_ok=True)
if not os.path.exists(PANCREAS_LABELS_PATH):
    os.makedirs(PANCREAS_LABELS_PATH, exist_ok=True)


def bar(percent, length=20) -> str:
    full = math.floor(percent*length)
    empty = length - full
    text = str(math.floor(percent*100))+"%"
    return "[" + "#"*full + " "*empty + "] " + text


# Create preprocessed datasets if they do not exist
if len(os.listdir(PANCREAS_IMAGES_PATH)) == 0 or len(os.listdir(PANCREAS_LABELS_PATH)) == 0:
    print("Could not find preprocessed pancreas dataset. Generating...")
    filenames = [x for x in os.listdir(PANCREAS_IMAGES_TR) if not x.startswith(".")]
    for i, filename in enumerate(filenames):
        percent = i / len(filenames)
        sys.stdout.write("\r"+bar(percent))
        sys.stdout.flush()

        x_file = os.path.join(PANCREAS_IMAGES_TR, filename)
        y_file = os.path.join(PANCREAS_LABELS_TR, filename)
        x = nib.load(x_file).get_fdata() # pyright: ignore
        y = nib.load(y_file).get_fdata() # pyright: ignore
        x, y = preprocess(x, y)

        filename = filename.split(".")[0] + ".npy"
        x_file = os.path.join(PANCREAS_IMAGES_PATH, filename)
        y_file = os.path.join(PANCREAS_LABELS_PATH, filename)
        np.save(x_file, x)
        np.save(y_file, y)
    sys.stdout.write("\r"+" "*50+"\r")
    sys.stdout.flush()
    print("Done!")


# Pancreas dataset file constants
PANCREAS_IMAGES = sorted([os.path.join(PANCREAS_IMAGES_PATH, x) for x in os.listdir(PANCREAS_IMAGES_PATH)])
PANCREAS_LABELS = sorted([os.path.join(PANCREAS_LABELS_PATH, x) for x in os.listdir(PANCREAS_LABELS_PATH)])


class PancreasPretextDataset(Dataset):
    def __init__(self):
        self.x = np.stack([np.load(x) for x in PANCREAS_IMAGES])
        self.x = torch.from_numpy(self.x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


class PancreasDataset(Dataset):
    def __init__(self):
        self.x = np.stack([np.load(x) for x in PANCREAS_IMAGES])
        self.x = torch.from_numpy(self.x)

        self.y = np.rint(np.stack([np.load(y) for y in PANCREAS_LABELS])).astype(np.int32)
        self.C = np.max(self.y) + 1
        self.y = torch.from_numpy(np.eye(self.C)[self.y].transpose(0, 4, 1, 2, 3)) # pyright: ignore

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

