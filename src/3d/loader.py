import math
import os
import sys

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from metrics import weighted_dice_loss
from model import create_classification_head, create_unet3d
from pretext import x_preprocess, xy_preprocess, rotation_preprocess, rpl_preprocess


# Pancreas dataset constants
PANCREAS_PATH = os.path.join(os.environ.get("VIRTUAL_ENV", "."), "..", "..", "self-supervised-3d-tasks", "Task07_Pancreas")
PANCREAS_IMAGES_PATH = os.path.join(PANCREAS_PATH, "images_resized_128_bbox_labeled", "train")
PANCREAS_LABELS_PATH = os.path.join(PANCREAS_PATH, "images_resized_128_bbox_labeled", "train_labels")
PANCREAS_IMAGES = sorted([os.path.join(PANCREAS_IMAGES_PATH, x) for x in os.listdir(PANCREAS_IMAGES_PATH) if not x.startswith(".")])
PANCREAS_LABELS = sorted([os.path.join(PANCREAS_LABELS_PATH, x) for x in os.listdir(PANCREAS_LABELS_PATH) if not x.startswith(".")])

# Pancreas dataset cache
if False:
    PANCREAS_IMAGES_PRE = PANCREAS_IMAGES_TR
    PANCREAS_IMAGES_PRE_CACHE = os.path.join(PANCREAS_PATH, ".imagesPre.npy")
    PANCREAS_IMAGES_DOWN = PANCREAS_IMAGES_TR
    PANCREAS_IMAGES_DOWN_CACHE = os.path.join(PANCREAS_PATH, ".imagesDown.npy")
    PANCREAS_LABELS_DOWN = PANCREAS_LABELS_TR
    PANCREAS_LABELS_DOWN_CACHE = os.path.join(PANCREAS_PATH, ".labelsDown.npy")


def bar(percent, length=20) -> str:
    full = math.floor(percent*length)
    empty = length - full
    text = str(math.floor(percent*100))+"%"
    return "[" + "#"*full + " "*empty + "] " + text


# Cache the preprocessed datasets on disk
if False:
    if not os.path.exists(PANCREAS_IMAGES_PRE_CACHE):
        print("Could not find pretext pancreas data on disk. Generating...")
        x_out = []
        for i, x_file in enumerate(PANCREAS_IMAGES_PRE):
            percent = i / len(PANCREAS_IMAGES_PRE)
            sys.stdout.write("\r"+bar(percent))
            sys.stdout.flush()
            x = nib.load(x_file).get_fdata() # pyright: ignore
            x = x_preprocess(x)
            x_out.append(x)
        sys.stdout.write("\r"+" "*50+"\r")
        sys.stdout.flush()
        print("Saving...")
        np.save(PANCREAS_IMAGES_PRE_CACHE, np.stack(x_out))
    if not os.path.exists(PANCREAS_IMAGES_DOWN_CACHE) or not os.path.exists(PANCREAS_LABELS_DOWN_CACHE):
        print("Could not find downstream pancreas data on disk. Generating...")
        x_out = []
        y_out = []
        for i, (x_file, y_file) in enumerate(zip(PANCREAS_IMAGES_DOWN, PANCREAS_LABELS_DOWN)):
            percent = i / len(PANCREAS_IMAGES_DOWN)
            sys.stdout.write("\r"+bar(percent))
            sys.stdout.flush()
            x = nib.load(x_file).get_fdata() # pyright: ignore
            y = nib.load(y_file).get_fdata() # pyright: ignore
            x, y = xy_preprocess(x, y)
            x_out.append(x)
            y_out.append(y)
        sys.stdout.write("\r"+" "*50+"\r")
        sys.stdout.flush()
        print("Saving...")
        np.save(PANCREAS_IMAGES_DOWN_CACHE, np.stack(x_out))
        np.save(PANCREAS_LABELS_DOWN_CACHE, np.stack(y_out))


class PancreasPretextDataset(Dataset):
    def __init__(self, pretext_preprocess):
        self.pretext_preprocess = pretext_preprocess
        self.x = torch.from_numpy(np.load(PANCREAS_IMAGES_PRE_CACHE)).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.pretext_preprocess(self.x[idx])


class PancreasDataset(Dataset):
    def __init__(self):
        self.x = np.stack([np.load(x).transpose(3, 0, 1, 2) for x in PANCREAS_IMAGES])
        self.x = torch.from_numpy(self.x)
        print("x:", self.x.min(), self.x.max())

        self.y = np.rint(np.stack([np.load(y) for y in PANCREAS_LABELS])).astype(np.int32)
        self.C = np.max(self.y) + 1
        self.y = torch.from_numpy(np.squeeze(np.eye(self.C)[self.y], axis=-2).transpose(0, 4, 1, 2, 3)) # pyright: ignore
        print("y:", self.y.shape)

        # self.x = [torch.from_numpy(np.load(x).transpose(3, 0, 1, 2)) for x in PANCREAS_IMAGES]
        # self.y = [torch.from_numpy(np.load(y).transpose(3, 0, 1, 2)) for y in PANCREAS_LABELS]
        # self.x = torch.from_numpy(np.load(PANCREAS_IMAGES_DOWN_CACHE)).float()
        # self.y = torch.from_numpy(np.load(PANCREAS_LABELS_DOWN_CACHE)).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def run_rotation():
    encoder, _, model = create_unet3d()
    classifier = create_classification_head(encoder, 10) # 10 rotations

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    classifier = classifier.to(device)

    # Pretext 
    dataset = PancreasPretextDataset(rotation_preprocess)
    loss = torch.nn.CrossEntropyLoss()
    x, y = dataset[:2]
    x = x.to(device)
    y = y.to(device)
    preds = classifier(x)
    print(f"Rotation Loss: {loss(preds, y)}")

    # Finetune
    dataset = PancreasDataset()
    loss = weighted_dice_loss
    x, y = dataset[:2]
    x = x.to(device)
    y = y.to(device)
    preds = model(x)
    print(f"Segmentation Loss: {loss(preds, y)}")


def run_rpl():
    encoder, _, model = create_unet3d()
    classifier = create_classification_head(encoder, 26) # 26 relative position

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    classifier = classifier.to(device)

    # pretext 
    dataset = PancreasPretextDataset(rpl_preprocess)
    loss = torch.nn.CrossEntropyLoss()
    x, y = dataset[:2]
    x = x.to(device)
    y = y.to(device)
    preds = classifier(x)
    print(f"RPL Loss: {loss(preds, y)}")

    # finetune
    dataset = PancreasDataset()
    loss = weighted_dice_loss
    x, y = dataset[:2]
    x = x.to(device)
    y = y.to(device)
    preds = model(x)
    print(f"Segmentation Loss: {loss(preds, y)}") 


if __name__ == "__main__":
    run_rotation()
    # run_rpl()

