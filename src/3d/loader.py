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
PANCREAS_PATH = os.path.join(os.environ.get("VIRTUAL_ENV", "."), "..", "Task07_Pancreas")
PANCREAS_IMAGES_TR = [os.path.join(PANCREAS_PATH, "imagesTr", x) for x in os.listdir(os.path.join(PANCREAS_PATH, "imagesTr")) if not x.startswith(".")]
PANCREAS_LABELS_TR = [os.path.join(PANCREAS_PATH, "labelsTr", x) for x in os.listdir(os.path.join(PANCREAS_PATH, "labelsTr")) if not x.startswith(".")]
PANCREAS_IMAGES_TS = [os.path.join(PANCREAS_PATH, "imagesTs", x) for x in os.listdir(os.path.join(PANCREAS_PATH, "imagesTs")) if not x.startswith(".")]

# Pancreas dataset cache
PANCREAS_IMAGES_PRE = PANCREAS_IMAGES_TR + PANCREAS_IMAGES_TS
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
        self.x = torch.from_numpy(np.load(PANCREAS_IMAGES_DOWN_CACHE)).float()
        self.y = torch.from_numpy(np.load(PANCREAS_LABELS_DOWN_CACHE)).long()

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
    # pretext
    encoder_pretext, _, _ = create_unet3d(filters_in=2)  # rpl uses 2-channel input
    classifier = create_classification_head(encoder_pretext, 26, data_dim=32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = classifier.to(device)

    # pretext training
    dataset = PancreasPretextDataset(rpl_preprocess)
    loss_fn = torch.nn.CrossEntropyLoss()
    x, y = dataset[:2]
    x = x.to(device)
    y = y.to(device)
    preds = classifier(x)
    print(f"RPL Loss: {loss_fn(preds, y)}")

    # note: for pretext -> finetune use two encoders with different 
    # input channels but shared weights (excluding conv1)

    # create fresh encoder that accepts real ct (1-channel)
    encoder_finetune, decoder_finetune, model = create_unet3d(filters_in=1)

    # load compatible weights from pretext encoder
    state_dict_pretext = encoder_pretext.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict_pretext.items() if "layers.0.conv1.weight" not in k} # first conv has different in_channels
    encoder_finetune.load_state_dict(filtered_state_dict, strict=False)

    # finetune 
    model = model.to(device)
    dataset = PancreasDataset()
    x, y = dataset[:2]
    x = x.to(device)
    y = y.to(device)
    loss = weighted_dice_loss
    preds = model(x)
    print(f"Segmentation Loss: {loss(preds, y)}")


if __name__ == "__main__":
    run_rotation()
    run_rpl()
