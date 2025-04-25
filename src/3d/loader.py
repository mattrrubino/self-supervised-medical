import math
import os
import sys

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from metrics import weighted_dice_loss
from model import MulticlassClassifier, create_classification_head, rpl_create_classification_head, create_unet3d
from pretext import preprocess, rotation_preprocess, rpl_preprocess


# Pancreas dataset path constants
PANCREAS_PATH = os.path.join(os.environ.get("VIRTUAL_ENV", "."), "..", "Task07_Pancreas")
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
    def __init__(self, pretext_preprocess):
        self.pretext_preprocess = pretext_preprocess
        self.x = np.stack([np.load(x) for x in PANCREAS_IMAGES])
        self.x = torch.from_numpy(self.x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.pretext_preprocess(self.x[idx])


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


if False:
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


if False:
    def run_rpl():
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # pretext
        encoder_pretext, _, _ = create_unet3d(filters_in=2)
        classifier = rpl_create_classification_head(encoder_pretext, 26, input_shape=(2, 39, 39, 39))

        encoder_pretext = encoder_pretext.to(device)
        classifier = classifier.to(device)

        dataset = PancreasPretextDataset(rpl_preprocess)
        loss_fn = torch.nn.CrossEntropyLoss()

        x, y = dataset[:2]
        x = x.to(device)
        y = y.to(device)
        print(f"x.shape (RPL input): {x.shape}")
        preds = classifier(x)
        print(f"RPL Loss: {loss_fn(preds, y)}")

        # finetune
        encoder_finetune, decoder_finetune, model = create_unet3d(filters_in=1)

        # ld pretext weights into finetune encoder (excluding conv1)
        state_dict_pretext = encoder_pretext.state_dict()
        filtered_state_dict = {
            k: v for k, v in state_dict_pretext.items()
            if "layers.0.conv1" not in k  # skip first conv layer (wrong in_channels)
        }
        encoder_finetune.load_state_dict(filtered_state_dict, strict=False)

        model = model.to(device)
        dataset = PancreasDataset()
        loss = weighted_dice_loss

        x, y = dataset[:2]
        x = x.to(device)
        y = y.to(device)
        print(f"x.shape (CT input): {x.shape}")
        preds = model(x)
        print(f"Segmentation Loss: {loss(preds, y)}")


if __name__ == "__main__":
    PancreasDataset()
    # run_rotation()
    # run_rpl()

