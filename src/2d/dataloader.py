import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os 
from pathlib import Path

from pretext_2d import rotate_2dimages, rplify, jigsawify, exemplar_preprocess
from torch.utils.data import random_split
from torchvision.transforms.functional import to_pil_image, to_tensor
import torch

#custom dataset class to load the training 
class Retinal2dDataset(Dataset):
    #@param takes the prediciton (csv file)
    #@param image_dir takes the images
    #@param transform, transforms each images to the specifies dimension
    def __init__(self, preds_file, image_dir, transform=None, task='rotate', is_training=True):
        self.preds_file = preds_file
        self.image_dir = image_dir
        self.transform = transform
        
        # Load the CSV file
        self.df = pd.read_csv(preds_file)
        
        self.image_paths = self.df['id_code'].values + ".png"
        self.labels = self.df['diagnosis'].values
        self.task = task


        self.is_training = is_training
        if task == "jigsaw":
            self.num_patches = 4
            self.jitter = 20
            self.permutation = [list(np.random.permutation(self.num_patches**2))]
        
        self.images = []
        self.rotated_labels = []

        self.rpl_pairs = []
        self.rpl_labels = []

        self.preprocess()

    def __len__(self):
        return len(self.image_paths)


    def preprocess(self):
        for ind, image_name in enumerate(self.image_paths):
            img_name = os.path.join(self.image_dir, image_name)
            # image = Image.open(img_name)
            try:
                image = Image.open(img_name)
                image.load()  # <-- Force loading immediately to catch errors
                if self.transform:
                    image = self.transform(image)
                self.images.append(image)
            except Exception as e:
                print(f"Warning: Failed to load image {img_name}. Error: {e}")
            
    #     return img, label
    def __getitem__(self, idx):
        img = self.images[idx]
        label = None

        if self.task == "rotate":
            img = to_pil_image(img)
            img, label = rotate_2dimages(img)
            img = to_tensor(img)

        if self.task == "rpl":
            img, label = rplify(img)

        if self.task == "jigsaw":
            img, label = jigsawify(img, self.is_training, self.num_patches, self.jitter, self.permutation)
            # img = torch.from_numpy(img)
            label = torch.from_numpy(label)
        
        if self.task == "exe":
            #lablel is a tuple of positive and negative values
            label = exemplar_preprocess(img)


        #we want the actual labels for the finetune task
        if self.task == "finetune":
            label = self.labels[idx]

        

        return img, label


def load_2dimages(batch_size = 32, train_split = .95, task = "rotate"):

    # Only use transform for rotate and finetune
    transform = None
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image_dataset = Retinal2dDataset(
        preds_file= str(Path(__file__).parent.parent.parent / "dataset/2d/train.csv"), 
        image_dir=str(Path(__file__).parent.parent.parent / "dataset/2d/train_images"), 
        transform=transform,
        task=task
    )

    if task == "finetune":
        return image_dataset

    train_size = int(train_split * len(image_dataset))
    val_size = len(image_dataset) - train_size

        
    train_dataset, val_dataset = random_split(image_dataset, [train_size, val_size])
    val_dataset.dataset.is_training = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    permuation = None
    if task == "jigsaw":
        permuation = image_dataset.permutation


    return train_loader, val_loader, permuation


