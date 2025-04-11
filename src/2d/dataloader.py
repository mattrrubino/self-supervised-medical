import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os 
from pretext_2d import rotate_2dimages
from torch.utils.data import random_split


#custom dataset class to load the training 
class Retinal2dDataset(Dataset):
    #@param takes the prediciton (csv file)
    #@param image_dir takes the images
    #@param transform, transforms each images to the specifies dimension
    def __init__(self, preds_file, image_dir, transform=None, task='rotate'):
        self.preds_file = preds_file
        self.image_dir = image_dir
        self.transform = transform
        
        # Load the CSV file
        self.df = pd.read_csv(preds_file)
        
        
        self.image_paths = self.df['id_code'].values + ".png"
        self.labels = self.df['diagnosis'].values

        self.task = task
        
        self.images = []
        self.rotated_labels = []

        self.preprocess()


    def __len__(self):
        return len(self.image_paths)

    #def preprocess transforms all the images and applies the nesseccary pre-text task
    def preprocess(self):
        for image_name in self.image_paths:
            img_name = os.path.join(self.image_dir, image_name)
            image = Image.open(img_name)
            
            if self.task == "rotate":
                image, rotation_label = rotate_2dimages(image)
                self.rotated_labels.append(rotation_label)
            if self.transform:
                image = self.transform(image)
            
            self.images.append(image)
            
            
            

    def __getitem__(self, idx):
        img = self.images[idx]
        label = None
        if self.task == "rotate":
            label = self.rotated_labels[idx]
        
        #we want the actual labels for the finetune task
        if self.task == "finetune":
            label = self.labels[idx]

        return img, label


#@def function returns a loaded image dataloader of the function
#@param batchsize are the size of the batches
def load_2dimages(batch_size = 32, task = "rotate"):
    #transform the images 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image_dataset = Retinal2dDataset(preds_file="/home/caleb/school/deep_learning/self-supervised-medical/dataset/2d/train.csv", 
        image_dir= "/home/caleb/school/deep_learning/self-supervised-medical/dataset/2d/train_images", transform=transform, task=task)

    train_size = int(0.85 * len(image_dataset))
    val_size = len(image_dataset) - train_size

    
    train_dataset, val_dataset = random_split(image_dataset, [train_size, val_size])

    # Create DataLoaders for both training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, val_loader

