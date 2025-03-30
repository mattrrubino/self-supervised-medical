import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os


#custom dataset class to load the training 
class Retinal2dDataset(Dataset):
    #@param takes the prediciton (csv file)
    #@param image_dir takes the images
    #@param transform, transforms each images to the specifies dimension
    def __init__(self, preds_file, image_dir, transform=None):
        self.preds_file = preds_file
        self.image_dir = image_dir
        self.transform = transform
        
        # Load the CSV file
        self.df = pd.read_csv(preds_file)
        
        
        self.image_paths = self.df['id_code'].values + ".png"
        self.labels = self.df['diagnosis'].values

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_paths[idx])
        image = Image.open(img_name)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


#@def function returns a loaded image dataloader of the function
#@param filepath is the filepath of the training/test images
def load_2dimages():
    #transform the images 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image_dataset = Retinal2dDataset(preds_file="/home/caleb/school/deep_learning/self-supervised-medical/dataset/2d/train.csv", 
        image_dir= "/home/caleb/school/deep_learning/self-supervised-medical/dataset/2d/train_images", transform=transform)
    
    
    dataloader = DataLoader(image_dataset, batch_size=32, shuffle=True)

    
    
    for images, _ in dataloader:
        print(images.shape)
        print(images.min(), images.max())

def main():
    load_2dimages()

if __name__ == "__main__":
    main()

