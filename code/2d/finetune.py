import os
import random
import re

import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
import numpy as np

from loader import load_images
from train import train


TASKS = ["base", "rotate", "rpl", "jigsaw", "exe"]
PERCENTS = [5, 10, 25, 50, 100]


def load_latest_checkpoint(task):
    if task == "base":
        return {}
    path = f"code/2d/model_ckpt/{task}"
    checkpoints = os.listdir(path)
    pattern = re.compile(r"(.*?)(\d+)\.pth")
    numbered = []
    for checkpoint in checkpoints:
        match = pattern.search(checkpoint)
        if match:
            numbered.append((int(match.group(2)), checkpoint))
    filename = os.path.join(path, max(numbered, key=lambda x: x[0])[1])
    return torch.load(filename)


#we need to reset the model after each % of dataset training
def reset_model_weights(pre_task = "rotate", device="cuda"):
    model = models.densenet121(weights=None)
    checkpoint = load_latest_checkpoint(pre_task)

    if pre_task == "rotate":
        #only used for helping with shape requirments while loading
        model.classifier = torch.nn.Linear(model.classifier.in_features, 4)
        
        #load everything except the pretrained classifier
        model.load_state_dict(checkpoint["model_state_dict"])

        model.classifier = torch.nn.Linear(model.classifier.in_features, 5)
    elif pre_task == "rpl":
        # FIRST: load the model exactly like RPL pretraining
        model.features.conv0 = torch.nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 8)
        model.load_state_dict(checkpoint["model_state_dict"])

        # SECOND: NOW modify conv0 back to 3 channels for finetuning
        model.features.conv0 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # THIRD: Replace classifier for 5-class finetuning task
        model.classifier = torch.nn.Linear(model.classifier.in_features, 5)
    elif pre_task == "exe":
        #only used for helping with shape requirments while loading
        model.classifier = torch.nn.Linear(model.classifier.in_features, model.classifier.in_features)
        
        #load everything except the pretrained classifier
        model.load_state_dict(checkpoint["model_state_dict"])

        model.classifier = torch.nn.Linear(model.classifier.in_features, 5)
    elif pre_task == "jigsaw":
        ###CHANGE THIS IF YOU CHANGE THE PURMUATION
        model.classifier = torch.nn.Linear(model.classifier.in_features, 16)
        
        #load everything except the pretrained classifier
        model.load_state_dict(checkpoint["model_state_dict"])

        model.classifier = torch.nn.Linear(model.classifier.in_features, 5)
    else:
        print("using baseline")
        model.classifier = torch.nn.Linear(model.classifier.in_features, 5)

    model.to(device)

    optimzer = optim.Adam(model.classifier.parameters(), lr=0.00005)

    criterion = nn.CrossEntropyLoss()

    for param in model.features.parameters():
        param.requires_grad = False
    
    return model, optimzer, criterion
    

def main(num_epochs=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: " + str(device))
    
    for pre_task in TASKS:
        #we only want to finetune the classifier, not the densenet features
        task = "finetune"
        
        print(f"preprocessing images for {pre_task}, this might take a moment ...")
        dataset = load_images(task=task)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        print("preprocessing done, begining training ...")

        training_percent = np.array(PERCENTS) / 100.0
        all_kappa_scores = np.zeros(len(training_percent))
        for i in range(0, len(training_percent)):
            print("Data percent: " + str(training_percent[i] * 100) + "%")
            print('--------------------------------')
                
            mean_kappa_scores = 0
            for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
                #reset the model for each fold
                model, optimizer, criterion  = reset_model_weights(pre_task, device)
                
                print(f'\nFOLD {fold + 1}')
                print('--------------------------------')
                
                #since we are doing a kfold with warmup, we need to freeze the weights
                #back just to be safe
                train_pool = list(train_ids)
        
                # sample only a percent of the training depending on the training_percent
                subset_size = int(training_percent[i] * len(train_pool))
                sampled_train_ids = random.sample(train_pool, subset_size)
                
                train_subsampler = Subset(dataset, sampled_train_ids)
                val_subsampler = Subset(dataset, val_ids)
                
                train_dataloader = DataLoader(train_subsampler, batch_size=32, shuffle=True)
                val_dataloader = DataLoader(val_subsampler, batch_size=32, shuffle=True)
                
                #save for last fold
                if fold == 4:
                    kappa_score = train(train_dataloader, val_dataloader, num_epochs, model, optimizer, criterion, task, device, fold, training_percent[i])
                else:
                    kappa_score = train(train_dataloader, val_dataloader, num_epochs, model, optimizer, criterion, task, device)
                mean_kappa_scores += kappa_score
        
            mean_kappa_scores = mean_kappa_scores / 5
            all_kappa_scores[i] = mean_kappa_scores
        
        np.save("results/kappa_scores_" + pre_task + ".npy", all_kappa_scores)
        print(all_kappa_scores.shape)
        print(training_percent.shape)
    

if __name__ == "__main__":
    main()

