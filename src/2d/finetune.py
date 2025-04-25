from dataloader import load_2dimages
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from train2D import train
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
import numpy as np
from eval import plot_kappa
import random

#we need to reset the model after each % of dataset training
def reset_model_weights(pre_task = "rotate", device="cuda"):
    model = models.densenet121(weights=None)
    if pre_task == "rotate":
        checkpoint = torch.load("/home/caleb/school/deep_learning/self-supervised-medical/src/2d/model_ckpt/rotate/checkpoint25.pth")

        #only used for helping with shape requirments while loading
        model.classifier = torch.nn.Linear(model.classifier.in_features, 4)
        
        #load everything except the pretrained classifier
        model.load_state_dict(checkpoint["model_state_dict"])

        model.classifier = torch.nn.Linear(model.classifier.in_features, 5)

    if pre_task == "rpl":
        checkpoint = torch.load("/Users/aspensmith/Desktop/self-supervised-medical/src/2d/model_ckpt/rpl/checkpoint5.pth")

        model.classifier = torch.nn.Linear(model.classifier.in_features, 8)  # 8 relative positions
        model.load_state_dict(checkpoint["model_state_dict"])
        model.classifier = torch.nn.Linear(model.classifier.in_features, 5)


    if pre_task == "jigsaw":
        print("loading jigsaw checkpoint")
        checkpoint = torch.load("./model_ckpt/jigsaw/checkpoint25.pth")
        
        ###CHANGE THIS IF YOU CHANGE THE PURMUATION
        model.classifier = torch.nn.Linear(model.classifier.in_features, 16)
        
        #load everything except the pretrained classifier
        model.load_state_dict(checkpoint["model_state_dict"])

        model.classifier = torch.nn.Linear(model.classifier.in_features, 5)


    model.to(device)

    optimzer = optim.Adam(model.classifier.parameters(), lr=0.00005)

    criterion = nn.CrossEntropyLoss()

    for param in model.features.parameters():
        param.requires_grad = False
    
    return model, optimzer, criterion
    

def main():
    pre_task = input("What task are we finetuning for (rotate/rpl/jigsaw): ")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: " + str(device))
    
    #we only want to finetune the classifier, not the densenet features
    model, optimzer, criterion = reset_model_weights(pre_task, device)
    task = "finetune"
    
    print("preprocessing images, this might take a moment ...")
    dataset = load_2dimages(task=task)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    print("preprocessing done, begining training ...")

    
    training_percent = np.array([0.05, 0.10, 0.25, 0.50, 1.00])
    all_kappa_scores = np.zeros(len(training_percent))
    for i in range(0, len(training_percent)):
        print("Data percent: " + str(training_percent[i] * 100) + "%")
        print('--------------------------------')
        
        if(i != 0):
            #we need to reset the model weights after testing on the previous training percent
            model, optimzer, criterion  = reset_model_weights(pre_task)
            
        mean_kappa_scores = 0
        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            
            #reset the model for each fold
            if fold != 0:
                model, optimzer, criterion  = reset_model_weights(pre_task)
            
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
                kappa_score = train(train_dataloader, val_dataloader, 20, model, optimzer, criterion, task, device, fold, training_percent[i])
            else:
                kappa_score = train(train_dataloader, val_dataloader, 20, model, optimzer, criterion, task, device)
            mean_kappa_scores += kappa_score
    

        mean_kappa_scores = mean_kappa_scores / 5
        all_kappa_scores[i] = mean_kappa_scores
    
    np.save("kappa_scores_" + pre_task + ".npy", all_kappa_scores)
    print(all_kappa_scores.shape)
    print(training_percent.shape)
    plot_kappa(all_kappa_scores, training_percent)
    

if __name__ == "__main__":
    main()