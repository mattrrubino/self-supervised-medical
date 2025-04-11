from dataloader import load_2dimages
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from train2D import train


def main():
    task = input("What task are we finetuning for (rotate): ")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device: " + str(device))
    
    model = models.densenet121(weights=None)
    if(task == "rotate"):
        
        checkpoint = torch.load("/home/caleb/school/deep_learning/self-supervised-medical/src/2d/model_ckpt/rotate/checkpoint19.pth")
        
        #only used for helping with shape requirments while loading
        model.classifier = torch.nn.Linear(model.classifier.in_features, 4)
        
        #load everything except the pretrained classifier
        model.load_state_dict(checkpoint["model_state_dict"])

        model.classifier = torch.nn.Linear(model.classifier.in_features, 5)
        

    task = task + "_finetune"
    model.to(device)

    #we only want to finetune the classifier, not the densenet features
    for param in model.features.parameters():
        param.requires_grad = False

    optimzer = optim.Adam(model.classifier.parameters(), lr=0.00005)

    criterion = nn.CrossEntropyLoss()
    print("preprocessing images, this might take a moment ...")
    train_dataloader, val_dataloader = load_2dimages(task="finetune")
    print("preprocessing done, begining training ...")

    train(train_dataloader, val_dataloader, 20, model, optimzer, criterion, task, device)


    








if __name__ == "__main__":
    main()