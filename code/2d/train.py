import os
import sys

import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score, cohen_kappa_score

from loader import load_images


TASKS = ["rotate", "rpl", "jigsaw", "exe"]


#@def trains the model
#@param train_dataloader is the training dataset
#@param num_epochs is how many epochs we want to train for
#@param is th emodel we are training
#@param optimizer is the optimizer we are using
#@param criterion is the loss function we are using
#@param task is the task we are solving for
#@param device is the device we are using
#@param fold is the fold we are using for cross validation, used to save finetuning checkpoints
#@param training_percent is the percent of the dataset we are using for finetuning, used to save finetuning checkpoints
def train(train_dataloader, val_dataloader, num_epochs, model, optimizer, criterion, task, device, fold=None, training_percent=None):
    model.train()
    
    #final valdiation score
    val_score = 0
    
    for epoch in range(num_epochs):
        
        #warmup epochs for the finetuning task
        if task == "finetune" and epoch >= 5:
            for param in model.parameters():
                param.requires_grad = True
            
            # Update optimizer to include all params
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
        
        
        running_loss = 0

        all_outputs = []
        all_labels = []

        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            if task != "exe":
                labels = labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            if task == "rotate" or task == "jigsaw":
                labels = torch.argmax(labels, dim=1)
                labels = labels.squeeze()
            elif task == "rpl":
                labels = labels.squeeze().long()
            if task != "exe":    
                loss = criterion(outputs, labels)
                all_labels = all_labels + labels.cpu().tolist()
                all_outputs = all_outputs + torch.argmax(outputs.detach(), dim=1).cpu().tolist()
            else:
                #positive, negative, and achor
                positive = labels[0].to(device)
                negative = labels[1].to(device)
                positive = model(positive)
                negative = model(negative)
                loss = criterion(outputs, positive, negative)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        running_loss = running_loss / len(train_dataloader)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}')
        if task != "exe":
            metrics(all_outputs, all_labels, task, "train")

        val_loss, val_score = validate(model, val_dataloader, criterion, epoch, num_epochs, device, task)

        model.train()
        if (epoch % 25 == 0  or epoch == num_epochs - 1) and epoch != 0:
            save_checkpoint(model, running_loss, val_loss, epoch, optimizer, task, fold, training_percent)
            print(f"Checkpoint saved at epoch {epoch}")

    if task == "finetune":
        return val_score


def metrics(outputs, labels, task, split, epoch=None, num_epochs=None):
    if task in ["rotate", "rpl", "jigsaw"]:
        f1 = f1_score(labels, outputs, average='macro')
        print(f"F1 {split} Score: {f1}")
    if task == "finetune":
        kappa = cohen_kappa_score(labels, outputs, weights='quadratic')
        print(f"Kappa {split} Score: {kappa}")
        if split == "val" and epoch == (num_epochs - 1):
            cross_validation_score = kappa
            return cross_validation_score

    return None    


def save_checkpoint(model, train_loss, val_loss, epoch, optimizer, task,fold=None, training_percent=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 
        'train_loss': train_loss, 
        'val_loss': val_loss 
    }
    if fold is not None:
        filepath = "code/2d/model_ckpt/" + task + "/" + "checkpoint" + str(epoch) + "_fold" + str(fold) + "training_percent_" + str(training_percent) + ".pth"
    else:
        filepath = "code/2d/model_ckpt/" + task + "/" + "checkpoint" + str(epoch) + ".pth"
    
    # create parent directory if it doesn't exist
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)
    torch.save(checkpoint, filepath)


def validate(model, val_dataloader, criterion, epoch, num_epochs, device, task):
    model.eval()
    running_loss = 0
    
    all_outputs = []
    all_labels = []
    
    for inputs, labels in val_dataloader:
        inputs = inputs.to(device)
        if task != "exe":
            labels = labels.to(device)
        outputs = model(inputs)
        
        if task == "rotate" or task == "jigsaw":
            labels = torch.argmax(labels, dim=1)
            labels = labels.squeeze()
        elif task == "rpl":
            labels = labels.squeeze().long()
        
        if task != "exe":
            loss = criterion(outputs, labels)
            all_labels = all_labels + labels.cpu().tolist()
            all_outputs = all_outputs + torch.argmax(outputs.detach(), dim=1).cpu().tolist() 
        else:
            #positive, negative, and achor
            positive = labels[0].to(device)
            negative = labels[1].to(device)
            positive = model(positive)
            negative = model(negative)
            loss = criterion(outputs, positive, negative)

        running_loss += loss.item()
    
    running_loss = running_loss / len(val_dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {running_loss}')
    
    if task != "exe":
        score = metrics(all_outputs, all_labels, task, "val", epoch, num_epochs)
    else:
        score = None
    
    return running_loss, score


def main(num_epochs=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: " + str(device))

    for task in TASKS:
        model = models.densenet121(weights=None)
        print(f"preprocessing images for {task}, this might take a moment ...")
        #permuation is only used for the jigaw task will not be used for the other tasks
        train_dataloader, val_dataloader, permuation = load_images(task=task, batch_size=8)
        print("preprocessing done, beginning training ...")

        if task == "rotate":
            #We are predicting four classes for the densenet rotation
            model.classifier = torch.nn.Linear(model.classifier.in_features, 4)
            criterion = nn.CrossEntropyLoss()
        elif task == "jigsaw":
            model.classifier = torch.nn.Linear(model.classifier.in_features, len(permuation[0]))
            criterion = nn.CrossEntropyLoss()
        elif task == "rpl":
            # first conv layer accepts 2 input channels
            model.features.conv0 = torch.nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.classifier = torch.nn.Linear(model.classifier.in_features, 8)
            criterion = nn.CrossEntropyLoss()
        elif task == "exe":
            model.classifier = torch.nn.Linear(model.classifier.in_features, model.classifier.in_features)
            criterion = nn.TripletMarginLoss(margin=1.0, p=2)
        else:
            print(f"Unknown task: {task}")
            sys.exit(-1)
        
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        train(train_dataloader, val_dataloader, num_epochs, model, optimizer, criterion, task, device)
    
     
if __name__ == "__main__":
    main()

