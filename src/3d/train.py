import sys
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from loader import PancreasDataset, PancreasPretextDataset
from metrics import weighted_dice_loss
from model import MulticlassClassifier, SkipStripper, UNet3dEncoder, UNet3dDecoder, UNet3d


#@def trains the model
#@param train_dataloader is the training dataset
#@param val_dataloader is the validation dataset
#@param num_epochs is how many epochs we want to train for
#@param model is the model we are training
#@param optimizer is the optimizer we are using
#@param criterion is the loss
#@param task is the pretext task name (or finetune)
#@param device is the device to use
def train(train_dataloader, val_dataloader, num_epochs, model, optimizer, criterion, task, device, fold=None):
    model.train()
    
    #final valdiation score
    val_score = 0
    
    for epoch in range(num_epochs):
        running_loss = 0

        all_outputs = []
        all_labels = []

        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            if task == "rotate":
                labels = torch.argmax(labels, dim=1)
                labels = labels.squeeze()
            loss = criterion(outputs, labels)
            
            
            all_labels = all_labels + labels.cpu().tolist()
            all_outputs = all_outputs + torch.argmax(outputs.detach(), dim=1).cpu().tolist()

            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        running_loss = running_loss / len(train_dataloader)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}')
        # metrics(all_outputs, all_labels, task, "train")
        

        val_loss, val_score = validate(model, val_dataloader, criterion, epoch, num_epochs, device, task)
        if (epoch % 50 == 0  or epoch == num_epochs -1) and epoch != 0:
            save_checkpoint(model, running_loss, val_loss, epoch, optimizer, task, fold)
            print(f"Checkpoint saved at epoch {epoch}")

        
    if task == "finetune":
        return val_score


def save_checkpoint(model, train_loss, val_loss, epoch, optimizer, task,fold=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 
        'train_loss': train_loss, 
        'val_loss': val_loss 
    }
    if fold is not None:
        filepath = "/home/caleb/school/deep_learning/self-supervised-medical/src/2d/model_ckpt/" + task + "/" + "checkpoint" + str(epoch) + "_fold" + str(fold) + ".pth"
    else:
        filepath = "/home/caleb/school/deep_learning/self-supervised-medical/src/2d/model_ckpt/" + task + "/" + "checkpoint" + str(epoch) + ".pth"
    torch.save(checkpoint, filepath)


def validate(model, val_dataloader, criterion, epoch, num_epochs, device, task):
    model.eval()
    running_loss = 0
    
    all_outputs = []
    all_labels = []
    
    for inputs, labels in val_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        
        if task == "rotate":
            labels = torch.argmax(labels, dim=1)
            labels = labels.squeeze()
        loss = criterion(outputs, labels)

        all_labels = all_labels + labels.cpu().tolist()
        all_outputs = all_outputs + torch.argmax(outputs.detach(), dim=1).cpu().tolist() 

        running_loss += loss.item()
    
    running_loss = running_loss / len(val_dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {running_loss}')
    # score = metrics(all_outputs, all_labels, task, "val", epoch, num_epochs)
    
    return running_loss, score



def main():
    task = input("What Task are we solving for (rotate, finetune): ")
    num_epochs = int(input("How many epochs we trying to do (paper recomends 1000 for training for 2D tasks): "))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: " + str(device))

    if task == "rotate":
        encoder = UNet3dEncoder(1)
        classifier = MulticlassClassifier(4**3*1024, 10)
        model = torch.nn.Sequential(encoder, SkipStripper(), classifier)
        lr = 0.001
        criterion = nn.CrossEntropyLoss()
    elif task == "finetune":
        model = UNet3d(UNet3dEncoder(1), UNet3dDecoder(3))
        lr = 0.00001
        criterion = weighted_dice_loss
    else:
        print(f"Unknown task: {task}")
        sys.exit(-1)
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train(train_dataloader, val_dataloader, num_epochs, model, optimizer, criterion, task, device)
    

if __name__ == "__main__":
    main()

