from dataloader import load_2dimages
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score, cohen_kappa_score
import numpy as np

#used when finetuing the model


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
        if task == "finetune" and epoch >=5:
            for param in model.parameters():
                param.requires_grad = True
            
            # Update optimizer to include all params
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
        
        
        running_loss = 0

        all_outputs = []
        all_labels = []

        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            if task == "rotate" or task == "jigsaw":
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
        metrics(all_outputs, all_labels, task, "train")

        

        val_loss, val_score = validate(model, val_dataloader, criterion, epoch, num_epochs, device, task)

        model.train()
        if (epoch % 25 == 0  or epoch == num_epochs -1) and epoch != 0:
            save_checkpoint(model, running_loss, val_loss, epoch, optimizer, task, fold, training_percent)
            print(f"Checkpoint saved at epoch {epoch}")

        
    if task == "finetune":
        return val_score


def metrics(outputs, labels, task, split, epoch=None, num_epochs=None):
    if task == "rotate" or task == "jigsaw":
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
        filepath = "/home/caleb/school/deep_learning/self-supervised-medical/src/2d/model_ckpt/" + task + "/" + "checkpoint" + str(epoch) + "_fold" + str(fold) + "training_percent_" + str(training_percent) + ".pth"
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
        
        if task == "rotate" or task == "jigsaw":
            labels = torch.argmax(labels, dim=1)
            labels = labels.squeeze()
        loss = criterion(outputs, labels)

        all_labels = all_labels + labels.cpu().tolist()
        all_outputs = all_outputs + torch.argmax(outputs.detach(), dim=1).cpu().tolist() 

        running_loss += loss.item()
    
    running_loss = running_loss / len(val_dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {running_loss}')
    score = metrics(all_outputs, all_labels, task, "val", epoch, num_epochs)
    
    return running_loss, score



def main():
    
    model = models.densenet121(weights=None)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"


    task = input("What Task are we solving for (rotate, jigsaw): ")
    num_epochs = int(input("How many epochs we trying to do (Reccomend around 15-25 for training for 2D tasks): "))

    print("Using device: " + str(device))
     
    print("preprocessing images, this might take a moment ...")
    #permuation is only used for the jigaw task will not be used for the other tasks
    train_dataloader, val_dataloader, permuation = load_2dimages(task=task)
    print("preprocessing done, begining training ...")

    
    if task == "rotate":
        #We are predicting four classes for the densenet rotation
        model.classifier = torch.nn.Linear(model.classifier.in_features, 4)

    if task == "jigsaw":
        model.classifier = torch.nn.Linear(model.classifier.in_features, len(permuation))

    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
   

    train(train_dataloader, val_dataloader, num_epochs, model, optimizer, criterion, task, device)
    




if __name__ == "__main__":
    main()