import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from loader import PancreasDataset, PancreasPretextDataset
from metrics import weighted_dice_loss, weighted_dice_per_class
from model import create_unet3d


#@def trains the model
#@param train_dataloader is the training dataset
#@param val_dataloader is the validation dataset
#@param num_epochs is how many epochs we want to train for
#@param model is the model we are training
#@param optimizer is the optimizer we are using
#@param criterion is the loss
#@param task is the pretext task name (or finetune)
#@param device is the device to use
def train(train_dataloader, val_dataloader, num_epochs, model, optimizer, criterion, device, fold=None):
    model.train()

    # Freeze the encoder
    # for param in model.encoder.parameters():
        # param.requires_grad = False
    
    for epoch in range(num_epochs):
        # Unfreeze the encoder
        if epoch == 25:
            for param in model.encoder.parameters():
                param.requires_grad = True

        running_loss = 0
        background_dice = 0
        pancreas_dice = 0
        tumor_dice = 0

        for i, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            torch.nn.utils.clip_grad_value_(model.parameters(), 1)
            optimizer.step()

            l = loss.item()
            bg = weighted_dice_per_class(preds, y, 0).item()
            pan = weighted_dice_per_class(preds, y, 1).item()
            tum = weighted_dice_per_class(preds, y, 2).item()

            running_loss += l
            background_dice += bg
            pancreas_dice += pan
            tumor_dice += tum
            print(f"{i+1}/{len(train_dataloader)} - loss: {l} - bg: {bg} - pan: {pan} - tum: {tum}")

        running_loss /= len(train_dataloader)
        background_dice /= len(train_dataloader)
        pancreas_dice /= len(train_dataloader)
        tumor_dice /= len(train_dataloader)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss}, Background Dice: {background_dice}, Pancreas Dice: {pancreas_dice}, Tumor Dice: {tumor_dice}')
        validate(model, val_dataloader, criterion, epoch, num_epochs, device)


def save_checkpoint(model, train_loss, val_loss, epoch, optimizer, task,fold=None):
    return # TODO
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


def validate(model, val_dataloader, criterion, epoch, num_epochs, device):
    model.eval()
    running_loss = 0
    
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    
    running_loss /= len(val_dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {running_loss}')
    model.train()


def main():
    task = input("What Task are we solving for (rotate, finetune): ")
    num_epochs = int(input("How many epochs we trying to do (paper recommends 1000 for training for 2D tasks): "))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: " + str(device))

    if task == "rotate":
        pass # TODO
    elif task == "finetune":
        _, _, model = create_unet3d()
        dataset = PancreasDataset()
        train_size = int(len(dataset)*0.95)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        # lr = 0.00001
        lr = 0.001
        criterion = weighted_dice_loss
    else:
        print(f"Unknown task: {task}")
        sys.exit(-1)
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    train(train_dataloader, val_dataloader, num_epochs, model, optimizer, criterion, device)
    

if __name__ == "__main__":
    main()

