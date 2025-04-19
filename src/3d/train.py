import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from loader import PancreasDataset, PancreasPretextDataset
from metrics import weighted_dice, weighted_dice_loss, weighted_dice_per_class
from model import create_unet3d


def batch_text(epoch, i, n, l, m):
    metric_text = " - ".join(f"{k} {v:08.7f}" for k, v in sorted(m.items()))
    return f"[EPOCH {epoch}] BATCH {i:02}/{n}: - loss {l:08.7f} - {metric_text}"


def epoch_text(epoch, lt, mt, lv, mv):
    metric_t_text = " - t_".join(f"{k} {v:05.4f}" for k, v in sorted(mt.items()))
    metric_v_text = " - v_".join(f"{k} {v:05.4f}" for k, v in sorted(mv.items()))
    return f"[EPOCH {epoch}] - t_loss {lt:05.4f} - t_{metric_t_text} - v_loss {lv:05.4f} - v_{metric_v_text}"


def compute_metrics(preds, y):
    return {
        "dice": weighted_dice(preds, y).item(),
        "dice_0": weighted_dice_per_class(preds, y, 0).item(),
        "dice_1": weighted_dice_per_class(preds, y, 1).item(),
        "dice_2": weighted_dice_per_class(preds, y, 2).item(),
    }


def average_metrics(ml):
    return {key: sum([m[key] for m in ml])/len(ml) for key in ml[0]}


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
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    for epoch in range(num_epochs):
        # Unfreeze the encoder
        if epoch == 25:
            for param in model.encoder.parameters():
                param.requires_grad = True

        train_loss = 0
        mt = []

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
            m = compute_metrics(preds, y)
            train_loss += l
            mt.append(m)
            sys.stdout.write("\r"+batch_text(epoch, i, len(train_dataloader), l, m))
            sys.stdout.flush()

        train_loss /= len(train_dataloader)
        validation_loss, mv = validate(model, val_dataloader, criterion, device)

        sys.stdout.write("\r"+" "*120+"\r")
        sys.stdout.flush()
        print(epoch_text(epoch, train_loss, average_metrics(mt), validation_loss, mv))


def validate(model, val_dataloader, criterion, device):
    model.eval()
    validation_loss = 0
    mv = []
    
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            mv.append(compute_metrics(outputs, labels))
    
    validation_loss /= len(val_dataloader)
    model.train()
    return validation_loss, average_metrics(mv)


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

