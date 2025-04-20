import json
import os
import sys

import torch

from metrics import weighted_dice, weighted_dice_per_class


RESULTS_PATH = os.path.join(os.environ.get("VIRTUAL_ENV", "."), "..", "results")
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH, exist_ok=True)


def batch_text(epoch, i, n, l, m):
    metric_text = " - ".join(f"{k} {v:08.7f}" for k, v in sorted(m.items()))
    return f"[EPOCH {epoch}] BATCH {i:02}/{n}: - loss {l:08.7f} - {metric_text}"


def epoch_text(epoch, prefix, l, m):
    metric_text = " - ".join(f"{k} {v:08.7f}" for k, v in sorted(m.items()))
    return f"[EPOCH {epoch}] {prefix}: - loss {l:08.7f} - {metric_text}"


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
def train(train_dataloader, val_dataloader, wu_epochs, num_epochs, model, optimizer, criterion, device, filename):
    model.to(device)
    model.train()

    # Freeze the encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    json_data = []
    for epoch in range(num_epochs):
        # Unfreeze the encoder
        if epoch == wu_epochs:
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
        mt = average_metrics(mt)
        validation_loss, mv = validate(model, val_dataloader, criterion, device)

        sys.stdout.write("\r"+" "*120+"\r")
        sys.stdout.flush()
        print(epoch_text(epoch, "TRAIN", train_loss, mt))
        print(epoch_text(epoch, "VALID", validation_loss, mv))

        data = {
            "train_loss": train_loss,
            **{f"train_{key}": value for key, value in mt.items()},
            "validation_loss": validation_loss,
            **{f"validation_{key}": value for key, value in mv.items()},
        }
        json_data.append(data)
        with open(os.path.join(RESULTS_PATH, filename), "w") as f:
            json.dump(json_data, f)


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

