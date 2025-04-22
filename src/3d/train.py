import json
import os
import sys

import torch

from metrics import weighted_dice, weighted_dice_per_class


RESULTS_PATH = os.path.join(os.environ.get("VIRTUAL_ENV", "."), "..", "results")
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH, exist_ok=True)
EPSILON = 1e-4


def batch_text(epoch, i, n, l, m):
    metric_text = " - ".join(f"{k} {v:08.7f}" for k, v in sorted(m.items()))
    return f"[EPOCH {epoch}] BATCH {i:02}/{n}: - loss {l:08.7f} - {metric_text}"


def epoch_text(epoch, prefix, l, m):
    metric_text = " - ".join(f"{k} {v:08.7f}" for k, v in sorted(m.items()))
    return f"[EPOCH {epoch}] {prefix}: - loss {l:08.7f} - {metric_text}"


def compute_metrics(preds, y, metrics):
    op = {
        "dice": weighted_dice,
        "dice_0": lambda a, b: weighted_dice_per_class(a, b, 0),
        "dice_1": lambda a, b: weighted_dice_per_class(a, b, 1),
        "dice_2": lambda a, b: weighted_dice_per_class(a, b, 2),
        "accuracy": lambda a, b: (torch.argmax(a, dim=-1) == b).float().mean(),
    }
    return {metric: op[metric](preds, y).item() for metric in metrics}


def average_metrics(ml):
    return {key: sum([m[key] for m in ml])/len(ml) for key in ml[0]}


def train(train_dataloader, val_dataloader, wu_epochs, num_epochs, model, optimizer, criterion, metrics, device, json_file=None, weight_file=None):
    model.to(device)
    model.train()

    # Freeze the encoder
    encoder = model.encoder if hasattr(model, "encoder") else model[0]
    for param in encoder.parameters():
        param.requires_grad = False
    
    json_data = []
    min_validation_loss = float("inf")
    for epoch in range(num_epochs):
        # Unfreeze the encoder
        if epoch == wu_epochs:
            for param in encoder.parameters():
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
            m = compute_metrics(preds, y, metrics)
            train_loss += l
            mt.append(m)
            sys.stdout.write("\r"+batch_text(epoch+1, i, len(train_dataloader), l, m))
            sys.stdout.flush()

        train_loss /= len(train_dataloader)
        mt = average_metrics(mt)
        validation_loss, mv = validate(model, val_dataloader, criterion, metrics, device)

        sys.stdout.write("\r"+" "*120+"\r")
        sys.stdout.flush()
        print(epoch_text(epoch+1, "TRAIN", train_loss, mt))
        print(epoch_text(epoch+1, "VALID", validation_loss, mv))

        if json_file is not None: 
            data = {
                "train_loss": train_loss,
                **{f"train_{key}": value for key, value in mt.items()},
                "validation_loss": validation_loss,
                **{f"validation_{key}": value for key, value in mv.items()},
            }
            json_data.append(data)
            with open(os.path.join(RESULTS_PATH, json_file), "w") as f:
                json.dump(json_data, f)

        if weight_file is not None and validation_loss - min_validation_loss < EPSILON:
            min_validation_loss = validation_loss
            torch.save(encoder.state_dict(), os.path.join(RESULTS_PATH, weight_file))


def validate(model, val_dataloader, criterion, metrics, device):
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
            mv.append(compute_metrics(outputs, labels, metrics))
    
    validation_loss /= len(val_dataloader)
    model.train()
    return validation_loss, average_metrics(mv)

