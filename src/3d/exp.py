import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from loader import PancreasDataset, PancreasPretextDataset
from metrics import weighted_dice_loss
from model import create_unet3d, create_multiclass_classifier, create_multipatch_classifier, create_multipatch_embedder
from pretext import exemplar_preprocess, jigsaw_preprocess, rotation_preprocess, rpl_preprocess, PERMUTATIONS
from train import train, RESULTS_PATH


PERCENTS = [5, 10, 25, 50, 100]


def run_finetune_experiment(json_file, percent_train, percent_val=0.05, wu_epochs=25, num_epochs=400, batch_size=4, weight_file=None):
    assert num_epochs >= 1, "Total number of epochs must be greater than or equal to one."
    assert wu_epochs >= 0, "Total number of warmup epochs must be greater than or equal to zero."
    assert num_epochs >= wu_epochs, "Total number of epochs must be greater than or equal to the number of warmup epochs."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator().manual_seed(42)

    encoder, _, model = create_unet3d()
    if weight_file is not None:
        encoder.load_state_dict(torch.load(os.path.join(RESULTS_PATH, weight_file)))

    dataset = PancreasDataset()
    train_dataset, val_dataset = random_split(dataset, [1-percent_val, percent_val], generator)
    train_dataset, _ = random_split(train_dataset, [percent_train, 1-percent_train], generator)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    criterion = weighted_dice_loss
    metrics = ["dice", "dice_0", "dice_1", "dice_2"]
    train(train_dataloader, val_dataloader, wu_epochs, num_epochs, model, optimizer, criterion, metrics, device, json_file)


def run_pretext_experiment(json_file, weight_file, pretext_preprocess, create_classifier, criterion, n_classes=None, patches=None, percent_val=0.05, num_epochs=1000, batch_size=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator().manual_seed(42)

    encoder, _, _ = create_unet3d()
    classifier = create_classifier(encoder, n_classes, patches)

    dataset = PancreasPretextDataset()
    collate = lambda data: pretext_preprocess(torch.stack(data))
    train_dataset, val_dataset = random_split(dataset, [1-percent_val, percent_val], generator)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    lr = 1e-3
    optimizer = optim.Adam(classifier.parameters(), lr=lr, eps=1e-7)
    metrics = ["accuracy"]
    train(train_dataloader, val_dataloader, 0, num_epochs, classifier, optimizer, criterion, metrics, device, json_file, weight_file)


def run_baseline_experiments():
    for percent in PERCENTS:
        json_file = f"baseline_pancreas_{percent}.json"
        percent_train = percent/100.0
        run_finetune_experiment(json_file, percent_train)


def run_rotation_experiments():
    json_file = "rotation_pancreas.json"
    weight_file = "rotation_pancreas.pth"
    criterion = torch.nn.CrossEntropyLoss()
    run_pretext_experiment(json_file, weight_file, rotation_preprocess, create_multiclass_classifier, criterion, n_classes=10)

    for percent in PERCENTS:
        json_file = f"rotation_pancreas_{percent}.json"
        percent_train = percent/100.0
        run_finetune_experiment(json_file, percent_train, weight_file=weight_file)


def run_rpl_experiments():
    json_file = "rpl_pancreas.json"
    weight_file = "rpl_pancreas.pth"
    criterion = torch.nn.CrossEntropyLoss()
    run_pretext_experiment(json_file, weight_file, rpl_preprocess, create_multipatch_classifier, criterion, n_classes=26, patches=2)

    for percent in PERCENTS:
        json_file = f"rpl_pancreas_{percent}.json"
        percent_train = percent/100.0
        run_finetune_experiment(json_file, percent_train, weight_file=weight_file)


def run_jigsaw_experiments():
    json_file = "jigsaw_pancreas.json"
    weight_file = "jigsaw_pancreas.pth"
    criterion = torch.nn.CrossEntropyLoss()
    run_pretext_experiment(json_file, weight_file, jigsaw_preprocess, create_multipatch_classifier, criterion, n_classes=len(PERMUTATIONS), patches=27)

    for percent in PERCENTS:
        json_file = f"jigsaw_pancreas_{percent}.json"
        percent_train = percent/100.0
        run_finetune_experiment(json_file, percent_train, weight_file=weight_file)


def run_exemplar_experiments():
    json_file = "exemplar_pancreas.json"
    weight_file = "exemplar_pancreas.pth"
    criterion = torch.nn.TripletMarginLoss()
    run_pretext_experiment(json_file, weight_file, exemplar_preprocess, create_multipatch_embedder, criterion, n_classes=1024, patches=3)

    for percent in PERCENTS:
        json_file = f"exemplar_pancreas_{percent}.json"
        percent_train = percent/100.0
        run_finetune_experiment(json_file, percent_train, weight_file=weight_file)


if __name__ == "__main__":
    pass
    # run_baseline_experiments()
    # run_rotation_experiments()
    # run_rpl_experiments()
    #run_jigsaw_experiments()
    #run_exemplar_experiments()

