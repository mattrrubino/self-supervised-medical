import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from loader import PancreasDataset
from metrics import weighted_dice_loss
from model import create_unet3d
from train import train


def run_finetune_experiment(filename, percent_train, percent_val=0.05, wu_epochs=25, num_epochs=400, batch_size=4, model_checkpoint=None):
    assert num_epochs >= 1, "Total number of epochs must be greater than or equal to one."
    assert wu_epochs >= 0, "Total number of warmup epochs must be greater than or equal to zero."
    assert num_epochs >= wu_epochs, "Total number of epochs must be greater than or equal to the number of warmup epochs."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator().manual_seed(42)

    _, _, model = create_unet3d()
    # TODO: Load model from checkpoint if exists

    dataset = PancreasDataset()
    train_dataset, val_dataset = random_split(dataset, [1-percent_val, percent_val], generator)
    train_dataset, _ = random_split(train_dataset, [percent_train, 1-percent_train], generator)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-7)
    criterion = weighted_dice_loss
    train(train_dataloader, val_dataloader, wu_epochs, num_epochs, model, optimizer, criterion, device, filename)


def run_experiments():
    for percent in [5, 10, 25, 50, 100]:
        filename = f"baseline_pancreas_{percent}.json"
        percent_train = percent/100.0
        run_finetune_experiment(filename, percent_train)


if __name__ == "__main__":
    run_experiments()

