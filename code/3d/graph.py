import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from train import RESULTS_PATH


TASKS = ["baseline", "jigsaw", "rotation", "rpl", "exemplar"]


def smooth(y, count=10):
    return np.arange(0, len(y), count), [np.mean(y[i*count:(i+1)*count]) for i in range(len(y)//count)]


def create_train_graph():
    results = [x for x in os.listdir(RESULTS_PATH) if "baseline_pancreas" in x]
    x = sorted([int(re.match(r"baseline_pancreas_(\d+).json", x).group(1)) for x in results]) # pyright: ignore

    plt.figure(figsize=(6, 4))
    for task in TASKS:
        y = []
        for percent in x:
            filepath = os.path.join(RESULTS_PATH, f"{task}_pancreas_{percent}.json")
            with open(filepath) as f:
                data = json.load(f)
            dice = sum([x["validation_dice"] for x in data]) / len(data)
            y.append(dice)
        plt.plot(x, y, label=task, marker="o")

    plt.xticks(x)
    plt.grid()
    plt.xlabel("Percentage of Labeled Images", fontweight="bold")
    plt.ylabel("Average Dice Score", fontweight="bold")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_PATH, "train_pancreas.png"))


def create_epoch_graph():
    plt.figure(figsize=(6, 4))
    for task in TASKS:
        filepath = os.path.join(RESULTS_PATH, f"{task}_pancreas_100.json")
        with open(filepath) as f:
            data = json.load(f)
        x, y = smooth([d["validation_dice"] for d in data])
        plt.plot(x, y, label=task, marker="o")

    plt.grid()
    plt.xlabel("Epochs", fontweight="bold")
    plt.ylabel("Average Dice Score", fontweight="bold")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_PATH, "epoch_pancreas.png"))


if __name__ == "__main__":
    create_train_graph()
    create_epoch_graph()

