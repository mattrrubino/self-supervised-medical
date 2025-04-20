import json
import os
import re

import matplotlib.pyplot as plt

from train import RESULTS_PATH


GRAPHS_PATH = os.path.join(os.environ.get("VIRTUAL_ENV", "."), "..", "graphs")
if not os.path.exists(GRAPHS_PATH):
    os.makedirs(GRAPHS_PATH, exist_ok=True)


def create_baseline_graph():
    results = [x for x in os.listdir(RESULTS_PATH) if "baseline_pancreas" in x]
    x = sorted([int(re.match(r"baseline_pancreas_(\d+).json", x).group(1)) for x in results]) # pyright: ignore
    y = []
    for percent in x:
        filepath = os.path.join(RESULTS_PATH, f"baseline_pancreas_{percent}.json")
        with open(filepath) as f:
            data = json.load(f)
        dice = max([x["validation_dice"] for x in data])
        y.append(dice)

    plt.plot(x, y, label="baseline", marker="o")
    plt.xticks(x)
    plt.grid(True)
    plt.xlabel("Percentage of Labeled Images", fontweight="bold")
    plt.ylabel("Average Dice Score", fontweight="bold")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(GRAPHS_PATH, "baseline_pancreas.png"))


if __name__ == "__main__":
    create_baseline_graph()

