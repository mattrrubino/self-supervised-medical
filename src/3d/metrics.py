import numpy as np


def weighted_dice(preds: np.ndarray, y: np.ndarray, smooth=0.00001):
    # Do not include axis for batch or channel dimension
    axis = (2, 3, 4)
    intersection = (y * preds).sum(axis=axis) + smooth / 2
    union = y.sum(axis=axis) + preds.sum(axis=axis) + smooth
    return (2*intersection/union).mean()


def weighted_dice_loss(preds: np.ndarray, y: np.ndarray):
    return 1 - weighted_dice(preds, y)

