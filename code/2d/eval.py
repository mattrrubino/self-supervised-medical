import torch
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from train2D import train
import numpy as np
import matplotlib.pyplot as plt

# Example 1D array


#plot the mean_kappa value
def plot_kappa(kappas, training_percent):
    
    training_percent = training_percent * 100
    # Plot
    plt.plot(training_percent, kappas)
    plt.title('Valaidation Kappa Score')
    plt.xlabel('Percentage of lablled images')
    plt.ylabel('Average validation kappa score')
    plt.grid(True)
    plt.show()



        

