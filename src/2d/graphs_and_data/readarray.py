import numpy as np
import matplotlib.pyplot as plt

rot = np.load("/home/caleb/school/deep_learning/self-supervised-medical/src/2d/graphs_and_data/kappa_scores_rotate.npy")
jigsaw = np.load("/home/caleb/school/deep_learning/self-supervised-medical/src/2d/graphs_and_data/kappa_scores_jigsaw2.npy")

training_percent = [5, 10, 25, 50, 100]

# Plot
plt.plot(training_percent, rot, label='Rotation', marker='o') # Plot the first line with circle markers
plt.plot(training_percent, jigsaw, label='Jigsaw', marker='x')
plt.legend()

plt.title('Valaidation Kappa Score')
plt.xlabel('Percentage of labeled images')
plt.ylabel('Average validation kappa score')
plt.grid(True)
plt.show()


