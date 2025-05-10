import numpy as np
import matplotlib.pyplot as plt


rot = np.load("results/kappa_scores_rotate.npy")
jigsaw = np.load("results/kappa_scores_jigsaw.npy")
base = np.load("results/kappa_scores_base.npy")
rpl = np.load("results/kappa_scores_rpl.npy")
exe = np.load("results/kappa_scores_exe.npy")

training_percent = [5, 10, 25, 50, 100]

# Plot
plt.figure(figsize=(6, 4))
plt.plot(training_percent, base, label='baseline',  marker='o')
plt.plot(training_percent, jigsaw, label='jigsaw', marker ='o') # Plot the first line with circle markers
plt.plot(training_percent, rot, label='rotation', marker ='o') # Plot the first line with circle markers
plt.plot(training_percent, rpl, label='rpl', marker ='o') # Plot the second line with square markers
plt.plot(training_percent, exe, label='exemplar',  marker ='o', ) # Plot the second line with square markers
plt.legend()
plt.ylim(bottom=.5)

plt.xticks(training_percent)
plt.grid()
plt.xlabel('Percentage of labeled images', fontweight="bold")
plt.ylabel('Average validation kappa score', fontweight="bold")
plt.grid(True)
plt.savefig("results/train_fundus.png")


