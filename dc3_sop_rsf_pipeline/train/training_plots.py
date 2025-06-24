import matplotlib.pyplot as plt
from numpy import *

################################################################################
# Plot neurons and learning rate.                                              #
################################################################################

# Setup
N = [5,10,20,50,100] # Hidden layers size.
epsilon = [0.0001, 0.001, 0.002, 0.005, 0.010] # Learning rate init.
acc = zeros((5,5))
err = zeros((5,5))
for i in range(5):
  for j in range(5):
    data = loadtxt('train/neurons_learning/neurons_%d_%d.dat' % (i,j), skiprows=1)
    acc[i,j] = data[0]
    err[i,j] = data[1]

# Start figure.
fig = plt.figure()
ax  = fig.add_axes([0.15, 0.15, 0.80, 0.80])

# Plot.
for n in range(5):
  ax.plot(epsilon, acc[:,n], 'C%d-' % n, lw=2, label='(%d,%d,%d)' % (N[n],N[n],N[n]))
  ax.fill_between(epsilon, acc[:,n]-err[:,n], acc[:,n]+err[:,n], alpha=0.2, color='C%d' % n, lw=0)
 
# Add details.
ax.set_xlabel(r'Learning rate initialization')
ax.set_ylabel(r'Accuracy')
ax.set_ylim(acc.min() - 0.01, acc.max() + 0.01)
ax.set_xlim(0,0.010)
ax.legend(fontsize=12,ncol=1, title='Neural network size')

# Save figure.
fig.savefig("train/training_plots/neurons_and_learning.png")
plt.close()

################################################################################
# Plot number of hidden layers.                                                #
################################################################################

# Setup
N = [1,2,3,4,5] # Hidden layers size.
acc = zeros(5)
err = zeros(5)
for n in range(len(N)):
    data = loadtxt('train/hidden_layers/hidden_layers_%d.dat' % (n+1), skiprows=1)
    acc[n] = data[0]
    err[n] = data[1]


# Start figure.
fig = plt.figure()
ax  = fig.add_axes([0.15, 0.15, 0.80, 0.80])

# Plot.
ax.plot(N, acc, 'C0-', lw=3)
ax.fill_between(N, acc-err, acc+err, alpha=0.2, color='C0', lw=0)
 
# Add details.
ax.set_xlabel(r'Number of hidden layers')
ax.set_ylabel(r'Accuracy')
ax.set_ylim(acc.min() - 0.01, acc.max() + 0.01)
ax.set_xlim(1,5)

# Save figure.
fig.savefig("train/training_plots/hidden_layers.png")
plt.close()

################################################################################
# Plot learning curve.                                                         #
################################################################################

# Setup
f = arange(1,11)/10
N = f * 414000 * 0.8 # Size of training set.
acc = zeros(len(N))
err = zeros(len(N))
for n in range(len(N)):
  data = loadtxt('train/learning_curve/learning_curve_%.1f.dat' % f[n], skiprows=1)
  acc[n] = data[0]
  err[n] = data[1]


# Start figure.
fig = plt.figure()
ax  = fig.add_axes([0.15, 0.15, 0.80, 0.80])

# Plot.
ax.plot(N, acc, 'C1-', lw=2)
ax.fill_between(N, acc-err, acc+err, alpha=0.2, color='C1', lw=0)
 
# Add details.
ax.set_xlabel(r'Training set size')
ax.set_ylabel(r'Accuracy')
ax.set_ylim(acc.min() - 0.01, acc.max() + 0.01)
ax.set_xlim(0,400000)

# Save figure.
fig.savefig("train/training_plots/learning_curve.png")
plt.close()

################################################################################
# Plot grid search.                                                             #
################################################################################

# Setup
neurons = [5, 10, 20, 50, 100]
layers = [1, 2, 3, 4, 5]
learning_rates = [0.0001, 0.001, 0.002, 0.005, 0.01]

neurons_idx = {n: i for i,n in enumerate(neurons)}
layers_idx = {l: j for j,l in enumerate(layers)}

for lr in learning_rates:
  acc = zeros((len(neurons), len(layers)))
  for n in neurons:
    for l in layers:
      fname = f"train/hyperparameters/grid_{n}n_{l}l_{lr}lr.dat"
      data = loadtxt(fname, skiprows=1)
      acc[neurons_idx[n], layers_idx[l]] = data[0]

  # Start figure and plot.
  fig, ax = plt.subplots()
  im  = ax.imshow(acc, cmap="viridis", origin='lower', aspect='auto')
  ax.set_xticks(range(len(layers)))
  ax.set_yticks(range(len(neurons)))

  # Add details.
  ax.set_xticklabels(layers)
  ax.set_yticklabels(neurons)
  ax.set_xlabel('Number of Hidden Layers')
  ax.set_ylabel('Neurons per Layer')
  ax.set_title(f'Accuracy for learning rate {lr}')

  for i in range(len(neurons)):
    for j in range(len(layers)):
      text = ax.text(j, i, f"{acc[i, j]:.2f}", ha="center", va="center", color="w")

  # Save figure.
  fig.colorbar(im)
  fig.tight_layout
  fig.savefig(f"train/training_plots/gridsearch_{lr}.png")
  plt.close()

################################################################################