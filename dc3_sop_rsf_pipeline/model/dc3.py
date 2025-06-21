from numpy import *
from scipy.linalg import norm
import joblib

import numpy as np
from read_functions import read
from constants import y_lattices

def dc3(X,alpha):
  # Load coherence factor cutoff. 
  alpha_cut = read('data/alpha_cutoff.dat')
  # Load neural network.
  NN = joblib.load('model/neural_network.pkl')
  # Load cutoff distance to perfect lattices.
  d_cut = read('data/distance_cutoff_vs_lattice.dat')
  y = NN.predict(X) # Load neural-network prediction.
  d = compute_distance(X,y) # Compute distance vector to ideal lattice points.

  print("Initial NN predictions:", np.unique(y, return_counts=True))

  # DC3 logics.
  for n in range(len(alpha)):

    label =int(y[n])

    if alpha[n] >= alpha_cut: # If crystalline.
      if label in y_lattices:
        idx = list(y_lattices).index(label)
        if d[n] > d_cut[idx]:
          y[n] = -1
      else:
        y[n] = -1 # Unknown crystal.
        continue 
    else:
      y[n] = 3 # Liquid or amorphous.

  print("After applying DC3 logic:", np.unique(y, return_counts=True))
  return y # Return predicted labels.


# Compute distance from all examples in X to their ideal lattices.
def compute_distance(X,y):
  X0 = read('data/perfect_lattices/X.dat')
  d = zeros(len(y))
  for i in range(len(X0)):
    mask = (y==i)
    d[mask] = norm(X[mask]-X0[i],axis=1)
  return d

