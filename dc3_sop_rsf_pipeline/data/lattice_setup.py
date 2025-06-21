from numpy import *
import joblib

import sys
from constants import lattices, n_lattices, y_lattices, N_feat
from read_functions import read


# Compute the average feature vector for the perfect lattice.
X = zeros((len(lattices),N_feat))

for i in range(n_lattices):
  X_tmp = read('data/X_files/X_%s.dat' % lattices[i])
  print(f"{lattices[i]} shape: {X_tmp.shape}")
  X[i] = X_tmp.mean(axis=0)

# Scale and save data.
scaler = joblib.load('data/scaler.pkl')
X = scaler.transform(X)
savetxt('data/perfect_lattices/X.dat', X, fmt='%.8e')
savetxt('data/perfect_lattices/y.dat', y_lattices, fmt='%d')