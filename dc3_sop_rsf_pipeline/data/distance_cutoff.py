import numpy as np
from read_functions import read
from constants import y_lattices

X = np.loadtxt('data/X_files/X_all.dat')
y = np.loadtxt('data/y_all.dat')

X0 = np.loadtxt('data/perfect_lattices/X.dat')

print("Unique labels in y:", np.unique(y))
y = y.astype(int)

cutoff = []

for i, lattice in zip(range(len(X0)), y_lattices):
    mask = y == lattice
    if lattice == 3:
        continue
    d = np.linalg.norm(X[mask] - X0[i], axis=1)
    cutoff.append(np.percentile(d, 99))

cutoff = np.array(cutoff)
np.savetxt('data/distance_cutoff_vs_lattice.dat', cutoff, fmt='%.6f')