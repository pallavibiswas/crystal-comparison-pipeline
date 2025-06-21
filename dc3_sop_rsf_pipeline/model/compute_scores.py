from numpy import *

import sys
from dc3 import dc3
from read_functions import read

system = sys.argv[1] # Specify data set.

# Load data set and compute dc3 predictions.
X = read(f'data/X_files/X_{system}.dat')
y_true = read(f'data/y_{system}.dat').astype(int)
alpha = read(f'data/alpha_files/alpha_{system}.dat')

split_point = int(0.8 * len(X))

X_test = X[split_point:]
y_test = y_true[split_point:]
alpha_test = alpha[split_point:]

y_pred = dc3(X_test,alpha_test)

# Save dc3 predictions.
savetxt('model/y_pred_%s.dat' % system, y_pred, fmt='%d')
