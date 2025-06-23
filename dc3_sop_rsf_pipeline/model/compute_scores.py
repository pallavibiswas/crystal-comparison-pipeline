from numpy import *

import sys
from dc3 import dc3
from read_functions import read
from sklearn.model_selection import train_test_split

system = sys.argv[1] # Specify data set.

# Load data set and compute dc3 predictions.
X = read(f'data/X_files/X_{system}.dat')
y = read(f'data/y_{system}.dat').astype(int)
alpha = read(f'data/alpha_files/alpha_{system}.dat')

X_train, X_test, y_train, y_test, alpha_train, alpha_test = train_test_split(X, y, alpha, test_size=0.2, stratify=y, random_state=42)

y_pred = dc3(X_test,alpha_test)

# Save dc3 predictions.
savetxt('model/y_pred_%s.dat' % system, y_pred, fmt='%d')
