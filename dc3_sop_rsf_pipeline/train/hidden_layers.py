from numpy import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

import sys
from read_functions import read

################################################################################
# Input parameters and setup.                                                  #
################################################################################

n = int(sys.argv[1])
hidden_layer_sizes = n*[100]

# Load data set.
X = read('data/X_files/X_all.dat')
y = read('data/y_all.dat')

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

################################################################################
# Compute.                                                                     #
################################################################################

cv_results = cross_validate(MLPClassifier(early_stopping=True, hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=0.005), X, y, cv=5, n_jobs=-1)
mean = cv_results['test_score'].mean()
std = cv_results['test_score'].std()
savetxt('train/hidden_layers/hidden_layers_%d.dat' % n, array([mean,std]), header=' mean | std ', fmt='%.4f')

################################################################################