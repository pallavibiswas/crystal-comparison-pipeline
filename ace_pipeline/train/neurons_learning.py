from numpy import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import sys
from read_functions import read

################################################################################
# Input parameters and setup.                                                  #
################################################################################

i = int(sys.argv[1])
j = int(sys.argv[2])

# Auxiliary arrays.
N = [5,10,20,50,100] # Hidden layers size.
epsilon = [0.0001, 0.001, 0.002, 0.005, 0.010] # Learning rate init.
N = N[i]
epsilon = epsilon[j]

# Load data set.
X = read('data/X_all.dat')
y = read('data/y_all.dat')

#Missing values handling
X = SimpleImputer(strategy="mean").fit_transform(X)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

################################################################################
# Compute.                                                                     #
################################################################################

cv_results = cross_validate(MLPClassifier(early_stopping=True, hidden_layer_sizes=[N,N,N], learning_rate_init=epsilon), X, y, cv=5, n_jobs=-1)
mean = cv_results['test_score'].mean()
std = cv_results['test_score'].std()
savetxt('train/neurons_learning/neurons_%d_%d.dat' % (i,j), array([mean,std]), header=' mean | std ', fmt='%.4f')

################################################################################