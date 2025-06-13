from numpy import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

import sys
from read_functions import read

################################################################################
# Input parameters and setup.                                                  #
################################################################################

f = int(sys.argv[1])/10

# Load data set.
X = read('data/X_all.dat')
y = read('data/y_all.dat')
idx = random.choice(arange(len(y)), int(f*len(y)), replace=False)
X = X[idx]
y = y[idx]

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

################################################################################
# Compute.                                                                     #
################################################################################

cv_results = cross_validate(MLPClassifier(early_stopping=True, hidden_layer_sizes=[100,100,100], learning_rate_init=0.005), X, y, cv=5, n_jobs=-1)
mean = cv_results['test_score'].mean()
std = cv_results['test_score'].std()
savetxt('train/learning_curve/learning_curve_%.1f.dat' % f, array([mean,std]), header=' mean | std ', fmt='%.4f')

################################################################################