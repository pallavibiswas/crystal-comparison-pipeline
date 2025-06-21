import numpy as np
import joblib
from read_functions import read
from sklearn.impute import SimpleImputer

X = read('data/X_all.dat')
y = read('data/y_all.dat')

split_point = int(0.8 * len(X))

X_test = X[split_point:]
y_test = y[split_point:]

X_test = SimpleImputer(strategy="mean").fit_transform(X)

NN = joblib.load('model/neural_network.pkl')

y_pred = NN.predict(X_test)

np.savetxt('model/y_pred.dat', y_pred, fmt='%d')