from numpy import *
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
import joblib

import sys
from read_functions import read

X = read('data/X_all.dat')
y = read('data/y_all.dat')

X = X[:340000]
y = y[:340000]

X = SimpleImputer(strategy="mean").fit_transform(X)

NN = make_pipeline(
    StandardScaler(), 
    MLPClassifier(hidden_layer_sizes=(20,), early_stopping=True, learning_rate_init=0.005, verbose=1)) 
NN.fit(X,y)

joblib.dump(NN,'model/neural_network.pkl')

mlp = NN.named_steps['mlpclassifier']
with open("model/mlp_summary.txt", "w") as f:
    f.write("MLPClassifier Summary\n\n")
    f.write(f"Hidden layers: {mlp.hidden_layer_sizes}\n")
    f.write(f"Learning rate init: {mlp.learning_rate_init}\n")
    f.write(f"Early stopping: {mlp.early_stopping}\n")
    f.write(f"Number of iterations: {mlp.n_iter_}\n")
    f.write(f"Final loss: {mlp.loss_:.6f}\n")
    f.write(f"Validation score: {mlp.best_validation_score_:.6f}\n")