import numpy as np
import joblib
from read_functions import read
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

X = read('data/X_all.dat')
y = read('data/y_all.dat')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

NN = joblib.load('model/neural_network.pkl')

y_pred = NN.predict(X_test)

np.savetxt('model/y_pred.dat', y_pred, fmt='%d')