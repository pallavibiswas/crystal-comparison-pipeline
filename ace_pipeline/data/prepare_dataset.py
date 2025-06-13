import numpy as np
import os

label_map = {
    'fcc' : 0,
    'bcc' : 1,
    'hcp' : 2,
    'liquid' : 3
}

X_all = []
y_all = []

data_dir = "data"

for phase, label in label_map.items():
    fname = f'X_{phase}.dat'
    path = os.path.join(data_dir, fname)
    print(f"loading {path}...")

    X = np.genfromtxt(path, skip_header=1)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X = X[~np.isnan(X).any(axis=1)]

    y = np.full((X.shape[0],), label)

    X_all.append(X)
    y_all.append(y)

max_cols = max(X.shape[1] for X in X_all)
X_all = [np.pad(X, ((0, 0), (0, max_cols - X.shape[1])), constant_values=np.nan) for X in X_all]

X_all = np.vstack(X_all)
y_all = np.concatenate(y_all)

np.savetxt('data/X_all.dat', X_all, fmt = '%.6e')
np.savetxt('data/y_all.dat', y_all, fmt = '%d')
print("Data Combined Processing done.")
