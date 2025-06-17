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
alpha_all = []

data_dir = "data"

for phase, label in label_map.items():
    X_path = os.path.join(data_dir, f'X_{phase}.dat')
    alpha_path = os.path.join(data_dir, f'alpha_{phase}.dat')

    print(f"loading {X_path} and {alpha_path}...")

    X = np.loadtxt(X_path)
    alpha = np.loadtxt(alpha_path)

    if len(alpha.shape) == 2:
        alpha = alpha.squeeze

    y = np.full((X.shape[0],), label)

    X_all.append(X)
    y_all.append(y)
    alpha_all.append(alpha)

X_all = np.vstack(X_all)
y_all = np.concatenate(y_all)
alpha_all = np.concatenate(alpha_all)

np.savetxt('data/X_all.dat', X_all, fmt = '%.6e')
np.savetxt('data/y_all.dat', y_all, fmt = '%d')
np.savetxt('data/alpha_all.dat', alpha_all, fmt = '%.6f')

print("Data Combined Processing done.")
