from numpy import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from read_functions import read

neurons = [5, 10, 20, 50, 100]
layers = [1, 2, 3, 4, 5]
learning_rates = [0.0001, 0.001, 0.002, 0.005, 0.01]

out_dir = 'train/hyperparameters'

X = read('data/X_files/X_all.dat')
y = read('data/y_all.dat')

X = StandardScaler().fit_transform(X)

best_score = -1
best_combo = ()

for n in neurons:
    for l in layers:
        for lr in learning_rates:
            hl = [n]*l
            mlp_clf = MLPClassifier(
                early_stopping=True,
                hidden_layer_sizes=hl,
                learning_rate_init=lr
            )
            cv_results = cross_validate(mlp_clf, X, y, cv=5, n_jobs=-1)
            mean = cv_results['test_score'].mean()
            std = cv_results['test_score'].std()
            
            fname = f"{out_dir}/grid_{n}n_{l}l_{lr}lr.dat"
            savetxt(fname, array([mean,std]), header=' mean | std ', fmt='%.4f')

            print(f"Saved: {fname}")

            if mean > best_score:
                best_score = mean
                best_combo = (n, l, lr, std)

print("\nBest Hyperparameter Combination: ")
print(f"Neurons per Layer: {best_combo[0]}")
print(f"No. of Hidden Layers: {best_combo[1]}")
print(f"Learning Rate: {best_combo[2]}")
print(f"Mean Accuracy: {best_score:.4f}")
print(f"Standard Deviation: {best_combo[3]:.4f}")

