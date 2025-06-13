import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = np.loadtxt('data/X_all.dat')
y = np.loadtxt('data/y_all.dat')

X_std = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

plt.figure(figsize=(10,8))
colors = ['red', 'green', 'yellow', 'blue']
labels = ['fcc', 'bcc', 'hcp', 'liquid']

for i, color in enumerate(colors):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=labels[i], c=color, alpha=0.5, s=15)

plt.title("PCA Projection for SOP & RSF features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/pca_plots/pca_projection_dc3.png")
plt.show()


plt.figure(figsize=(8,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')

plt.title(" Explained Variance Ratio for SOP & RSF features")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Ratio")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/pca_plots/explained_variance_ratio_dc3.png")
plt.show()