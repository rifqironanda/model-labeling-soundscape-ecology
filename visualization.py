import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_pca_clusters(X, labels, outliers, save_path=None):
    # Reduce to 2D for visualization
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X)

    plt.figure(figsize=(8, 6))

    for i in range(len(X_2d)):
        if outliers[i] == -1:
            plt.scatter(X_2d[i, 0], X_2d[i, 1], marker='x')  # outlier
        else:
            plt.scatter(X_2d[i, 0], X_2d[i, 1])  # normal

    plt.title("PCA Cluster Visualization")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()