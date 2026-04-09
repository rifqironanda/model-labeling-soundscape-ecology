import os
import numpy as np

from utils import load_dataset
from preprocessing import scale_features
from modeling import (
    apply_pca,
    clustering_kmeans,
    detect_outliers,
    evaluate_clustering  # NEW
)

from visualization import plot_pca_clusters  # NEW

DATA_DIR = r"C:\skripsi\ta\biokustik-downloader\biokustik-downloader\output\unsupervisedLearning\rawdata"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("Loading dataset...")
    X, file_names = load_dataset(DATA_DIR)
    print("Shape sebelum scaling:", X.shape)

    print("Scaling features...")
    X_scaled, scaler = scale_features(X)

    print("Applying PCA...")
    X_pca, pca = apply_pca(X_scaled)

    print("Clustering (K=2)...")
    labels, kmeans = clustering_kmeans(X_pca)

    print("Detecting outliers...")
    outliers, iso = detect_outliers(X_pca)

    # ✅ Evaluation
    print("Evaluating clustering...")
    metrics = evaluate_clustering(X_pca, labels)

    print("\n=== CLUSTERING METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # ✅ Visualization
    print("Generating visualization...")
    plot_pca_clusters(
        X_pca,
        labels,
        outliers,
        save_path=os.path.join(OUTPUT_DIR, "pca_plot.png")
    )

    # Save outputs
    np.save(os.path.join(OUTPUT_DIR, "features.npy"), X_pca)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)
    np.save(os.path.join(OUTPUT_DIR, "outliers.npy"), outliers)

    print("\n=== SUMMARY ===")
    print(f"Total samples: {len(file_names)}")
    print(f"Cluster 0: {(labels==0).sum()}")
    print(f"Cluster 1: {(labels==1).sum()}")
    print(f"Outliers detected: {(outliers==-1).sum()}")


if __name__ == "__main__":
    main()