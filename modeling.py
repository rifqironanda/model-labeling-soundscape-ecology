from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# NEW
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def apply_pca(X, variance=0.95):
    pca = PCA(n_components=variance)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def clustering_kmeans(X, k=2):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    return labels, model


def detect_outliers(X, contamination=0.05):
    iso = IsolationForest(contamination=contamination, random_state=42)
    outliers = iso.fit_predict(X)
    return outliers, iso


def evaluate_clustering(X, labels):
    results = {}

    if len(set(labels)) > 1:  # avoid error if only 1 cluster
        results["silhouette"] = silhouette_score(X, labels)
        results["davies_bouldin"] = davies_bouldin_score(X, labels)
        results["calinski_harabasz"] = calinski_harabasz_score(X, labels)
    else:
        results["silhouette"] = None
        results["davies_bouldin"] = None
        results["calinski_harabasz"] = None

    return results