from sklearn.preprocessing import StandardScaler

def scale_features(X):
    if X is None or len(X) == 0:
        raise ValueError("Dataset kosong!")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler