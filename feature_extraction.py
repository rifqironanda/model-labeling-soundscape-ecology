import numpy as np
import librosa

def aggregate(feat):
    return np.concatenate([
        np.mean(feat, axis=1),
        np.std(feat, axis=1),
        np.max(feat, axis=1),
        np.min(feat, axis=1)
    ])

def extract_features(file_path, sr=22050, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=sr)

        if y is None or len(y) == 0:
            return None

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        if mfcc.shape[1] == 0:
            return None

        # Spectral
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        spectral = np.vstack([spec_centroid, spec_bandwidth, spec_rolloff])

        # Temporal
        zcr = librosa.feature.zero_crossing_rate(y=y)
        rms = librosa.feature.rms(y=y)
        temporal = np.vstack([zcr, rms])

        # Aggregate
        mfcc_feat = aggregate(mfcc)
        spec_feat = aggregate(spectral)
        temp_feat = aggregate(temporal)

        features = np.concatenate([mfcc_feat, spec_feat, temp_feat])

        features = np.nan_to_num(features)

        if features.shape[0] != 72:
            return None

        return features

    except Exception as e:
        print(f"Error di file {file_path}: {e}")
        return None