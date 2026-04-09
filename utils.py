import os
import numpy as np
from tqdm import tqdm
from feature_extraction import extract_features

def load_dataset(data_dir):
    features = []
    file_names = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                path = os.path.join(root, file)

                feat = extract_features(path)

                if feat is not None and feat.size > 0:
                    features.append(feat)
                    file_names.append(file)
                else:
                    print(f"Skip file: {path}")

    return np.array(features), file_names